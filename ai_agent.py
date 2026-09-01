"""
ai_agent.py — Layer 3: the Groq-powered agent that answers questions about one chat.

Flow for a single question
--------------------------
  1. Build a system prompt describing THIS chat (users, date range, totals)
     plus the accuracy rules.
  2. Send: system prompt + recent conversation history + the new question +
     the tool catalogue (ai_tools.TOOL_SPECS) to Groq.
  3. If the model replies with tool_calls, run each in Python
     (ai_tools.run_tool), append the JSON results, and call Groq again.
     This loop is what makes it an "agent": the LLM decides WHICH
     deterministic function to run, Python computes, the LLM reads the
     result and decides whether it needs more.
  4. When the model answers in plain text, return it together with a trace
     of the tool calls (shown in the UI's "How was this computed?" panel).

There is no separate "classify the question" step: choosing a tool IS the
classification. The model reads the tool descriptions and picks e.g.
get_nth_message for a positional question or semantic_search for a topic
question — more robust than a hand-written rule set.

Privacy
-------
Only what is needed leaves the machine: the question, recent Q&A turns, the
tool *results* (a few numbers / at most a few dozen messages) and the small
chat summary in the system prompt. The full chat file is never sent.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Optional

import groq

import config
from ai_tools import TOOL_SPECS, ChatTools, run_tool

SYSTEM_PROMPT_TEMPLATE = """You are an analyst for ONE uploaded WhatsApp chat. Answer only about this chat.

Chat: {n_users} participants ({users}); {first_date} to {last_date}; {total_messages} messages, {media} media, {notifications} system notifications. Today: {today}.

Rules:
- ALWAYS call a tool for any count, ranking, date, position (Nth/first/last), or quote. Never estimate or reuse stale results.
- Routing: "summary / overview of this chat" -> get_topic_overview ONLY (it already includes totals); plain totals or "how many users" -> get_chat_statistics; rankings -> get_most_active_users; one user or comparing two -> get_user_statistics; "how many ... in March / on Sundays / 10PM-midnight" -> count_messages; Nth/first/last/second-last -> get_nth_message; list messages / on a date / longest -> get_messages; busiest day/hour/weekday -> get_activity_breakdown; exact word -> search_messages; topic/theme -> semantic_search (+ get_message_context); "what is this chat about / field / insights" -> get_topic_overview.
- Convert relative time ("last week", "in March") to concrete dates using today's date and the chat's range.
- Names: pass as written; tools resolve partial names. If a tool says unknown/ambiguous, say so and list candidates. Write names exactly as in the chat.
- Follow-ups: resolve "he/she/that user" from earlier turns.
- Never invent messages, numbers, dates or users. If a tool returns found=false, count=0 or an error, say the information is not in the chat. Quote messages verbatim. Media placeholders are media files, not text. group_notification is the system, never a participant.
- Topic/field/summary answers: separate fact from inference ("appears to be about ... based on repeated messages about ..."), cite themes, note thin evidence, keep to 4-5 lines unless asked for more.
- Style: direct answer first, then brief evidence; plain sentences; never mention tool names.
"""


def build_system_prompt(tools: ChatTools) -> str:
    stats = tools.get_chat_statistics()
    users = tools.users
    shown = users if len(users) <= 40 else users[:40] + [f"... and {len(users) - 40} more (see get_chat_statistics)"]
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_users=len(users),
        users=", ".join(shown),
        first_date=stats["first_message_date"],
        last_date=stats["last_message_date"],
        total_messages=stats["total_messages"],
        media=stats["media_messages"],
        notifications=stats["group_notifications"],
        today=date.today().isoformat(),
    )


class ChatAgent:
    def __init__(
        self,
        tools: ChatTools,
        api_key: Optional[str] = None,
        model: str = config.GROQ_MODEL,
        max_tool_rounds: int = config.MAX_TOOL_ROUNDS,
        client: Optional[groq.Groq] = None,
    ):
        self.tools = tools
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.client = client or groq.Groq(api_key=api_key)  # key is never logged or stored elsewhere
        self.system_prompt = build_system_prompt(tools)

    # ------------------------------------------------------------------
    def _call(self, messages: list[dict], tool_choice: str):
        """One Groq call with free-tier recoveries.
        413 (request too large) is retried up to three times, each time smaller:
          1. shrink the largest tool result to a third
          2. shrink EVERY tool result to ~500 chars and drop conversation history
          3. keep only the system prompt, the question and a compact note of the tool results
        429 with a short 'try again in Ns' waits that long and retries once."""
        max_out = config.MAX_COMPLETION_TOKENS

        def create():
            return self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SPECS, tool_choice=tool_choice,
                temperature=config.TEMPERATURE, max_completion_tokens=max_out,
            )

        for attempt in range(4):
            try:
                return create()
            except groq.APIStatusError as err:
                if err.status_code == 413 and attempt < 3:
                    max_out = config.RETRY_COMPLETION_TOKENS
                    _shrink_messages(messages, level=attempt + 1)
                    continue
                if err.status_code == 429 and attempt == 0:
                    wait = _retry_after_seconds(getattr(err, "message", str(err)))
                    if wait is not None and wait <= config.AUTO_RETRY_WAIT_SECONDS:
                        time.sleep(wait + 0.5)
                        continue
                raise

    # ------------------------------------------------------------------
    def ask(self, question: str, history: list[dict]) -> tuple[str, list[dict]]:
        """
        question : the new user message
        history  : previous turns as [{"role": "user"|"assistant", "content": str}, ...]
        returns  : (answer_text, trace) where trace lists every tool call made.
        Raises groq.* exceptions on API failure — the UI maps them to friendly text.
        """
        messages: list[dict] = (
            [{"role": "system", "content": self.system_prompt}]
            + _trim_history(history)
            + [{"role": "user", "content": question}]
        )
        trace: list[dict] = []

        for _round in range(self.max_tool_rounds + 1):
            final_round = _round == self.max_tool_rounds
            response = self._call(messages, tool_choice="none" if final_round else "auto")
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                answer = (msg.content or "").strip()
                if final_round and answer:
                    answer += "\n\n(I reached my analysis-step limit; the answer above may be incomplete.)"
                return answer or "I couldn't produce an answer for that. Could you rephrase?", trace

            # Record the assistant's tool request, then run every requested tool.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in tool_calls
                ],
            })
            for call in tool_calls:
                args = _parse_args(call.function.arguments)
                output = run_tool(self.tools, call.function.name, args) if args is not None \
                    else json.dumps({"error": "Tool arguments were not valid JSON."})
                trace.append({
                    "tool": call.function.name, "input": args,
                    "output": output[:2000], "error": output.startswith('{"error"'),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": output,
                })

        return "I ran out of analysis steps before finishing. Try a narrower question.", trace


def _parse_args(raw) -> Optional[dict]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return None


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last N plain-text turns, always starting with a user turn."""
    h = [
        {"role": m["role"], "content": m["content"][: config.MAX_HISTORY_CHARS]}
        for m in history
        if m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str) and m["content"].strip()
    ]
    h = h[-config.MAX_HISTORY_MESSAGES:]
    while h and h[0]["role"] != "user":
        h = h[1:]
    return h


def _shrink_messages(messages: list[dict], level: int) -> None:
    """In-place size reduction used after a 413. Preserves tool_call <-> tool_result pairing."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if level == 1 and tool_idx:
        i = max(tool_idx, key=lambda k: len(messages[k]["content"]))   # the largest result
        c = messages[i]["content"]
        messages[i]["content"] = c[: max(800, len(c) // 3)] + "... [truncated: too large for the API limit]"
        return
    if level == 2:
        for i in tool_idx:
            c = messages[i]["content"]
            if len(c) > 500:
                messages[i]["content"] = c[:500] + "... [truncated]"
        # drop plain-text conversation history (keep system, the last user question and tool traffic)
        first_tool = tool_idx[0] if tool_idx else len(messages)
        keep_head = [messages[0]] + [m for m in messages[1:first_tool] if m.get("role") == "user"][-1:]
        messages[:] = keep_head + messages[first_tool:]
        return
    # level 3: collapse everything after the question into one compact note
    system, question = messages[0], next((m for m in messages if m.get("role") == "user"), None)
    notes = "\n".join(f"- {m.get('name', 'tool')}: {m['content'][:300]}" for m in messages if m.get("role") == "tool")
    messages[:] = [system] + ([question] if question else []) + [
        {"role": "user", "content": "Tool results gathered so far (compact):\n" + notes +
                                    "\nAnswer the question from these results only; do not call more tools."}
    ]


def _retry_after_seconds(text: str):
    """Groq's 429 message ends with e.g. 'Please try again in 6.5s' or 'in 2m3.1s'."""
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)s", text or "")
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def friendly_api_error(err: Exception) -> str:
    """Turn Groq SDK exceptions into a message that is safe to show in the UI."""
    if isinstance(err, groq.AuthenticationError):
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file or Streamlit secrets."
    if isinstance(err, groq.PermissionDeniedError):
        return "Groq refused the request (403). The key may lack access to this model, or your network blocks api.groq.com."
    if isinstance(err, groq.RateLimitError):
        wait = _retry_after_seconds(getattr(err, "message", str(err)))
        when = f"in about {int(wait) + 1} seconds" if wait else "in a minute"
        return (f"Groq's free-tier rate limit was hit (8,000 tokens per minute). Please try again {when}. "
                "Tip: ask one question at a time; upgrading to Groq's Developer tier removes this limit.")
    if isinstance(err, groq.APIConnectionError):
        return "Could not reach the Groq API. Check your internet connection."
    if isinstance(err, groq.NotFoundError):
        return f"The model '{config.GROQ_MODEL}' was not found. Set GROQ_MODEL to a model listed at console.groq.com/docs/models."
    if isinstance(err, groq.BadRequestError):
        detail = getattr(err, "message", None) or str(err)
        if "context" in detail.lower() or "token" in detail.lower():
            return "The request was too large for the model's context window. Try a narrower question or start a new conversation."
        return f"Groq rejected the request: {detail[:300]}"
    if isinstance(err, groq.APIStatusError) and err.status_code == 413:
        return ("This request was too large for Groq's free-tier limit (8,000 tokens per minute). "
                "Try a narrower question, or clear the chat history and ask again.")
    if isinstance(err, groq.APIStatusError):
        return f"Groq API error ({err.status_code}). Please try again in a moment."
    return f"Unexpected error: {type(err).__name__}: {err}"
