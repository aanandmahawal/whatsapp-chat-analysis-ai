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
from datetime import date
from typing import Optional

import groq

import config
from ai_tools import TOOL_SPECS, ChatTools, run_tool

SYSTEM_PROMPT_TEMPLATE = """You are an analyst assistant for ONE uploaded WhatsApp chat export. You answer questions about this chat only.

## This chat
- Participants ({n_users}): {users}
- Date range: {first_date} to {last_date} ({total_messages} messages, {media} media, {notifications} group notifications)
- Today's date: {today}

## How to work
- You have deterministic tools. ALWAYS call a tool for any count, ranking, percentage, date, position ("Nth message"), first/last message, or exact quote. Never estimate these yourself and never reuse an earlier tool result if the question needs a fresh computation.
- Choose the tool by question type:
  * totals / summary / who is in the chat -> get_chat_statistics
  * rankings ("most active", "who talks most", "most emojis/links/media") -> get_most_active_users
  * one person's numbers, or comparing two people -> get_user_statistics (once per person)
  * "how many messages ... in January / on Sundays / between 10 PM and midnight / containing X" -> count_messages
  * "Nth / first / last / latest message" -> get_nth_message
  * "show X's first 5 messages", "what happened on/around a date", "longest message" -> get_messages
  * busiest day / month / hour, "what day of the week / time is the group most active" -> get_activity_breakdown
  * a specific word or phrase -> search_messages
  * a topic, theme, "what did people say about Y", "summarize conversations about Y" -> semantic_search, then get_message_context on interesting message_ids
  * "what is this group about / which field / main topics / kind of conversations / interesting insights" -> get_topic_overview (optionally + semantic_search on candidate topics)
- Relative time phrases ("last week", "in March") must be converted to concrete dates/months using today's date and the chat's date range before calling a tool.
- User names: pass names as the person wrote them; tools resolve partial names and case. If a tool reports a name is unknown or ambiguous, tell the person clearly and list the candidates. Always write usernames exactly as they appear in the chat.
- Follow-ups: resolve pronouns and references ("he", "she", "that user", "what about Rahul?") from the conversation so far. Same question type, new subject.

## Accuracy rules (strict)
- Never invent messages, numbers, dates, users or rankings. Every fact you state must come from a tool result in this conversation.
- Quote messages verbatim from tool results; do not paraphrase a quote as if it were verbatim.
- If a tool returns found=false, count=0 or an error, say plainly that the information is not available in the chat (e.g. "Ujjwal is not a participant in this chat" or "Ujjwal only sent 2 messages, so there is no 3rd message").
- Media placeholders are not text; say the message was a media file.
- "group_notification" is the system (joins, leaves, subject changes), not a participant. Never rank or count it as a user.
- For evidence-based answers (what someone said, what was discussed) cite the sender and date of the messages you rely on.
- For topic / field / summary questions: clearly separate facts from interpretation. Say "the group appears to be about ... based on repeated messages about ..." and name the supporting themes. If the evidence is thin or mixed, say so instead of guessing. Keep such answers to about 4-5 lines unless asked for more.
- If asked for a specific number of lines or items, respect it.

## Style
- Lead with the direct answer, then brief supporting evidence. Plain sentences; a short list only when listing several messages or items.
- Do not mention tool names or internal mechanics.
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SPECS,
                # On the last allowed round force a text answer instead of more tool calls.
                tool_choice="none" if final_round else "auto",
                temperature=config.TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            )
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
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str) and m["content"].strip()
    ]
    h = h[-config.MAX_HISTORY_MESSAGES:]
    while h and h[0]["role"] != "user":
        h = h[1:]
    return h


def friendly_api_error(err: Exception) -> str:
    """Turn Groq SDK exceptions into a message that is safe to show in the UI."""
    if isinstance(err, groq.AuthenticationError):
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file or Streamlit secrets."
    if isinstance(err, groq.PermissionDeniedError):
        return "Groq refused the request (403). The key may lack access to this model, or your network blocks api.groq.com."
    if isinstance(err, groq.RateLimitError):
        return "Groq's rate limit was hit. Wait a few seconds and try again (free-tier limits are per minute and per day)."
    if isinstance(err, groq.APIConnectionError):
        return "Could not reach the Groq API. Check your internet connection."
    if isinstance(err, groq.NotFoundError):
        return f"The model '{config.GROQ_MODEL}' was not found. Set GROQ_MODEL to a model listed at console.groq.com/docs/models."
    if isinstance(err, groq.BadRequestError):
        detail = getattr(err, "message", None) or str(err)
        if "context" in detail.lower() or "token" in detail.lower():
            return "The request was too large for the model's context window. Try a narrower question or start a new conversation."
        return f"Groq rejected the request: {detail[:300]}"
    if isinstance(err, groq.APIStatusError):
        return f"Groq API error ({err.status_code}). Please try again in a moment."
    return f"Unexpected error: {type(err).__name__}: {err}"
