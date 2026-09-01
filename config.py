"""
config.py — all configuration for the AI Chat Assistant lives here.

Nothing in this file is secret. The Groq API key is *read* here (from an
environment variable, a local .env file, or Streamlit secrets) but is never
written into source code and never displayed in the UI.

Every value can be overridden with an environment variable of the same name,
e.g.  GROQ_MODEL=openai/gpt-oss-20b
"""
from __future__ import annotations

import os

# Load a local .env file if python-dotenv is installed. Harmless in production
# (Streamlit Cloud has no .env; secrets come from st.secrets instead).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------

# Production model on Groq with strong tool-calling. Cheaper/faster alternative
# with the same API: "openai/gpt-oss-20b". Check console.groq.com/docs/models
# for the current list — Groq deprecates models periodically.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Low temperature = more reliable tool calls and fewer invented details.
TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# Max tokens Groq may generate per call (the answer, not the tool results).
MAX_COMPLETION_TOKENS = _env_int("GROQ_MAX_COMPLETION_TOKENS", 1500)

# How many "think -> call tool -> read result" rounds one question may take.
# Most questions need 1-2; topic/summary questions may need 3-5.
MAX_TOOL_ROUNDS = _env_int("MAX_TOOL_ROUNDS", 8)

# How many previous chat turns (user + assistant messages) are sent back so
# follow-ups like "how many messages did he send?" work.
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 12)


def get_api_key() -> str | None:
    """
    Resolve the Groq API key, in priority order:
      1. GROQ_API_KEY environment variable (also populated from .env locally)
      2. st.secrets["GROQ_API_KEY"]         (Streamlit Community Cloud)
    Returns None if neither is set; the UI shows setup help instead of crashing.
    """
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        import streamlit as st  # lazy import so config.py works outside Streamlit
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip() or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tool output limits (keep every tool result small enough for the context)
# ---------------------------------------------------------------------------

# Individual messages longer than this are truncated inside tool results.
MAX_MESSAGE_CHARS = _env_int("MAX_MESSAGE_CHARS", 400)

# Hard cap on the size of any single tool result sent to the LLM (characters).
MAX_TOOL_RESULT_CHARS = _env_int("MAX_TOOL_RESULT_CHARS", 14000)

# Default and maximum number of rows a list-type tool may return.
DEFAULT_LIST_LIMIT = _env_int("DEFAULT_LIST_LIMIT", 20)
MAX_LIST_LIMIT = _env_int("MAX_LIST_LIMIT", 100)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Path to the Hinglish stop-word list that already ships with the project.
STOPWORDS_PATH = os.getenv("STOPWORDS_PATH", "stop_hinglish.txt")

# Chats with more text messages than this skip building the semantic index
# (keeps memory/CPU bounded on Streamlit's free tier). Keyword search still works.
MAX_INDEXED_MESSAGES = _env_int("MAX_INDEXED_MESSAGES", 150_000)
