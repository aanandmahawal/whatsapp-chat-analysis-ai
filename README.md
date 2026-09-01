# WhatsApp Chat Analyzer — with AI Chat Assistant

Upload an exported WhatsApp `.txt` chat, explore the analytics dashboard, and ask
natural-language questions about the conversation ("Who is the most active user?",
"What was Ujjwal's 3rd message?", "What is this group mainly about?").

## Project structure

```
whatsapp-chat-analyzer/
├── app.py               Streamlit UI: existing dashboard + new "AI Chat Assistant" tab
├── preprocessor.py      (unchanged) WhatsApp .txt -> Pandas DataFrame
├── helper.py            (unchanged) dashboard analytics
├── config.py            NEW  API-key resolution, model name, limits (no secrets inside)
├── ai_tools.py          NEW  Layer 1: deterministic Pandas tools + their JSON schemas
├── retrieval.py         NEW  Layer 2: in-memory TF-IDF retrieval for topic questions
├── ai_agent.py          NEW  Layer 3: Groq tool-calling loop
├── requirements.txt
├── .env.example         copy to .env locally (git-ignored)
├── .streamlit/secrets.toml.example
├── .gitignore
└── stop_hinglish.txt
```

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your real key in .env
streamlit run app.py
```

Get a key at https://console.groq.com/keys. `.env` is git-ignored — never commit it.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub **without** `.env` or `.streamlit/secrets.toml` (both are in `.gitignore`).
2. Create the app on https://share.streamlit.io and point it at `app.py`.
3. Open **App → ⋮ → Settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Save. Streamlit exposes this as `st.secrets["GROQ_API_KEY"]`; `config.get_api_key()`
   checks the environment variable first, then Streamlit secrets, so the same code works in both places.

The key is never printed, logged, or shown in the UI.

## How the AI assistant works

There is **no vector database and no LangChain**. The design is a *tool-calling agent*:

```
question ──► Groq LLM (sees 15 tool descriptions + a 300-token chat summary)
                │  picks a tool + arguments
                ▼
         ai_tools.py  (Pandas: counts, rankings, Nth message, date/time filters …)
         retrieval.py (TF-IDF: only for "what is this group about / find messages about X")
                │  exact JSON result (a few numbers or ≤ 30 messages)
                ▼
             Groq LLM ──► natural-language answer
```

- **Exact questions** (counts, rankings, first/last/Nth message, busiest day, comparisons,
  emojis, links, media, keyword search) are computed by Pandas. The model never sees the
  whole chat and never estimates numbers.
- **Understanding questions** ("what is this group about?", "summarize discussions about
  internships") use `get_topic_overview` / `semantic_search`, which return a *small, relevant
  subset* of real messages for the model to read. Answers separate facts from inference.
- **Follow-ups** ("how many messages did *he* send?") work because the last 12 turns of the
  conversation are sent along with the question.
- Every answer has an expandable **"How was this computed?"** panel showing which tools ran.

### Why TF-IDF instead of embeddings + a vector DB

The corpus lives for one Streamlit session and is rebuilt on every upload. A persistent
vector store solves a problem this app doesn't have, and dense embedding models add
hundreds of MB of dependencies. Word + character n-gram TF-IDF builds in well under a
second, needs no downloads, and handles Hinglish spelling variation reasonably. If you ever
need true synonym matching, swap `MessageIndex._scores()` for a sentence-transformers
encoder — the rest of the interface is unchanged.

### Dependencies added (and why)

| Package | Why |
|---|---|
| `groq` | Official Groq SDK — LLM calls with tool calling |
| `scikit-learn` | `TfidfVectorizer` + cosine similarity for `retrieval.py` |
| `python-dotenv` | Reads `GROQ_API_KEY` from `.env` locally (unused on Streamlit Cloud) |

## Privacy

- The full chat file never leaves your machine. Only the question, recent Q&A turns, and the
  tool results (a handful of numbers or a few dozen messages) are sent to Groq.
- Nothing is written to disk or logged.
- Groq's data policy: https://console.groq.com/docs/your-data

## Configuration

All settings live in `config.py` and can be overridden with environment variables
(or Streamlit secrets): `GROQ_MODEL`, `GROQ_TEMPERATURE`, `MAX_TOOL_ROUNDS`,
`MAX_HISTORY_MESSAGES`, `MAX_TOOL_RESULT_CHARS`, `MAX_INDEXED_MESSAGES`.

Groq deprecates models periodically — if you get a "model not found" error, pick a
current one from https://console.groq.com/docs/models and set `GROQ_MODEL`.
