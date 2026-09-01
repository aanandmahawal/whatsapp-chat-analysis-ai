"""
ai_tools.py — Layer 1: deterministic analytics tools over the chat DataFrame.

Every public method on ChatTools:
  * takes plain Python arguments (strings / ints / bools),
  * computes its answer with Pandas (never with an LLM),
  * returns a JSON-serialisable dict.

The agent (ai_agent.py) exposes these methods to Groq as "tools". The LLM
decides WHICH tool to call and with WHAT arguments; Python does the actual
computing. That is what keeps counts, rankings and "Nth message" answers exact.

Reuse of the existing project (helper.py / preprocessor.py are NOT modified):
  * the DataFrame produced by preprocessor.preprocess()
  * helper.extract       -> the shared URLExtract instance for link detection
  * helper.emoji_helper  -> emoji counting
  * stop_hinglish.txt    -> stop-word list (loaded here as a proper set)
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urlparse

import emoji
import pandas as pd

import config
import helper

NOTIFICATION_USER = "group_notification"

# Text patterns WhatsApp uses for non-textual / placeholder messages (Android + iOS).
_MEDIA_PATTERNS = re.compile(
    r"^(<media omitted>|image omitted|video omitted|audio omitted|sticker omitted|"
    r"gif omitted|document omitted|contact card omitted|<attached: .+>)$",
    re.IGNORECASE,
)
_DELETED_PATTERNS = re.compile(
    r"^(this message was deleted|you deleted this message|null)$", re.IGNORECASE
)
_URL_HINT = re.compile(r"\w\.\w")  # cheap pre-filter before running URLExtract
_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class ToolError(Exception):
    """Raised for user-fixable problems (unknown user, bad date...). The agent
    forwards the message to the LLM so it can explain or ask a clarifying question."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _load_stopwords(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def _jsonable(obj: Any) -> Any:
    """Convert numpy/pandas scalars & timestamps so json.dumps() works."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M")
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def _clamp_limit(limit: Optional[int], default: int = config.DEFAULT_LIST_LIMIT) -> int:
    if limit is None:
        return default
    return max(1, min(int(limit), config.MAX_LIST_LIMIT))


def _parse_date(value: Optional[str], end_of_day: bool = False) -> Optional[pd.Timestamp]:
    if value in (None, "", "null"):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        raise ToolError(f"Could not understand date '{value}'. Use YYYY-MM-DD.")
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _parse_month(value) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, int) or str(value).isdigit():
        m = int(value)
        if 1 <= m <= 12:
            return m
        raise ToolError(f"Month number must be 1-12, got {value}.")
    name = str(value).strip().lower()
    for i, full in enumerate(_MONTHS, start=1):
        if full.startswith(name[:3]):
            return i
    raise ToolError(f"Unknown month '{value}'.")


def _parse_weekday(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    name = str(value).strip().lower()
    match = [w for w in _WEEKDAYS if w.startswith(name[:3])]
    if not match:
        raise ToolError(f"Unknown weekday '{value}'.")
    return match[0].capitalize()


# ---------------------------------------------------------------------------
# The tool class
# ---------------------------------------------------------------------------

class ChatTools:
    def __init__(self, df: pd.DataFrame, stopwords_path: str = config.STOPWORDS_PATH):
        self.df = self._enrich(df)          # working copy with helper columns
        self.stopwords = _load_stopwords(stopwords_path)
        self.users: list[str] = sorted(
            self.df.loc[~self.df["is_notification"], "user"].unique().tolist()
        )
        self.index = None                   # retrieval.MessageIndex, attached by the app
        if self.df.empty:
            raise ToolError("The chat contains no parseable messages.")

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # The original index == position in the exported file. Keep it as a stable
        # id so chronological ties (same minute) are broken by file order.
        d["message_id"] = d.index.astype(int)
        d = d[d["date"].notna()]
        # Index == message_id so .loc[message_id] works; the index itself stays
        # unnamed to avoid pandas' "both index level and column" ambiguity.
        d.index = pd.Index(d["message_id"].to_numpy(), name=None)
        d["text"] = (
            d["message"].astype(str)
            .str.replace("\u200e", "", regex=False)   # WhatsApp's invisible LRM mark
            .str.strip()
        )
        d["is_notification"] = d["user"] == NOTIFICATION_USER
        d["is_media"] = d["text"].str.match(_MEDIA_PATTERNS)
        d["is_deleted"] = d["text"].str.match(_DELETED_PATTERNS)
        d["is_text"] = ~(d["is_notification"] | d["is_media"] | d["is_deleted"])
        d["word_count"] = d["text"].str.split().str.len().fillna(0).astype(int)
        d.loc[~d["is_text"], "word_count"] = 0
        d["char_count"] = d["text"].str.len()

        def _links(t: str) -> list[str]:
            return helper.extract.find_urls(t) if _URL_HINT.search(t) else []

        d["links"] = d["text"].apply(_links)
        d["n_links"] = d["links"].str.len()
        return d

    # --------------------------------------------------------------- helpers
    def _fmt(self, row: pd.Series) -> dict:
        text = row["text"]
        if row["is_media"]:
            text = "<media message (image/video/audio/document)>"
        elif row["is_deleted"]:
            text = "<deleted message>"
        elif len(text) > config.MAX_MESSAGE_CHARS:
            text = text[: config.MAX_MESSAGE_CHARS] + "…"
        return {
            "message_id": int(row["message_id"]),
            "date": row["date"].strftime("%Y-%m-%d %H:%M"),
            "day_name": row["day_name"],
            "user": row["user"],
            "message": text,
            "is_media": bool(row["is_media"]),
        }

    def _rows(self, frame: pd.DataFrame) -> list[dict]:
        return [self._fmt(r) for _, r in frame.iterrows()]

    @staticmethod
    def _chrono(frame: pd.DataFrame, newest_first: bool = False) -> pd.DataFrame:
        """Chronological order; file order breaks ties for identical timestamps."""
        return frame.sort_values(["date", "message_id"], ascending=not newest_first, kind="stable")

    def _resolve(self, name: Optional[str]) -> Optional[str]:
        """Map a user-typed name to the exact username in the chat."""
        if name in (None, "", "Overall", "overall", "all", "everyone", "group"):
            return None
        name = str(name).strip()
        if name in self.users:
            return name
        lower = {u.lower(): u for u in self.users}
        if name.lower() in lower:
            return lower[name.lower()]
        partial = [u for u in self.users if name.lower() in u.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise ToolError(f"'{name}' matches several users: {partial}. Ask which one is meant.")
        fuzzy = difflib.get_close_matches(name.lower(), list(lower.keys()), n=3, cutoff=0.75)
        if len(fuzzy) == 1:
            return lower[fuzzy[0]]
        if fuzzy:
            raise ToolError(f"'{name}' is ambiguous; closest users: {[lower[f] for f in fuzzy]}.")
        raise ToolError(
            f"No user named '{name}' in this chat. Known users: {self.users[:40]}"
        )

    def _filter(
        self,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year: Optional[int] = None,
        month=None,
        day_name: Optional[str] = None,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        include_media: bool = True,
        text_only: bool = False,
    ) -> pd.DataFrame:
        d = self.df[~self.df["is_notification"]]
        u = self._resolve(user)
        if u:
            d = d[d["user"] == u]
        s, e = _parse_date(start_date), _parse_date(end_date, end_of_day=True)
        if s is not None:
            d = d[d["date"] >= s]
        if e is not None:
            d = d[d["date"] <= e]
        if year is not None:
            d = d[d["year"] == int(year)]
        m = _parse_month(month)
        if m is not None:
            d = d[d["month_num"] == m]
        wd = _parse_weekday(day_name)
        if wd:
            d = d[d["day_name"] == wd]
        if start_hour is not None or end_hour is not None:
            sh = 0 if start_hour is None else int(start_hour)
            eh = 24 if end_hour is None else int(end_hour)
            if not (0 <= sh <= 23 and 0 <= eh <= 24):
                raise ToolError("Hours must be 0-23 (end_hour may be 24 for midnight).")
            if sh < eh:
                d = d[(d["hour"] >= sh) & (d["hour"] < eh)]
            else:  # wraps past midnight, e.g. 22 -> 2
                d = d[(d["hour"] >= sh) | (d["hour"] < eh)]
        if not include_media:
            d = d[~d["is_media"]]
        if text_only:
            d = d[d["is_text"]]
        return d

    @staticmethod
    def _top(d: pd.DataFrame, col: str):
        if d.empty:
            return None
        vc = d[col].value_counts()
        return {"period": _jsonable(vc.index[0]), "messages": int(vc.iloc[0])}

    def _common_words(self, d: pd.DataFrame, limit: int) -> list[list]:
        words: Counter = Counter()
        for msg in d.loc[d["is_text"], "text"]:
            for w in msg.lower().split():
                w = w.strip(".,!?;:\"'()[]{}")
                if (w and w not in self.stopwords and not w.startswith("http")
                        and not all(emoji.is_emoji(ch) for ch in w)):
                    words[w] += 1
        return [[w, c] for w, c in words.most_common(limit)]

    @staticmethod
    def _emoji_counter(d: pd.DataFrame) -> Counter:
        frame = pd.DataFrame({"message": d.loc[d["is_text"], "text"].tolist()})
        if frame.empty:
            return Counter()
        e = helper.emoji_helper("Overall", frame)  # reuse the existing project logic
        return Counter(dict(zip(e["Emoji"], e["Count"])))

    # ============================================================ TOOLS ====

    # ---- statistics ------------------------------------------------------
    def get_chat_statistics(self) -> dict:
        d = self.df[~self.df["is_notification"]]
        first, last = self.df["date"].min(), self.df["date"].max()
        counts = d["user"].value_counts()
        return {
            "total_messages": int(len(d)),
            "text_messages": int(d["is_text"].sum()),
            "media_messages": int(d["is_media"].sum()),
            "deleted_messages": int(d["is_deleted"].sum()),
            "group_notifications": int(self.df["is_notification"].sum()),
            "total_users": len(self.users),
            "total_words": int(d["word_count"].sum()),
            "total_links": int(d["n_links"].sum()),
            "first_message_date": first.strftime("%Y-%m-%d"),
            "last_message_date": last.strftime("%Y-%m-%d"),
            "duration_days": int((last - first).days) + 1,
            "active_days": int(d["only_date"].nunique()),
            "users": [{"user": u, "messages": int(c)} for u, c in counts.head(40).items()],
        }

    def get_user_statistics(self, user: str) -> dict:
        u = self._resolve(user)
        if u is None:
            raise ToolError("get_user_statistics needs a specific user name.")
        d = self._chrono(self.df[self.df["user"] == u])
        all_msgs = self.df[~self.df["is_notification"]]
        total = int(len(all_msgs))
        rank = list(all_msgs["user"].value_counts().index).index(u) + 1
        text_only = d[d["is_text"]]
        return {
            "user": u,
            "message_count": int(len(d)),
            "percentage_of_total": round(len(d) / total * 100, 2) if total else 0,
            "rank_by_messages": rank,
            "of_users": len(self.users),
            "word_count": int(d["word_count"].sum()),
            "avg_words_per_text_message": round(text_only["word_count"].mean(), 2) if len(text_only) else 0,
            "media_count": int(d["is_media"].sum()),
            "link_count": int(d["n_links"].sum()),
            "emoji_count": int(sum(self._emoji_counter(d).values())),
            "first_message": self._fmt(d.iloc[0]),
            "last_message": self._fmt(d.iloc[-1]),
            "most_used_words": self._common_words(d, 10),
            "most_used_emojis": self._emoji_counter(d).most_common(5),
            "busiest_weekday": self._top(d, "day_name"),
            "busiest_month": self._top(d, "month"),
            "busiest_hour": self._top(d, "hour"),
        }

    def get_most_active_users(self, limit: int = 10, by: str = "messages") -> dict:
        d = self.df[~self.df["is_notification"]]
        by = (by or "messages").lower()
        if by == "messages":
            s = d["user"].value_counts()
        elif by == "words":
            s = d.groupby("user")["word_count"].sum().sort_values(ascending=False)
        elif by == "media":
            s = d.groupby("user")["is_media"].sum().sort_values(ascending=False)
        elif by == "links":
            s = d.groupby("user")["n_links"].sum().sort_values(ascending=False)
        elif by == "emojis":
            s = pd.Series({u: sum(self._emoji_counter(g).values()) for u, g in d.groupby("user")})
            s = s.sort_values(ascending=False)
        else:
            raise ToolError("by must be one of: messages, words, media, links, emojis")
        s = s[s > 0]
        total = int(s.sum())
        rows = [
            {"rank": i, "user": u, by: int(c), "percentage": round(c / total * 100, 2) if total else 0}
            for i, (u, c) in enumerate(s.head(_clamp_limit(limit)).items(), start=1)
        ]
        return {"ranked_by": by, "total": total, "users": rows}

    # ---- message retrieval ----------------------------------------------
    def get_nth_message(
        self, n: int = 1, user: Optional[str] = None, from_end: bool = False, include_media: bool = True,
    ) -> dict:
        """1-based positional lookup. n=1 -> first; from_end=True, n=1 -> last."""
        n = int(n)
        if n < 1:
            raise ToolError("n must be 1 or greater (1 = first message).")
        u = self._resolve(user)
        d = self._chrono(self._filter(user=u, include_media=include_media))
        scope = u or "the whole chat"
        if n > len(d):
            return {
                "found": False, "user": u, "requested_n": n,
                "detail": f"{scope} has only {len(d)} messages"
                          + ("" if include_media else " (excluding media)") + ".",
            }
        row = d.iloc[-n] if from_end else d.iloc[n - 1]
        return {"found": True, "user": u, "n": n, "from_end": bool(from_end),
                "of": int(len(d)), "message": self._fmt(row)}

    def get_messages(
        self,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        newest_first: bool = False,
        sort_by: str = "chronological",
        include_media: bool = True,
    ) -> dict:
        d = self._filter(user=user, start_date=start_date, end_date=end_date, include_media=include_media)
        sort_by = (sort_by or "chronological").lower()
        if sort_by == "longest":
            d = d[d["is_text"]].sort_values(["char_count", "message_id"], ascending=[False, True])
        else:
            d = self._chrono(d, newest_first=newest_first)
        limit, offset = _clamp_limit(limit), max(0, int(offset or 0))
        page = d.iloc[offset: offset + limit]
        out = {
            "user": self._resolve(user), "start_date": start_date, "end_date": end_date,
            "sort_by": sort_by, "total_matching": int(len(d)),
            "offset": offset, "returned": int(len(page)),
            "messages": self._rows(page), "truncated": bool(len(d) > offset + limit),
        }
        if sort_by == "longest":
            out["messages"] = [
                {**m, "characters": int(r["char_count"]), "words": int(r["word_count"])}
                for m, (_, r) in zip(out["messages"], page.iterrows())
            ]
        if self._resolve(user) is None:
            out["messages_per_user_top10"] = d["user"].value_counts().head(10).to_dict()
        return out

    def get_message_context(self, message_id: int, before: int = 5, after: int = 5) -> dict:
        """Messages surrounding a given message_id (chronological)."""
        mid = int(message_id)
        if mid not in self.df.index:
            raise ToolError(f"message_id {mid} does not exist.")
        pos = self.df.index.get_loc(mid)
        before, after = min(int(before), 30), min(int(after), 30)
        window = self.df.iloc[max(0, pos - before): pos + after + 1]
        return {"target_message_id": mid, "messages": self._rows(window)}

    # ---- counting & searching ------------------------------------------
    def count_messages(
        self,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        year: Optional[int] = None,
        month=None,
        day_name: Optional[str] = None,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        keyword: Optional[str] = None,
        include_media: bool = True,
    ) -> dict:
        d = self._filter(user, start_date, end_date, year, month, day_name, start_hour, end_hour, include_media)
        if keyword:
            d = d[d["text"].str.contains(re.escape(str(keyword)), case=False, na=False)]
        filters = dict(user=self._resolve(user), start_date=start_date, end_date=end_date, year=year,
                       month=month, day_name=day_name, start_hour=start_hour, end_hour=end_hour,
                       keyword=keyword, include_media=include_media)
        out = {
            "count": int(len(d)),
            "filters": _jsonable({k: v for k, v in filters.items() if v not in (None, "", True)}),
        }
        if self._resolve(user) is None:
            vc = d["user"].value_counts().head(10)
            out["per_user_top10"] = vc.to_dict()
            out["most_active_user"] = ({"user": vc.index[0], "messages": int(vc.iloc[0])} if len(vc) else None)
        return out

    def search_messages(
        self, keyword: str, user: Optional[str] = None, limit: int = 20,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> dict:
        if not keyword or not str(keyword).strip():
            raise ToolError("keyword must not be empty.")
        d = self._filter(user=user, start_date=start_date, end_date=end_date, text_only=True)
        hits = self._chrono(d[d["text"].str.contains(re.escape(str(keyword).strip()), case=False, na=False)])
        limit = _clamp_limit(limit)
        return {
            "keyword": keyword, "user": self._resolve(user),
            "total_matches": int(len(hits)),
            "matches_per_user": hits["user"].value_counts().head(10).to_dict(),
            "messages": self._rows(hits.head(limit)),
            "truncated": bool(len(hits) > limit),
        }

    def semantic_search(
        self, query: str, user: Optional[str] = None, limit: int = 15,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> dict:
        if self.index is None or self.index.size == 0:
            raise ToolError("Semantic search is not available for this chat; use search_messages with keywords instead.")
        u = self._resolve(user)
        s, e = _parse_date(start_date), _parse_date(end_date, end_of_day=True)
        hits = self.index.search(query, k=_clamp_limit(limit), user=u, start=s, end=e)
        return {
            "query": query, "user": u, "returned": len(hits),
            "note": "Ranked by relevance, not by date. Use get_message_context on a message_id to read the surrounding conversation.",
            "messages": [
                {**self._fmt(self.df.loc[h["message_id"]]), "relevance": round(h["score"], 3)}
                for h in hits
            ],
        }

    def get_topic_overview(self, sample_size: int = 60) -> dict:
        """Evidence pack for 'what is this group about?': distinctive terms + representative messages."""
        if self.index is None or self.index.size == 0:
            raise ToolError("Topic overview needs text messages to analyse; this chat has none indexed.")
        sample_ids = self.index.representative_sample(min(int(sample_size), 150))
        stats = self.get_chat_statistics()
        sample = self._chrono(self.df.loc[sample_ids])
        return {
            "chat_statistics": {k: stats[k] for k in ("total_messages", "total_users", "first_message_date", "last_message_date")},
            "distinctive_terms": self.index.top_terms(30),
            "most_common_words": self._common_words(self.df, 20),
            "top_emojis": self._emoji_counter(self.df).most_common(8),
            "representative_messages": self._rows(sample),
            "note": "The sample is spread across the whole time range and favours longer text messages. "
                    "Base the topic judgement on this evidence only and say so if it is thin or mixed.",
        }

    # ---- time analysis ---------------------------------------------------
    def get_activity_breakdown(
        self, dimension: str = "weekday", user: Optional[str] = None, limit: int = 12,
        start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> dict:
        """dimension: weekday | date | month | month_year | year | hour. First row is the busiest."""
        d = self._filter(user=user, start_date=start_date, end_date=end_date)
        dim = (dimension or "weekday").lower()
        col = {"weekday": "day_name", "date": "only_date", "month": "month", "year": "year", "hour": "hour"}.get(dim)
        if dim == "month_year":
            s = d.groupby([d["year"], d["month_num"], d["month"]]).size()
            s.index = [f"{m} {y}" for (y, _, m) in s.index]
        elif col:
            s = d[col].value_counts()
        else:
            raise ToolError("dimension must be one of weekday, date, month, month_year, year, hour")
        s = s.sort_values(ascending=False)
        rows = [{"period": _jsonable(k), "messages": int(v)} for k, v in s.head(_clamp_limit(limit)).items()]
        if dim == "hour":
            for r in rows:
                h = int(r["period"])
                r["period"] = f"{h:02d}:00-{(h + 1) % 24:02d}:00"
        out = {
            "dimension": dim, "user": self._resolve(user), "total_messages": int(len(d)),
            "busiest": rows[0] if rows else None,
            "quietest": rows[-1] if rows and len(s) <= _clamp_limit(limit) else None,
            "breakdown": rows,
        }
        if dim == "weekday" and self._resolve(user) is None and not d.empty:
            top_day = s.index[0]
            vc = d[d["day_name"] == top_day]["user"].value_counts().head(5)
            out["most_active_users_on_busiest_day"] = vc.to_dict()
        return out

    # ---- content analysis -----------------------------------------------
    def get_most_common_words(self, user: Optional[str] = None, limit: int = 20) -> dict:
        d = self._filter(user=user, text_only=True)
        return {
            "user": self._resolve(user), "text_messages_analysed": int(len(d)),
            "words": [{"word": w, "count": c} for w, c in self._common_words(d, _clamp_limit(limit))],
        }

    def get_emoji_statistics(self, user: Optional[str] = None, limit: int = 15) -> dict:
        d = self._filter(user=user, text_only=True)
        c = self._emoji_counter(d)
        per_user = None
        if self._resolve(user) is None:
            per_user = {u: int(sum(self._emoji_counter(g).values())) for u, g in d.groupby("user")}
            per_user = dict(sorted(per_user.items(), key=lambda kv: kv[1], reverse=True)[:10])
        return {
            "user": self._resolve(user), "total_emojis": int(sum(c.values())), "unique_emojis": len(c),
            "top_emojis": [{"emoji": e, "count": n} for e, n in c.most_common(_clamp_limit(limit))],
            "emojis_per_user_top10": per_user,
        }

    def get_link_statistics(self, user: Optional[str] = None, limit: int = 15) -> dict:
        d = self._filter(user=user)
        d = d[d["n_links"] > 0]
        domains: Counter = Counter()
        for links in d["links"]:
            for link in links:
                host = urlparse(link if "://" in link else "http://" + link).netloc.lower()
                domains[host[4:] if host.startswith("www.") else host] += 1
        return {
            "user": self._resolve(user), "total_links": int(d["n_links"].sum()),
            "messages_with_links": int(len(d)),
            "links_per_user": d.groupby("user")["n_links"].sum().sort_values(ascending=False).head(10).to_dict(),
            "top_domains": domains.most_common(10),
            "recent_link_messages": self._rows(self._chrono(d).tail(_clamp_limit(limit))),
        }

    def get_media_statistics(self, user: Optional[str] = None) -> dict:
        d = self._filter(user=user)
        m = d[d["is_media"]]
        by_month = m.groupby([m["year"], m["month"]]).size().sort_values(ascending=False).head(6)
        return {
            "user": self._resolve(user), "media_messages": int(len(m)),
            "media_per_user": m["user"].value_counts().head(10).to_dict(),
            "media_by_month": {f"{month} {year}": int(c) for (year, month), c in by_month.items()},
        }


# ---------------------------------------------------------------------------
# Tool specifications (JSON schema in Groq / OpenAI function-calling format).
# Descriptions matter: they are how the LLM decides which tool fits a question.
# ---------------------------------------------------------------------------

_USER = {"type": "string", "description": "Exact or partial participant name. Omit for the whole chat."}
_DATE = {"type": "string", "description": "Date in YYYY-MM-DD format."}
_LIMIT = {"type": "integer", "description": "Max rows to return (default 20, max 100)."}


def _tool(name: str, description: str, properties: dict, required: Optional[list] = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SPECS: list[dict] = [
    _tool("get_chat_statistics",
          "Overall totals: messages, users (with counts), words, media, links, deleted messages, date range, active days. "
          "Use for 'summary of this group', 'how many messages/users', or to check the spelling of a participant's name.",
          {}),
    _tool("get_user_statistics",
          "Full profile of ONE user: message count, % of total, rank, words, media, links, emojis, first & last message, "
          "top words/emojis, busiest weekday/month/hour. Call it twice (once per user) to compare two users.",
          {"user": _USER}, ["user"]),
    _tool("get_most_active_users",
          "Ranking of participants. by='messages' (most active / who talks most / who sent most), "
          "'words', 'media' (who shared most media), 'links' (who shared most links), 'emojis' (who uses most emojis).",
          {"limit": _LIMIT, "by": {"type": "string", "enum": ["messages", "words", "media", "links", "emojis"]}}),
    _tool("count_messages",
          "Exact count of messages matching filters, plus per-user breakdown when no user is given. Filters: user, date range, "
          "year, month (name or number), weekday (day_name), hour range (start_hour inclusive, end_hour exclusive, 24 = midnight), keyword. "
          "Answers 'how many messages did X send in March', 'who sent most messages on Sundays', 'messages between 10 PM and midnight' "
          "(start_hour=22, end_hour=24), 'who was most active in a period'.",
          {"user": _USER, "start_date": _DATE, "end_date": _DATE, "year": {"type": "integer"},
           "month": {"type": "string", "description": "Month name or number 1-12."},
           "day_name": {"type": "string", "description": "Weekday name, e.g. Sunday."},
           "start_hour": {"type": "integer", "description": "0-23, inclusive."},
           "end_hour": {"type": "integer", "description": "1-24, exclusive (24 = midnight)."},
           "keyword": {"type": "string", "description": "Only count messages containing this text."},
           "include_media": {"type": "boolean"}}),
    _tool("get_nth_message",
          "The Nth message in chronological order, 1-based. n=1 -> first message; from_end=true, n=1 -> last/latest message; "
          "n=3 -> third message. Give user for one person's messages, omit for the whole chat. Exact positional lookup; never guess this.",
          {"n": {"type": "integer", "description": "1-based position (default 1)."}, "user": _USER,
           "from_end": {"type": "boolean", "description": "Count from the most recent message instead (default false)."},
           "include_media": {"type": "boolean", "description": "Count media messages as messages (default true)."}}),
    _tool("get_messages",
          "List messages in order, optionally for one user and/or a date range. Use offset for paging ('next 20'), "
          "newest_first=true for the most recent, sort_by='longest' for the longest message(s). "
          "Answers 'show X's first 5 messages', 'what happened on/around DATE', 'longest message'.",
          {"user": _USER, "start_date": _DATE, "end_date": _DATE, "limit": _LIMIT,
           "offset": {"type": "integer"}, "newest_first": {"type": "boolean"},
           "sort_by": {"type": "string", "enum": ["chronological", "longest"]},
           "include_media": {"type": "boolean"}}),
    _tool("get_message_context",
          "Messages immediately before and after a given message_id, to read the surrounding conversation.",
          {"message_id": {"type": "integer"}, "before": {"type": "integer"}, "after": {"type": "integer"}},
          ["message_id"]),
    _tool("search_messages",
          "Exact keyword / substring search (case-insensitive). Best when the user gives a specific word, phrase, name or URL fragment. "
          "Returns matches with message_id, date and sender.",
          {"keyword": {"type": "string"}, "user": _USER, "limit": _LIMIT, "start_date": _DATE, "end_date": _DATE},
          ["keyword"]),
    _tool("semantic_search",
          "Meaning-based search for a topic or theme ('discussions about internships', 'what did X say about the project'). "
          "Finds related messages even when the exact word differs. Use for 'summarize conversations about Y' and 'find messages related to Z'; "
          "follow with get_message_context on interesting message_ids to read full conversations.",
          {"query": {"type": "string", "description": "Short natural-language description of the topic."},
           "user": _USER, "limit": _LIMIT, "start_date": _DATE, "end_date": _DATE},
          ["query"]),
    _tool("get_topic_overview",
          "Evidence pack for 'what is this group about / which field / main topics / kind of conversations / interesting insights': "
          "distinctive terms, common words, top emojis and a time-spread sample of representative messages. "
          "Optionally follow with semantic_search on 1-3 candidate topics for more evidence.",
          {"sample_size": {"type": "integer", "description": "Default 60, max 150."}}),
    _tool("get_activity_breakdown",
          "Message counts by 'weekday' (which day of the week is busiest), 'date' (busiest single day), 'month', 'month_year', 'year' "
          "or 'hour' (what time is the group most active), sorted busiest-first, optionally for one user or date range.",
          {"dimension": {"type": "string", "enum": ["weekday", "date", "month", "month_year", "year", "hour"]},
           "user": _USER, "limit": _LIMIT, "start_date": _DATE, "end_date": _DATE},
          ["dimension"]),
    _tool("get_most_common_words",
          "Most frequently used words (stop-words removed) for the whole chat or one user.",
          {"user": _USER, "limit": _LIMIT}),
    _tool("get_emoji_statistics",
          "Emoji usage: total, most frequent emojis, and emojis per user (who uses the most emojis).",
          {"user": _USER, "limit": _LIMIT}),
    _tool("get_link_statistics",
          "Links shared: totals, links per user (who shared most links), top domains, recent link messages.",
          {"user": _USER, "limit": _LIMIT}),
    _tool("get_media_statistics",
          "Media (photos/videos/audio/documents) shared: totals, per user (who sent most media), by month.",
          {"user": _USER}),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}


def run_tool(tools: ChatTools, name: str, args: dict) -> str:
    """Execute one tool by name and return a JSON string for the LLM. Never raises."""
    if name not in TOOL_NAMES:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        result = getattr(tools, name)(**(args or {}))
        text = json.dumps(_jsonable(result), ensure_ascii=False, default=str)
    except ToolError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except TypeError as e:  # bad / missing argument names
        return json.dumps({"error": f"Bad arguments for {name}: {e}"}, ensure_ascii=False)
    except Exception as e:  # never let a tool bug crash the app
        return json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"}, ensure_ascii=False)
    if len(text) > config.MAX_TOOL_RESULT_CHARS:
        text = json.dumps({
            "truncated": True,
            "note": "Result too large; call again with a smaller limit or more filters.",
            "partial_result": text[: config.MAX_TOOL_RESULT_CHARS],
        }, ensure_ascii=False)
    return text
