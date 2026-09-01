"""
retrieval.py — lightweight semantic retrieval for topic / meaning-based questions.

Why this exists
---------------
Deterministic Pandas tools (ai_tools.py) answer "how many / which / Nth"
questions exactly. But "what did people discuss about internships?" needs the
agent to *find the relevant messages* first, and a chat can be far too large
to hand to the LLM in full. This module is that finder. It is one tool the
agent can call — not the primary architecture.

Why TF-IDF and not a vector database
------------------------------------
* The corpus lives for one Streamlit session and is rebuilt on every upload,
  so a persistent store (Chroma / FAISS / Pinecone) adds nothing.
* TF-IDF needs no model download and no GPU, builds in well under a second
  for typical chats, and runs fully in memory.
* Character n-grams (3-5 chars) make it tolerant of Hinglish spelling
  variation ("kya" / "kyaa", "internship" / "internshp").
* If you ever need true synonym matching ("placement" ~ "job offer"), the
  only change is swapping `_scores()` for a sentence-transformers encoder;
  the rest of the interface stays the same.

What it never does
------------------
Lose metadata. Every hit is a `message_id` pointing back into the ChatTools
DataFrame, so the agent always sees the real author and real timestamp.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


class MessageIndex:
    def __init__(self, df: pd.DataFrame, stopwords: set[str]):
        """
        df must be the enriched frame from ChatTools (needs columns:
        message_id, text, is_text, user, date, word_count).
        """
        corpus = df[df["is_text"] & (df["word_count"] >= 1)]
        self.ids = corpus["message_id"].to_numpy()
        self.users = corpus["user"].to_numpy()
        self.dates = corpus["date"].to_numpy()
        self.word_counts = corpus["word_count"].to_numpy()
        self.texts = [_URL_RE.sub(" ", t).lower() for t in corpus["text"].tolist()]
        self.matrix = None
        self._word_matrix = None

        if not self.texts:
            return

        # Word/bigram vectoriser: carries topical meaning; also used for top_terms().
        self._word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            # Only keep stop-words the tokenizer could actually produce (avoids sklearn warnings).
            stop_words=[w for w in stopwords if re.fullmatch(r"\w\w+", w)] or None,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        # Character vectoriser: spelling tolerance for Hinglish / typos.
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        self._word_matrix = self._word_vec.fit_transform(self.texts)
        char_matrix = self._char_vec.fit_transform(self.texts)
        self.matrix = normalize(hstack([self._word_matrix, char_matrix]).tocsr(), norm="l2", axis=1)

    @property
    def size(self) -> int:
        return int(len(self.ids))

    # ------------------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        q = _URL_RE.sub(" ", query).lower()
        qv = normalize(
            hstack([self._word_vec.transform([q]), self._char_vec.transform([q])]).tocsr(),
            norm="l2", axis=1,
        )
        return np.asarray((self.matrix @ qv.T).todense()).ravel()

    def search(
        self,
        query: str,
        k: int = 15,
        user: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        min_score: float = 0.05,
    ) -> list[dict]:
        """Return up to k hits as [{message_id, score}], best first."""
        if self.matrix is None or not query.strip():
            return []
        scores = self._scores(query)
        mask = np.ones(len(scores), dtype=bool)
        if user:
            mask &= self.users == user
        if start is not None:
            mask &= self.dates >= np.datetime64(start)
        if end is not None:
            mask &= self.dates <= np.datetime64(end)
        scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)[: max(int(k), 1)]
        return [
            {"message_id": int(self.ids[i]), "score": float(scores[i])}
            for i in order
            if scores[i] >= min_score
        ]

    def top_terms(self, n: int = 30) -> list[list]:
        """Most distinctive words / bigrams across the chat (mean TF-IDF weight)."""
        if self._word_matrix is None:
            return []
        means = np.asarray(self._word_matrix.mean(axis=0)).ravel()
        vocab = self._word_vec.get_feature_names_out()
        out = []
        for i in np.argsort(-means)[: n * 3]:
            term = vocab[i]
            if term.isdigit() or len(term) < 3:
                continue
            out.append([term, round(float(means[i]), 4)])
            if len(out) >= n:
                break
        return out

    def representative_sample(self, n: int = 60) -> list[int]:
        """
        message_ids spread evenly across the timeline, preferring longer
        messages (they carry more topical signal than "ok" / "haan").
        Used to give the LLM evidence for "what is this group about?".
        """
        if self.size == 0:
            return []
        n = min(n, self.size)
        chosen = []
        for bucket in np.array_split(np.arange(self.size), n):
            if len(bucket) == 0:
                continue
            wc = self.word_counts[bucket]
            # Longest message in the bucket, capped so pasted essays don't always win.
            best = bucket[np.argmax(np.minimum(wc, 40))]
            chosen.append(int(self.ids[best]))
        return chosen
