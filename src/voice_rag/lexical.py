"""Lexical (keyword) retrieval: a small BM25 implementation over the same sentence chunks.

Embeddings are good at paraphrase but can under-rank exact tokens such as "L1" or "USB-C"; BM25
is the opposite. The two rankings are fused in :mod:`voice_rag.retrieve`. Hand-rolled because the
whole thing is thirty lines and avoids a dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from numpy.typing import NDArray

_TOKEN = re.compile(r"[a-z0-9]+")
# Generic English function words. They carry no topical signal and would otherwise let "the" or
# "is" create spurious keyword matches. Applied to BM25 only; other callers keep every token.
STOP_WORDS = frozenset(
    """
    a an the and or but if so of for to in on at by with from as into about
    is are was were be been being do does did have has had will would can could should
    shall may might must
    i you he she it we they me him her us them my your his its our their
    this that these those there here what which who whom whose when where why how
    not no yes
    """.split()
)


_MIN_STEM = 3


def singular(word: str) -> str:
    """Crude plural stripping: "batteries" -> "battery", "lamps" -> "lamp", "days" -> "day"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Crude suffix stripping so related forms share a token: delivery/delivered -> "deliver".

    Not a real stemmer: it removes a plural, then one of -ing/-ed/-ly/-y, and collapses a doubled
    final consonant ("shipping" -> "ship"). Good enough for keyword matching on a small KB.
    """
    word = singular(word)
    for suffix in ("ing", "ed", "ly", "y"):
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            word = word[: -len(suffix)]
            if len(word) > _MIN_STEM and word[-1] == word[-2] and word[-1] not in "aeiou":
                word = word[:-1]
            break
    return word


def tokenize(text: str, drop_stop_words: bool = False) -> list[str]:
    """Lowercase alphanumeric tokens with crude stemming; punctuation is dropped."""
    raw = _TOKEN.findall(text.lower())
    if drop_stop_words:
        raw = [t for t in raw if t not in STOP_WORDS]
    return [stem(t) for t in raw]


class BM25:
    """Okapi BM25 scorer over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = [Counter(tokenize(d, drop_stop_words=True)) for d in documents]
        self._lengths = np.array([sum(c.values()) for c in self._docs], dtype=np.float32)
        self._avg_len = float(self._lengths.mean()) if len(self._docs) else 0.0
        df: Counter[str] = Counter()
        for counts in self._docs:
            df.update(counts.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> NDArray[np.float32]:
        """BM25 score of every document for ``query`` (0 when no token overlaps)."""
        out = np.zeros(len(self._docs), dtype=np.float32)
        for token in set(tokenize(query, drop_stop_words=True)):
            idf = self._idf.get(token)
            if idf is None:
                continue
            for i, counts in enumerate(self._docs):
                tf = counts.get(token, 0)
                if tf:
                    norm = 1 - self._b + self._b * self._lengths[i] / self._avg_len
                    out[i] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        return out
