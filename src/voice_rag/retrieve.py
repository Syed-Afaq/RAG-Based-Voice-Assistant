"""Hybrid retrieval: cosine similarity and BM25 merged with reciprocal rank fusion (RRF).

RRF gives each chunk ``sum(1 / (60 + rank))`` over the two rankings, so a passage that both
methods like rises to the top without having to calibrate the two score scales against each other.

Low-confidence fallback: when even the best cosine score is below ``min_similarity`` the question
is probably phrased in words the KB never uses (or is off-topic). With a nine-sentence KB the
cheapest robust answer is to hand the model every passage and let the prompt rules decide. A large
KB would instead use query rewriting or a cross-encoder re-ranker here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from voice_rag.embedder import Embedder
from voice_rag.index import Index
from voice_rag.types import Retrieved

RRF_K = 60


@dataclass
class RetrievalResult:
    """What the pipeline gets back: ranked passages plus the facts needed to explain them."""

    hits: list[Retrieved]
    seconds: float
    best_cosine: float
    low_confidence: bool
    total_chunks: int


def _ranks(scores: NDArray[np.float32], positive_only: bool) -> dict[int, int]:
    """Map chunk index -> 1-based rank. With ``positive_only`` chunks scoring 0 get no rank."""
    order = np.argsort(-scores, kind="stable")
    ranks: dict[int, int] = {}
    for rank, i in enumerate(order, start=1):
        if positive_only and scores[i] <= 0:
            break
        ranks[int(i)] = rank
    return ranks


def search(index: Index, question: str, query_vector: NDArray[np.float32]) -> list[Retrieved]:
    """Rank every chunk by RRF of cosine similarity and BM25 (best first)."""
    cosine = index.vectors @ query_vector.astype(np.float32)
    bm25_ranks = _ranks(index.lexical.scores(question), positive_only=True)
    cosine_ranks = _ranks(cosine, positive_only=False)
    fused = {
        i: 1 / (RRF_K + cosine_ranks[i]) + (1 / (RRF_K + bm25_ranks[i]) if i in bm25_ranks else 0)
        for i in range(len(index.chunks))
    }
    return [
        Retrieved(
            chunk=index.chunks[i],
            score=float(cosine[i]),
            bm25_rank=bm25_ranks.get(i),
            fused=fused[i],
        )
        for i in sorted(fused, key=lambda i: (-fused[i], -cosine[i]))
    ]


def retrieve(
    index: Index,
    embedder: Embedder,
    question: str,
    top_k: int,
    min_similarity: float,
    anchor_terms: list[str] | None = None,
) -> RetrievalResult:
    """Embed the question and rank all chunks; return top-k, or everything on low confidence.

    ``anchor_terms`` (product codes the question refers to) guarantee that the passage defining each
    one is included even when it falls outside the top-k, e.g. "the total for both products" names
    no product word, yet needs both price sentences.
    """
    started = time.perf_counter()
    ranked = search(index, question, embedder.embed_query(question))
    best = max((r.score for r in ranked), default=0.0)
    low = best < min_similarity
    hits = ranked if low else ranked[:top_k]
    for term in anchor_terms or []:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        if not any(pattern.search(h.chunk.text) for h in hits):
            hits.extend(r for r in ranked[top_k:] if pattern.search(r.chunk.text))
    return RetrievalResult(
        hits=hits,
        seconds=time.perf_counter() - started,
        best_cosine=best,
        low_confidence=low,
        total_chunks=len(ranked),
    )
