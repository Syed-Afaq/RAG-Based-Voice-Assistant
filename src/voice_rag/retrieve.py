"""Similarity search over the index: cosine similarity via a single matrix-vector product."""

from __future__ import annotations

import time

import numpy as np

from voice_rag.embedder import Embedder
from voice_rag.index import Index
from voice_rag.types import Retrieved


def search(index: Index, query_vector: np.ndarray, top_k: int) -> list[Retrieved]:
    """Rank chunks by cosine similarity to an already-embedded query."""
    scores = index.vectors @ query_vector.astype(np.float32)
    order = np.argsort(-scores)[:top_k]
    return [Retrieved(chunk=index.chunks[i], score=float(scores[i])) for i in order]


def retrieve(
    index: Index, embedder: Embedder, question: str, top_k: int
) -> tuple[list[Retrieved], float]:
    """Embed the question and return the top-k passages plus the elapsed retrieval time."""
    started = time.perf_counter()
    hits = search(index, embedder.embed_query(question), top_k)
    return hits, time.perf_counter() - started
