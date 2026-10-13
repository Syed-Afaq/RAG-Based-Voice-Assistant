"""Deterministic stand-ins for the model-backed components, so unit tests need no Ollama."""

from __future__ import annotations

import re
import zlib

import numpy as np
from numpy.typing import NDArray

from voice_rag.embedder import normalise

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words embedding: texts sharing words get a higher cosine score."""

    model_name = "fake-embed"

    def __init__(self, model_name: str = "fake-embed") -> None:
        self.model_name = model_name
        self.document_calls = 0

    @staticmethod
    def _vector(text: str) -> NDArray[np.float32]:
        vec = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % DIM] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        self.document_calls += 1
        return normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return normalise(self._vector(text)[None, :])[0]
