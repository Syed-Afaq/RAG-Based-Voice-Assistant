"""Embedding backends behind a small protocol so tests can substitute a fake."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import ollama
from numpy.typing import NDArray

# nomic-embed-text is trained with task prefixes; they must be present for good retrieval.
# See https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder(Protocol):
    """Turns text into L2-normalised vectors. ``model_name`` is stored in the index metadata."""

    @property
    def model_name(self) -> str:
        """Identifier recorded in the index so a model change triggers a rebuild."""
        ...

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed passages for indexing. Returns an array of shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one question. Returns an array of shape ``(dim,)``."""
        ...


def normalise(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale every row to unit length so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class OllamaEmbedder:
    """Embeds text with a model served by Ollama (``nomic-embed-text`` by default)."""

    def __init__(self, client: ollama.Client, model: str, keep_alive: str | None = None) -> None:
        self._client = client
        self._model = model
        self._keep_alive = keep_alive
        self.last_load_ns: int | None = None  # load_duration of the most recent call

    @property
    def model_name(self) -> str:
        """The Ollama model tag."""
        return self._model

    def _embed(self, texts: list[str]) -> NDArray[np.float32]:
        response = self._client.embed(model=self._model, input=texts, keep_alive=self._keep_alive)
        self.last_load_ns = response.load_duration
        return normalise(np.asarray(response.embeddings, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed passages with the document prefix, in one batched request."""
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a question with the query prefix."""
        vector: NDArray[np.float32] = self._embed([QUERY_PREFIX + text])[0]
        return vector
