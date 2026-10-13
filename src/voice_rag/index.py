"""Build, persist and refresh the local vector index over the knowledge base.

Chunking: one sentence per chunk. Sentences are split only at ``.``, ``!`` or ``?`` followed by
whitespace. Semicolon clauses such as "Delivery to Islamabad takes 2–3 working days; Lahore takes
3–5 working days." stay together on purpose: splitting them would leave "Lahore takes 3–5 working
days" without the word "delivery", which is the term a question will mention.

Freshness: the metadata stores a SHA-256 of every KB file plus the embedding model name and the
chunking version. On load, anything that differs triggers a rebuild, and the caller is told why.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voice_rag.embedder import Embedder
from voice_rag.lexical import BM25
from voice_rag.types import Chunk

log = logging.getLogger(__name__)

CHUNKING_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed sentences, dropping empty pieces."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def chunk_file(path: Path) -> list[Chunk]:
    """Read one KB file and return a chunk per sentence, tagged with file name and position."""
    sentences = split_sentences(path.read_text(encoding="utf-8"))
    return [Chunk(text=s, source=path.name, index=i) for i, s in enumerate(sentences)]


def kb_files(kb_dir: Path) -> list[Path]:
    """All ``.txt`` files in the knowledge base, in a stable order."""
    return sorted(p for p in kb_dir.glob("*.txt") if p.is_file())


def file_hashes(kb_dir: Path) -> dict[str, str]:
    """SHA-256 hex digest of every KB file, keyed by file name."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in kb_files(kb_dir)}


@dataclass
class Index:
    """Chunks plus their unit-length embedding matrix (``vectors[i]`` belongs to ``chunks[i]``)."""

    chunks: list[Chunk]
    vectors: NDArray[np.float32]
    embed_model: str
    hashes: dict[str, str]
    build_seconds: float = 0.0
    lexical: BM25 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # BM25 statistics are cheap to derive, so they are rebuilt on load instead of persisted.
        self.lexical = BM25([c.text for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)


@dataclass
class RefreshReport:
    """Why an index was (or was not) rebuilt. Shown in the terminal on startup and ``/reload``."""

    rebuilt: bool
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """One-line human summary."""
        if not self.rebuilt:
            return "index up to date"
        return "index rebuilt: " + "; ".join(self.reasons)


def build_index(kb_dir: Path, embedder: Embedder) -> Index:
    """Chunk every KB file and embed all chunks in one batch."""
    chunks = [c for p in kb_files(kb_dir) for c in chunk_file(p)]
    if not chunks:
        raise ValueError(f"no sentences found in any .txt file under {kb_dir}")
    started = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    elapsed = time.perf_counter() - started
    log.debug("embedded %d chunks in %.2fs", len(chunks), elapsed)
    return Index(
        chunks=chunks,
        vectors=vectors,
        embed_model=embedder.model_name,
        hashes=file_hashes(kb_dir),
        build_seconds=elapsed,
    )


def save_index(index: Index, index_dir: Path) -> None:
    """Write the embedding matrix and a JSON sidecar with chunks, hashes and model info."""
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / EMBEDDINGS_FILE, index.vectors)
    meta = {
        "embed_model": index.embed_model,
        "chunking_version": CHUNKING_VERSION,
        "hashes": index.hashes,
        "chunks": [{"text": c.text, "source": c.source, "index": c.index} for c in index.chunks],
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")


def load_index(index_dir: Path) -> Index | None:
    """Load a saved index, or return ``None`` if it is missing or unreadable."""
    meta_path = index_dir / META_FILE
    vec_path = index_dir / EMBEDDINGS_FILE
    if not (meta_path.exists() and vec_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
        vectors = np.load(vec_path)
        chunks = [Chunk(**c) for c in meta["chunks"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("ignoring unreadable index in %s: %s", index_dir, exc)
        return None
    if meta.get("chunking_version") != CHUNKING_VERSION or len(chunks) != len(vectors):
        return None
    return Index(
        chunks=chunks, vectors=vectors, embed_model=meta["embed_model"], hashes=meta["hashes"]
    )


def stale_reasons(index: Index, kb_dir: Path, embed_model: str) -> list[str]:
    """List what changed since ``index`` was built; empty means it is still valid."""
    reasons: list[str] = []
    if index.embed_model != embed_model:
        reasons.append(f"embedding model changed ({index.embed_model} -> {embed_model})")
    current = file_hashes(kb_dir)
    for name in sorted(set(current) | set(index.hashes)):
        if name not in index.hashes:
            reasons.append(f"{name} added")
        elif name not in current:
            reasons.append(f"{name} removed")
        elif current[name] != index.hashes[name]:
            reasons.append(f"{name} changed")
    return reasons


def load_or_build(
    kb_dir: Path, index_dir: Path, embedder: Embedder, force: bool = False
) -> tuple[Index, RefreshReport]:
    """Return a valid index, rebuilding it when the KB, the model or the chunker changed."""
    existing = None if force else load_index(index_dir)
    if existing is None:
        reasons = ["forced"] if force else ["no saved index"]
    else:
        reasons = stale_reasons(existing, kb_dir, embedder.model_name)
        if not reasons:
            return existing, RefreshReport(rebuilt=False)
    index = build_index(kb_dir, embedder)
    save_index(index, index_dir)
    return index, RefreshReport(rebuilt=True, reasons=reasons)
