"""Small immutable data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the knowledge base: a single sentence and where it came from."""

    text: str
    source: str  # file name, e.g. "products.txt"
    index: int  # 0-based sentence number within the file


@dataclass(frozen=True)
class Retrieved:
    """A chunk returned by similarity search together with its cosine score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a guard check. ``messages`` are shown in the terminal as ``guard: ...``."""

    ok: bool
    messages: tuple[str, ...] = ()


@dataclass
class Timings:
    """Per-query wall-clock timings in seconds, plus the server-side durations Ollama reports."""

    retrieval: float = 0.0
    llm: float = 0.0
    tts: float = 0.0
    total: float = 0.0
    # Raw Ollama response numbers (nanoseconds / token counts); empty when the LLM was skipped.
    ollama: dict[str, int] = field(default_factory=dict)


@dataclass
class Answer:
    """Everything the CLI needs to display and speak one answer."""

    question: str
    text: str
    retrieved: list[Retrieved]
    guard_messages: list[str]
    timings: Timings
    llm_used: bool = True
    audio_path: str | None = None
