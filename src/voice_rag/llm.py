"""Chat-model backends behind a small protocol, with an Ollama implementation."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import ollama

from voice_rag.config import Settings

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
OLLAMA_METRIC_FIELDS = (
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "total_duration",
)


@dataclass
class LLMResponse:
    """Generated text plus wall-clock seconds and the raw server-side metrics."""

    text: str
    seconds: float
    metrics: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    """Anything that can answer a (system, user) prompt pair."""

    @property
    def model_name(self) -> str:
        """Model identifier for logs and results."""
        ...

    def generate(self, system: str, user: str) -> LLMResponse:
        """Return the model's reply for one system prompt and one user message."""
        ...


class LLMUnavailableError(RuntimeError):
    """Raised with an actionable message when Ollama or the model cannot be reached."""


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks that hybrid Qwen3 models sometimes leak."""
    return _THINK.sub("", text).strip()


class OllamaLLM:
    """Deterministic chat completion through a local Ollama server."""

    def __init__(self, client: ollama.Client, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._options: dict[str, Any] = {
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
        }
        if settings.num_thread is not None:
            self._options["num_thread"] = settings.num_thread

    @property
    def model_name(self) -> str:
        """The Ollama model tag."""
        return self._settings.llm_model

    def generate(self, system: str, user: str) -> LLMResponse:
        """Call ``/api/chat`` once and return the (think-stripped) reply with metrics."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        think = self._settings.think_option()
        extra: dict[str, Any] = {} if think is None else {"think": think}
        started = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options=self._options,
                keep_alive=self._settings.keep_alive,
                **extra,
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailableError(
                    f"model '{self.model_name}' is not available; "
                    f"run: ollama pull {self.model_name}"
                ) from exc
            raise LLMUnavailableError(f"Ollama error: {exc.error}") from exc
        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"cannot reach Ollama at {self._settings.ollama_host}; start it with: ollama serve"
            ) from exc
        seconds = time.perf_counter() - started
        metrics = {k: int(v) for k in OLLAMA_METRIC_FIELDS if (v := getattr(response, k, None))}
        text = strip_think(response.message.content or "")
        log.debug("llm %s: %.2fs, %s", self.model_name, seconds, metrics)
        return LLMResponse(text=text, seconds=seconds, metrics=metrics)

    def warm_up(self) -> float:
        """Load the model with a tiny request; returns seconds (model load time)."""
        started = time.perf_counter()
        self.generate("Reply with the single word OK.", "Ready?")
        return time.perf_counter() - started
