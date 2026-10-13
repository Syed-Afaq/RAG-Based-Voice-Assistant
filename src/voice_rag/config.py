"""Runtime settings with environment-variable overrides.

Every value has a default here. A variable named ``VOICE_RAG_<FIELD>`` in the environment (or in a
``.env`` file in the working directory) overrides it. Knowledge-base facts never live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

ENV_PREFIX = "VOICE_RAG_"


@dataclass(frozen=True)
class Settings:
    """All tunable parameters of the assistant."""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b-instruct"
    embed_model: str = "nomic-embed-text"

    # Retrieval
    top_k: int = 5  # the KB has 8 chunks; 5 keeps the delivery-fee passage in scope for scenario 6
    min_similarity: float = 0.55  # best-cosine floor; below it all passages are sent (see README)

    # Generation: deterministic and memory-lean by default
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 1024  # prompts are ~450 tokens; 1024 saves ~200 MB over 2048 on this box
    num_predict: int = 120
    num_thread: int | None = 6  # leaves cores free for other apps (e.g. a video call); None = all
    keep_alive: str = "10m"

    # Text-to-speech
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0

    # Paths
    kb_dir: Path = Path("kb")
    index_dir: Path = Path(".index")
    audio_dir: Path = Path("results/audio")

    def think_option(self) -> bool | None:
        """Return the ``think`` value to send to Ollama for the configured LLM.

        Qwen3 hybrid models (``qwen3:1.7b``, ``qwen3:4b``...) reason by default, which is slow and
        leaks ``<think>`` text, so we switch it off. Instruct-2507 builds are non-thinking and may
        reject the parameter, so we do not send it at all.
        """
        return None if "instruct" in self.llm_model.lower() else False


def _parse(value: str, target: type) -> object:
    """Convert an environment string into the type of the dataclass field."""
    if target is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if target is Path:
        return Path(value)
    return target(value)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from defaults, ``.env`` and the process environment.

    Args:
        env: Optional explicit mapping (used by tests). Defaults to ``os.environ`` after loading
            ``.env`` from the current directory.

    """
    if env is None:
        load_dotenv()
        env = dict(os.environ)

    overrides: dict[str, object] = {}
    for f in fields(Settings):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is None or raw.strip() == "":
            continue
        # Optional fields are declared as "int | None"; the base type is the first part.
        base = f.type.split(" | ")[0] if isinstance(f.type, str) else f.type
        target = {"int": int, "float": float, "str": str, "bool": bool, "Path": Path}[str(base)]
        overrides[f.name] = _parse(raw, target)
    return Settings(**overrides)  # type: ignore[arg-type]
