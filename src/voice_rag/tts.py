"""Text-to-speech protocol. The Kokoro implementation is added in the next stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SpeechResult:
    """Where the audio went and how long synthesis took (playback is timed separately)."""

    path: Path | None
    synthesis_seconds: float
    audio_seconds: float = 0.0


class Speaker(Protocol):
    """Anything that can turn text into audio."""

    def speak(self, text: str, name: str) -> SpeechResult:
        """Synthesize ``text`` into a file named after ``name`` and (optionally) play it."""
        ...
