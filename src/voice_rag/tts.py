"""Text-to-speech with Kokoro-82M, WAV output and playback through ``paplay``.

Synthesis and playback are separate steps so the CLI can show the answer and its timings before
the audio starts, and so timing-only runs can skip playback. Kokoro and torch are imported lazily:
they take several seconds and over a gigabyte of RAM, and nothing else in the package needs them.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import logging
import os
import shutil
import subprocess
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from voice_rag.config import Settings
from voice_rag.speech_text import normalize_for_speech

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000  # Kokoro's native output rate
KOKORO_REPO = "hexgrad/Kokoro-82M"
PLAYER = "paplay"

Synthesizer = Callable[[str], NDArray[np.float32]]


@dataclass
class SpeechResult:
    """Where the audio went and how long synthesis took (playback is timed separately)."""

    path: Path | None
    synthesis_seconds: float
    audio_seconds: float = 0.0
    spoken_text: str = ""


class Speaker(Protocol):
    """Anything that can turn text into a playable file."""

    def synthesize(self, text: str, name: str) -> SpeechResult:
        """Write speech for ``text`` to a file whose stem is ``name``."""
        ...

    def play(self, path: Path) -> float:
        """Play a WAV file and return the seconds spent playing."""
        ...


def _release_memory() -> None:
    """Return freed heap pages to the OS; loading the weights leaves transient garbage behind."""
    gc.collect()
    with contextlib.suppress(OSError, AttributeError):  # not glibc (e.g. macOS): skip
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _load_kokoro(voice: str, speed: float, threads: int | None) -> Synthesizer:
    """Build the real Kokoro synthesizer. Imports torch, so it is only called on demand.

    The weights are looked up in the Hugging Face cache first (no network round-trip, no rate-limit
    warning); only if that fails is a download attempted.
    """
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch prints deprecation notices while loading weights
        import torch
        from kokoro import KPipeline

        if threads is not None:
            torch.set_num_threads(threads)
        pipeline: Any
        if "HF_HUB_OFFLINE" in os.environ:
            pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO)
        else:
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO)
            except Exception:  # noqa: BLE001 - any cache miss means "go online once"
                os.environ.pop("HF_HUB_OFFLINE", None)
                log.info("Kokoro weights not cached; downloading %s", KOKORO_REPO)
                pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO)
            finally:
                os.environ.pop("HF_HUB_OFFLINE", None)
    _release_memory()

    def synthesize(text: str) -> NDArray[np.float32]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = [
                r.audio for r in pipeline(text, voice=voice, speed=speed) if r.audio is not None
            ]
        if not segments:
            return np.zeros(0, dtype=np.float32)
        audio = torch.cat(segments) if len(segments) > 1 else segments[0]
        return np.asarray(audio.numpy(), dtype=np.float32)

    return synthesize


class KokoroSpeaker:
    """Kokoro-82M (American English voice) writing 24 kHz WAV files under ``audio_dir``."""

    def __init__(
        self,
        settings: Settings,
        audio_dir: Path | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.audio_dir = audio_dir or settings.audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self._synthesize = synthesizer or _load_kokoro(
            settings.tts_voice, settings.tts_speed, settings.num_thread
        )
        self.load_seconds = time.perf_counter() - started
        self._player = shutil.which(PLAYER)
        if self._player is None:
            log.warning("%s not found; audio will be saved but not played", PLAYER)

    def warm_up(self) -> float:
        """Synthesize a short phrase so the voice pack is loaded; returns seconds taken."""
        started = time.perf_counter()
        self._synthesize("Ready.")
        return time.perf_counter() - started

    def synthesize(self, text: str, name: str) -> SpeechResult:
        """Normalize ``text`` for speech, synthesize it and save ``<audio_dir>/<name>.wav``."""
        spoken = normalize_for_speech(text)
        started = time.perf_counter()
        audio = self._synthesize(spoken)
        seconds = time.perf_counter() - started
        path = self.audio_dir / f"{name}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        log.debug("tts: %.2fs for %.1fs of audio -> %s", seconds, len(audio) / SAMPLE_RATE, path)
        return SpeechResult(
            path=path,
            synthesis_seconds=seconds,
            audio_seconds=len(audio) / SAMPLE_RATE,
            spoken_text=spoken,
        )

    def play(self, path: Path) -> float:
        """Play through PulseAudio (works under WSLg). Returns seconds spent, 0 if no player."""
        if self._player is None:
            return 0.0
        started = time.perf_counter()
        try:
            subprocess.run([self._player, str(path)], check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            log.warning("playback failed: %s", exc)
        return time.perf_counter() - started
