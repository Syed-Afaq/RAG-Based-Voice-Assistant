from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voice_rag.config import Settings
from voice_rag.tts import SAMPLE_RATE, KokoroSpeaker


def fake_synth(text: str) -> np.ndarray:
    """One second of silence per ten characters, so audio length tracks text length."""
    return np.zeros(int(SAMPLE_RATE * len(text) / 10), dtype=np.float32)


def test_synthesize_normalises_text_and_writes_wav(tmp_path: Path) -> None:
    speaker = KokoroSpeaker(Settings(), audio_dir=tmp_path, synthesizer=fake_synth)
    result = speaker.synthesize("The L1 lamp costs PKR 2,700.", "answer")
    assert result.spoken_text == "The L 1 lamp costs 2,700 Pakistani rupees."
    assert result.path == tmp_path / "answer.wav"
    data, rate = sf.read(result.path)
    assert rate == SAMPLE_RATE
    assert abs(len(data) / rate - result.audio_seconds) < 1e-6
    assert result.synthesis_seconds >= 0.0


def test_play_without_player_is_a_noop(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("voice_rag.tts.shutil.which", lambda _: None)
    speaker = KokoroSpeaker(Settings(), audio_dir=tmp_path, synthesizer=fake_synth)
    assert speaker.play(tmp_path / "missing.wav") == 0.0


def test_warm_up_runs_synthesizer(tmp_path: Path) -> None:
    calls: list[str] = []

    def synth(text: str) -> np.ndarray:
        calls.append(text)
        return fake_synth(text)

    speaker = KokoroSpeaker(Settings(), audio_dir=tmp_path, synthesizer=synth)
    assert speaker.warm_up() >= 0.0
    assert calls == ["Ready."]
    assert speaker.load_seconds >= 0.0
