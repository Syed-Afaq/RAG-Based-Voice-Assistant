from __future__ import annotations

from rich.console import Console

from voice_rag.cli import parse_args, print_answer
from voice_rag.types import Answer, Chunk, Retrieved, Timings


def test_parse_args_defaults_and_flags() -> None:
    args = parse_args([])
    assert args.ask is None and not args.no_audio and not args.no_tts and not args.reindex
    args = parse_args(
        ["--ask", "Q?", "--model", "qwen3:1.7b", "--no-audio", "--reindex", "-v"][:-1]
    )
    assert args.ask == "Q?" and args.model == "qwen3:1.7b" and args.no_audio and args.reindex


def test_print_answer_shows_sources_guards_answer_and_timings() -> None:
    console = Console(record=True, width=120, force_terminal=False)
    answer = Answer(
        question="Q?",
        text="No built-in battery.",
        retrieved=[Retrieved(Chunk("The L1 lamp.", "products.txt", 0), 0.9, bm25_rank=1)],
        guard_messages=["injection pattern: x"],
        timings=Timings(retrieval=0.05, llm=1.5, tts=0.7, total=2.3),
    )
    print_answer(console, answer)
    out = console.export_text()
    assert "products.txt" in out and "0.900" in out and "#1" in out
    assert "guard: injection pattern: x" in out
    assert "No built-in battery." in out
    assert "retrieval 50 ms" in out and "LLM 1.50s" in out and "TTS generation 0.70s" in out
