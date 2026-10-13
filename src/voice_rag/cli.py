"""Command-line interface: interactive loop and one-shot mode."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voice_rag import __version__
from voice_rag.config import Settings, load_settings
from voice_rag.embedder import OllamaEmbedder
from voice_rag.llm import LLMUnavailableError, OllamaLLM
from voice_rag.logging_setup import setup_logging
from voice_rag.pipeline import Pipeline
from voice_rag.tts import KokoroSpeaker
from voice_rag.types import Answer

log = logging.getLogger(__name__)

HELP = """\
Type a question and press Enter. Commands:
  /reload   re-check the knowledge-base files and rebuild the index if anything changed
  /help     show this help
  /quit     exit (Ctrl-D also works)"""


@dataclass
class LoadTimes:
    """Startup costs, reported separately from per-question timings."""

    index_seconds: float
    index_note: str
    embed_warm_seconds: float
    llm_warm_seconds: float
    tts_load_seconds: float | None
    tts_warm_seconds: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line flags."""
    parser = argparse.ArgumentParser(
        prog="voice-rag", description="Voice-response assistant for the store knowledge base."
    )
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--model", help="Ollama chat model tag (default from settings)")
    parser.add_argument("--no-audio", action="store_true", help="synthesize speech but do not play")
    parser.add_argument("--no-tts", action="store_true", help="skip speech synthesis entirely")
    parser.add_argument("--reindex", action="store_true", help="force an index rebuild at startup")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def check_ollama(client: ollama.Client, settings: Settings) -> None:
    """Fail fast with a command to run when the server or a model is missing."""
    try:
        available = {m.model for m in client.list().models}
    except (httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailableError(
            f"cannot reach Ollama at {settings.ollama_host}; start it with: ollama serve"
        ) from exc
    for tag in (settings.llm_model, settings.embed_model):
        if tag not in available and f"{tag}:latest" not in available:
            raise LLMUnavailableError(f"model '{tag}' is not pulled; run: ollama pull {tag}")


def start(
    settings: Settings, console: Console, use_tts: bool, reindex: bool
) -> tuple[Pipeline, LoadTimes]:
    """Connect, build or load the index, warm the models, and report each loading time."""
    client = ollama.Client(host=settings.ollama_host)
    check_ollama(client, settings)
    embedder = OllamaEmbedder(client, settings.embed_model, keep_alive=settings.keep_alive)
    llm = OllamaLLM(client, settings)

    with console.status("loading index and models..."):
        started = time.perf_counter()
        pipeline = Pipeline(settings, embedder, llm, speaker=None)
        if reindex:
            pipeline.reload(force=True)
        index_seconds = time.perf_counter() - started

        started = time.perf_counter()
        embedder.embed_query("warm-up")
        embed_warm = time.perf_counter() - started

        llm_warm = llm.warm_up()

        tts_load = tts_warm = None
        if use_tts:
            speaker = KokoroSpeaker(settings)
            tts_load = speaker.load_seconds
            tts_warm = speaker.warm_up()
            pipeline.speaker = speaker

    times = LoadTimes(
        index_seconds=index_seconds,
        index_note=pipeline.last_refresh.describe(),
        embed_warm_seconds=embed_warm,
        llm_warm_seconds=llm_warm,
        tts_load_seconds=tts_load,
        tts_warm_seconds=tts_warm,
    )
    return pipeline, times


def print_load_times(console: Console, settings: Settings, times: LoadTimes) -> None:
    """Startup summary table."""
    table = Table(title="Model loading (one-off, not included in per-question timings)")
    table.add_column("component")
    table.add_column("detail")
    table.add_column("seconds", justify="right")
    table.add_row("index", times.index_note, f"{times.index_seconds:.2f}")
    table.add_row("embedding model", settings.embed_model, f"{times.embed_warm_seconds:.2f}")
    table.add_row("LLM", settings.llm_model, f"{times.llm_warm_seconds:.2f}")
    if times.tts_load_seconds is not None and times.tts_warm_seconds is not None:
        table.add_row(
            "TTS",
            f"Kokoro-82M ({settings.tts_voice}) load + warm-up",
            f"{times.tts_load_seconds:.2f} + {times.tts_warm_seconds:.2f}",
        )
    console.print(table)


def print_answer(console: Console, answer: Answer) -> None:
    """Show retrieved passages, guard messages, the answer and a compact timing line."""
    table = Table(title="Retrieved passages", title_justify="left", show_lines=False)
    table.add_column("cosine", justify="right")
    table.add_column("bm25", justify="right")
    table.add_column("file", style="cyan")
    table.add_column("passage")
    for hit in answer.retrieved:
        rank = f"#{hit.bm25_rank}" if hit.bm25_rank else "-"
        table.add_row(f"{hit.score:.3f}", rank, hit.chunk.source, hit.chunk.text)
    console.print(table)
    for message in answer.guard_messages:
        console.print(f"[yellow]guard:[/yellow] {message}")
    console.print(Panel(answer.text, title="Answer", border_style="green"))
    t = answer.timings
    tts = f"{t.tts:.2f}s" if t.tts else "skipped"
    llm = f"{t.llm:.2f}s" if answer.llm_used else "skipped"
    console.print(
        f"[dim]timings: retrieval {t.retrieval * 1000:.0f} ms | LLM {llm} | "
        f"TTS generation {tts} | total {t.total:.2f}s[/dim]"
    )


def handle_question(pipeline: Pipeline, console: Console, question: str, play: bool) -> Answer:
    """Answer, display, then play the audio (if any)."""
    answer = pipeline.answer(question, speak=pipeline.speaker is not None)
    print_answer(console, answer)
    if play and pipeline.speaker is not None and answer.audio_path:
        with console.status("speaking..."):
            pipeline.speaker.play(Path(answer.audio_path))
    return answer


def interactive(pipeline: Pipeline, console: Console, play: bool) -> None:
    """Read questions until ``/quit`` or EOF."""
    console.print(HELP)
    while True:
        try:
            line = console.input("\n[bold]you>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        if line in {"/quit", "/exit", "quit", "exit"}:
            return
        if line == "/help":
            console.print(HELP)
            continue
        if line == "/reload":
            with console.status("checking knowledge base..."):
                started = time.perf_counter()
                report = pipeline.reload()
            console.print(
                f"{report.describe()} ({len(pipeline.index)} chunks, "
                f"{time.perf_counter() - started:.2f}s); products: "
                f"{', '.join(p.label for p in pipeline.products)}"
            )
            continue
        if line.startswith("/"):
            console.print(f"unknown command {line!r}; type /help")
            continue
        handle_question(pipeline, console, line, play)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m voice_rag`` and the ``voice-rag`` script."""
    args = parse_args(argv)
    setup_logging(args.verbose)
    console = Console()
    settings = load_settings()
    if args.model:
        settings = Settings(**{**settings.__dict__, "llm_model": args.model})

    try:
        pipeline, times = start(settings, console, use_tts=not args.no_tts, reindex=args.reindex)
    except LLMUnavailableError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    print_load_times(console, settings, times)

    try:
        if args.ask is not None:
            if args.ask.strip():
                handle_question(pipeline, console, args.ask, play=not args.no_audio)
            return 0
        interactive(pipeline, console, play=not args.no_audio)
    except LLMUnavailableError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
