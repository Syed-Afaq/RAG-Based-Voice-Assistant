"""Run the six assignment scenarios and the red-team set, and write results/RESULTS.md + JSON.

Usage:
    .venv/bin/python scripts/run_scenarios.py [--models qwen3:4b-instruct,...] [--no-tts]
                                              [--skip-red-team] [--out results]

For every model: unload whatever Ollama has loaded, measure model loading separately, run a
warm-up question, then record retrieval / LLM / TTS-generation / total time per question. Speech is
synthesised (not played) for the six scenarios so TTS generation time is real; the red-team set
runs without TTS to keep the run under half an hour. Pass/fail comes from the keyword checks in
scripts/questions.json and is reported as measured, failures included.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ollama

from voice_rag.config import Settings, load_settings
from voice_rag.embedder import OllamaEmbedder
from voice_rag.llm import OllamaLLM
from voice_rag.pipeline import Pipeline
from voice_rag.tts import KokoroSpeaker
from voice_rag.types import Answer

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "scripts" / "questions.json"
WARM_UP_QUESTION = "Which products do you sell?"


@dataclass
class CaseResult:
    group: str
    question: str
    answer: str
    passed: bool
    note: str
    sources: list[str]
    guards: list[str]
    retrieval_s: float
    llm_s: float
    tts_s: float
    total_s: float
    llm_used: bool
    prompt_tokens: int | None
    output_tokens: int | None
    mem_available_mb: int
    swap_used_mb: int
    name: str = ""


@dataclass
class ModelRun:
    model: str
    digest: str
    size_gb: float
    details: dict[str, Any]
    settings: dict[str, Any]
    load: dict[str, float]
    scenarios: list[CaseResult] = field(default_factory=list)
    red_team: list[CaseResult] = field(default_factory=list)


def memory_snapshot() -> tuple[int, int]:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0])
    return info["MemAvailable"] // 1024, (info["SwapTotal"] - info["SwapFree"]) // 1024


def hardware_description() -> dict[str, str]:
    cpu = "unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    total_kb = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
    os_name = platform.freedesktop_os_release().get("PRETTY_NAME", platform.platform())
    return {
        "cpu": cpu,
        "threads": str(platform.os.cpu_count()),
        "ram_visible_gb": f"{total_kb / 1024 / 1024:.1f}",
        "gpu": "none (CPU-only inference)",
        "os": f"{os_name} on WSL2 ({platform.release()}), Windows host",
        "python": platform.python_version(),
    }


def evaluate(text: str, case: dict[str, Any]) -> tuple[bool, str]:
    """Apply the tolerant regex checks; return (passed, reason)."""
    flags = re.IGNORECASE | re.DOTALL
    reasons: list[str] = []
    if case.get("must_any") and not any(re.search(p, text, flags) for p in case["must_any"]):
        reasons.append("missing expected content")
    for pattern in case.get("must_also", []):
        if not re.search(pattern, text, flags):
            reasons.append(f"missing: /{pattern}/")
    for pattern in case.get("must_not", []):
        if re.search(pattern, text, flags):
            reasons.append(f"forbidden: /{pattern}/")
    return (not reasons, "; ".join(reasons) or "ok")


def to_case(answer: Answer, case: dict[str, Any], group: str) -> CaseResult:
    passed, note = evaluate(answer.text, case)
    mem, swap = memory_snapshot()
    sources = [f"{h.chunk.source} ({h.score:.2f})" for h in answer.retrieved]
    return CaseResult(
        group=group,
        name=case.get("name", ""),
        question=case["question"],
        answer=answer.text,
        passed=passed,
        note=note,
        sources=sources,
        guards=answer.guard_messages,
        retrieval_s=answer.timings.retrieval,
        llm_s=answer.timings.llm,
        tts_s=answer.timings.tts,
        total_s=answer.timings.total,
        llm_used=answer.llm_used,
        prompt_tokens=answer.timings.ollama.get("prompt_eval_count"),
        output_tokens=answer.timings.ollama.get("eval_count"),
        mem_available_mb=mem,
        swap_used_mb=swap,
    )


def unload_all(client: ollama.Client) -> None:
    for running in client.ps().models:
        if running.model:
            client.generate(model=running.model, keep_alive=0)


def model_info(client: ollama.Client, tag: str) -> tuple[str, float, dict[str, Any]]:
    digest, size = "", 0.0
    for m in client.list().models:
        if m.model in (tag, f"{tag}:latest"):
            digest, size = (m.digest or "")[:12], (m.size or 0) / 1e9
    show = client.show(tag)
    d = show.details
    details = {
        "family": d.family if d else None,
        "parameter_size": d.parameter_size if d else None,
        "quantization": d.quantization_level if d else None,
    }
    return digest, size, details


def run_model(tag: str, base: Settings, use_tts: bool, skip_red_team: bool, out: Path) -> ModelRun:
    settings = Settings(**{**base.__dict__, "llm_model": tag, "audio_dir": out / "audio"})
    client = ollama.Client(host=settings.ollama_host)
    unload_all(client)
    time.sleep(2)
    digest, size, details = model_info(client, tag)
    print(f"\n=== {tag} ({digest}, {size:.1f} GB) ===")

    load: dict[str, float] = {}
    embedder = OllamaEmbedder(client, settings.embed_model, keep_alive=settings.keep_alive)
    llm = OllamaLLM(client, settings)
    started = time.perf_counter()
    pipeline = Pipeline(settings, embedder, llm)
    pipeline.reload(force=True)
    load["index_build_s"] = time.perf_counter() - started
    started = time.perf_counter()
    embedder.embed_query("warm-up")
    load["embed_model_s"] = time.perf_counter() - started
    load["llm_model_s"] = llm.warm_up()
    if use_tts:
        speaker = KokoroSpeaker(settings, audio_dir=out / "audio")
        load["tts_load_s"] = speaker.load_seconds
        load["tts_warm_s"] = speaker.warm_up()
        pipeline.speaker = speaker
    for key, value in load.items():
        print(f"  load {key}: {value:.2f}")

    pipeline.answer(WARM_UP_QUESTION, speak=use_tts)  # warm prompt cache; not recorded
    questions = json.loads(QUESTIONS.read_text())
    run = ModelRun(
        model=tag,
        digest=digest,
        size_gb=round(size, 2),
        details=details,
        settings={k: str(v) for k, v in settings.__dict__.items()},
        load=load,
    )
    for case in questions["scenarios"]:
        answer = pipeline.answer(case["question"], speak=use_tts)
        result = to_case(answer, case, group=f"scenario {case['id']}")
        run.scenarios.append(result)
        print(
            f"  [{'PASS' if result.passed else 'FAIL'}] S{case['id']} {result.total_s:.1f}s: {answer.text[:90]}"
        )
    if not skip_red_team:
        for case in questions["red_team"]:
            answer = pipeline.answer(case["question"], speak=False)
            result = to_case(answer, case, group=case["group"])
            run.red_team.append(result)
            print(f"  [{'PASS' if result.passed else 'FAIL'}] {case['group']}: {answer.text[:80]}")
    return run


def fmt(seconds: float) -> str:
    return f"{seconds:.2f}"


def write_markdown(
    runs: list[ModelRun], hardware: dict[str, str], out: Path, use_tts: bool
) -> None:
    lines = ["# Test results", ""]
    lines.append(
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/run_scenarios.py`."
    )
    lines.append("")
    lines.append("## Hardware")
    lines.append("")
    for k, v in hardware.items():
        lines.append(f"- **{k}**: {v}")
    lines.append(
        "- **memory cap**: WSL2 limited to 5 GB RAM + 4 GB swap via `.wslconfig` (see README issues)"
    )
    lines.append("")
    for run in runs:
        lines.append(f"## Model `{run.model}`")
        lines.append("")
        lines.append(f"- digest `{run.digest}`, {run.size_gb} GB on disk, {run.details}")
        s = run.settings
        lines.append(
            f"- settings: temperature {s['temperature']}, seed {s['seed']}, num_ctx {s['num_ctx']}, "
            f"num_predict {s['num_predict']}, num_thread {s['num_thread']}, top_k {s['top_k']}, "
            f"min_similarity {s['min_similarity']}, embed model {s['embed_model']}"
        )
        lines.append("")
        lines.append("### Model loading (one-off, excluded from per-question timings)")
        lines.append("")
        lines.append("| step | seconds |")
        lines.append("|---|---:|")
        for k, v in run.load.items():
            lines.append(f"| {k} | {fmt(v)} |")
        lines.append("")
        lines.append("### Six required scenarios")
        lines.append("")
        lines.append(
            "| # | question | answer | sources (cosine) | result | retrieval s | LLM s | TTS gen s | total s |"
        )
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for r in run.scenarios:
            llm = fmt(r.llm_s) if r.llm_used else "skipped"
            tts = fmt(r.tts_s) if r.tts_s else ("skipped" if not use_tts else "0.00")
            mark = "PASS" if r.passed else f"FAIL ({r.note})"
            lines.append(
                f"| {r.group.split()[-1]} | {r.question} | {r.answer} | {'; '.join(r.sources)} | {mark} | "
                f"{fmt(r.retrieval_s)} | {llm} | {tts} | {fmt(r.total_s)} |"
            )
        lines.append("")
        passed = sum(r.passed for r in run.scenarios)
        lines.append(f"Scenarios passed: **{passed}/{len(run.scenarios)}**.")
        lines.append("")
        if run.scenarios:
            lines.append("Guard activity:")
            lines.append("")
            for r in run.scenarios:
                for g in r.guards:
                    lines.append(f"- S{r.group.split()[-1]}: {g}")
            lines.append("")
        if run.red_team:
            lines.append("### Red-team set")
            lines.append("")
            lines.append("| group | question | answer | result | LLM s | guards |")
            lines.append("|---|---|---|---|---:|---|")
            for r in run.red_team:
                mark = "PASS" if r.passed else f"FAIL ({r.note})"
                llm = fmt(r.llm_s) if r.llm_used else "skipped"
                guards = "; ".join(g.split(";")[0] for g in r.guards) or "-"
                lines.append(
                    f"| {r.group} | {r.question} | {r.answer} | {mark} | {llm} | {guards} |"
                )
            lines.append("")
            groups: dict[str, list[CaseResult]] = {}
            for r in run.red_team:
                groups.setdefault(r.group, []).append(r)
            lines.append("| group | passed |")
            lines.append("|---|---:|")
            for g, rs in groups.items():
                lines.append(f"| {g} | {sum(r.passed for r in rs)}/{len(rs)} |")
            total = sum(r.passed for r in run.red_team)
            lines.append(f"| **all** | **{total}/{len(run.red_team)}** |")
            lines.append("")
        timed = [r for r in run.scenarios + run.red_team if r.llm_used]
        if timed:
            lines.append("### Timing summary (questions that reached the LLM)")
            lines.append("")
            llm_times = sorted(r.llm_s for r in timed)
            lines.append(
                f"- LLM time: min {fmt(llm_times[0])} s, median {fmt(llm_times[len(llm_times) // 2])} s, max {fmt(llm_times[-1])} s over {len(timed)} questions"
            )
            lines.append(
                f"- retrieval time: median {fmt(sorted(r.retrieval_s for r in timed)[len(timed) // 2])} s"
            )
            if use_tts and run.scenarios:
                tts_times = [r.tts_s for r in run.scenarios if r.tts_s]
                if tts_times:
                    lines.append(
                        f"- TTS generation (six scenarios): min {fmt(min(tts_times))} s, max {fmt(max(tts_times))} s"
                    )
            swaps = [r.swap_used_mb for r in timed]
            avail = [r.mem_available_mb for r in timed]
            lines.append(
                f"- memory during run: available {min(avail)}-{max(avail)} MB, swap used {min(swaps)}-{max(swaps)} MB"
            )
            lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", default="qwen3:4b-instruct", help="comma-separated Ollama tags")
    parser.add_argument("--no-tts", action="store_true", help="skip speech synthesis")
    parser.add_argument("--skip-red-team", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_settings()
    hardware = hardware_description()
    runs = [
        run_model(tag.strip(), base, not args.no_tts, args.skip_red_team, out)
        for tag in args.models.split(",")
    ]
    payload = {
        "generated": datetime.now(UTC).isoformat(),
        "hardware": hardware,
        "runs": [asdict(r) for r in runs],
    }
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(runs, hardware, out, not args.no_tts)
    for run in runs:
        print(
            f"\n{run.model}: scenarios {sum(r.passed for r in run.scenarios)}/{len(run.scenarios)}, "
            f"red-team {sum(r.passed for r in run.red_team)}/{len(run.red_team)}"
        )
    print(f"wrote {out / 'RESULTS.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()
