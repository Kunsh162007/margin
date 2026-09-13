"""Benchmark every candidate model and write the comparison.

Quality is a property of the weights and quantisation, so it is measured once,
on the fastest backend, over the full tier. Speed is a property of the
hardware, so it is measured on every backend. The CPU pass also re-runs the
smoke tier, which confirms the CPU build produces the same quality.

    uv run python -m evals.bench_models                    # all models on disk, gpu full + cpu smoke
    uv run python -m evals.bench_models --models qwen3.5-4b,gemma4-e4b --backends cpu

Results already on disk are reused, so an interrupted benchmark resumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from evals import regression
from evals.run import RESULTS_DIR, evaluate
from margin import config, hardware
from margin.runtime import models
from margin.runtime.server import ServerError

BENCH_DIR = RESULTS_DIR / "bench"
console = Console()

REPORT_COLUMNS = (
    ("quality_score", "Quality", 3),
    ("agent.args_acc", "Tool args", 3),
    ("agent.irrelevance_acc", "No-tool", 3),
    ("agent.parallel_acc", "Parallel", 3),
    ("agent.task_success_rate", "Tasks", 3),
    ("agent.schema_valid_rate", "Valid calls", 3),
    ("agent.text_parsed_call_rate", "Text-parsed", 2),
    ("grounded.answer_acc", "QA", 3),
    ("grounded.refusal_acc", "Refusal", 3),
    ("structured.key_term_coverage", "Coverage", 3),
    ("notes.sentence_support_rate", "Notes support", 3),
)
SPEED_COLUMNS = (
    ("speed.gen_tps", "Gen tok/s", 1),
    ("speed.prompt_tps", "Prompt tok/s", 0),
    ("speed.ttft_long_s", "TTFT 2k (s)", 2),
    ("agent.call_latency_mean_s", "Tool call (s)", 2),
    ("speed.rss_gb", "RAM (GB)", 2),
    ("speed.vram_gb", "VRAM (GB)", 2),
    ("speed.load_s", "Load (s)", 1),
)
VISION_COLUMNS = (
    ("vision.cer_typed", "CER typed", 3),
    ("vision.cer_handwriting", "CER hand", 3),
    ("vision.latency_mean_s", "s / image", 1),
    ("vision.rss_gb", "RAM (GB)", 2),
    ("vision.ocr_cer_typed", "OCR CER typed", 3),
    ("vision.ocr_cer_handwriting", "OCR CER hand", 3),
    ("vision.ocr_latency_mean_s", "OCR s / image", 2),
)


CPU_TOOL_CALL_LIMIT_S = 10.0
RAM_LIMIT_GB = 8.0
VRAM_LIMIT_GB = 8.0


def select_defaults(docs: dict[tuple[str, str], dict[str, Any]], quality_backend: str) -> dict[str, str | None]:
    """Apply the selection rule fixed in DESIGN.md D3 before any result was seen.

    Quality comes from the quality backend's full-tier run. CPU default: best quality
    whose CPU tool decision takes <= 10 s and whose server memory (text or vision) fits
    8 GB. GPU default: best quality whose model fits 8 GB of VRAM.
    """
    quality = {m: d["quality_score"] for (m, b), d in docs.items() if b == quality_backend and d.get("quality_score") is not None}

    def cpu_eligible(model_id: str) -> bool:
        doc = docs.get((model_id, "cpu"))
        if doc is None:
            return False
        m = doc["metrics"]
        latency = m.get("agent.call_latency_mean_s")
        memory = max((v for v in (m.get("speed.rss_gb"), m.get("vision.rss_gb")) if v is not None), default=None)
        return latency is not None and memory is not None and latency <= CPU_TOOL_CALL_LIMIT_S and memory <= RAM_LIMIT_GB

    def gpu_eligible(model_id: str) -> bool:
        vram = docs[(model_id, quality_backend)]["metrics"].get("speed.vram_gb")
        return vram is not None and vram <= VRAM_LIMIT_GB

    def best(candidates: list[str]) -> str | None:
        return max(candidates, key=lambda m: quality[m]) if candidates else None

    return {
        "cpu": best([m for m in quality if cpu_eligible(m)]),
        "gpu": best([m for m in quality if gpu_eligible(m)]) if quality_backend != "cpu" else None,
    }


def plan(model_ids: list[str], backends: list[str]) -> list[tuple[str, str, str]]:
    runs = []
    for model_id in model_ids:
        for backend in backends:
            runs.append((model_id, backend, "smoke" if backend == "cpu" else "full"))
    return runs


def run_one(model_id: str, backend: str, tier: str) -> dict[str, Any] | None:
    out = BENCH_DIR / f"{model_id}-{backend}-{tier}.json"
    if out.exists():
        return regression.load(out)
    console.rule(f"{model_id} · {backend} · {tier}")
    try:
        doc, raw = evaluate(model_id, backend, tier)
    except (ServerError, FileNotFoundError) as exc:
        console.print(f"[red]skipped[/] {model_id} on {backend}: {exc}")
        return None
    regression.save(out, doc)
    regression.save(BENCH_DIR / "raw" / out.name, raw)
    console.print(f"quality {doc['quality_score']:.3f}  gen {doc['metrics'].get('speed.gen_tps') or 0:.1f} tok/s")
    return doc


def _cell(doc: dict[str, Any] | None, key: str, digits: int) -> str:
    if doc is None:
        return "—"
    value = doc.get(key) if key == "quality_score" else doc["metrics"].get(key)
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_tables(docs: dict[tuple[str, str], dict[str, Any]], model_ids: list[str], backends: list[str]) -> str:
    lines = []
    quality_backend = next((b for b in backends if b != "cpu"), backends[0])
    lines.append(f"### Quality (backend: {quality_backend})\n")
    lines.append("| Model | " + " | ".join(label for _, label, _ in REPORT_COLUMNS) + " |")
    lines.append("|---|" + "---|" * len(REPORT_COLUMNS))
    for m in model_ids:
        doc = docs.get((m, quality_backend))
        lines.append(f"| {m} | " + " | ".join(_cell(doc, k, d) for k, _, d in REPORT_COLUMNS) + " |")
    for backend in backends:
        lines.append(f"\n### Speed and memory (backend: {backend})\n")
        lines.append("| Model | " + " | ".join(label for _, label, _ in SPEED_COLUMNS) + " |")
        lines.append("|---|" + "---|" * len(SPEED_COLUMNS))
        for m in model_ids:
            doc = docs.get((m, backend))
            lines.append(f"| {m} | " + " | ".join(_cell(doc, k, d) for k, _, d in SPEED_COLUMNS) + " |")
        vision_models = [m for m in model_ids if docs.get((m, backend)) and docs[(m, backend)]["metrics"].get("vision.cer_mean") is not None]
        if vision_models:
            lines.append(f"\n### Reading photos (backend: {backend})\n")
            lines.append("| Model | " + " | ".join(label for _, label, _ in VISION_COLUMNS) + " |")
            lines.append("|---|" + "---|" * len(VISION_COLUMNS))
            for m in vision_models:
                lines.append(f"| {m} | " + " | ".join(_cell(docs[(m, backend)], k, d) for k, _, d in VISION_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated ids; default: every candidate on disk")
    ap.add_argument("--backends", help="comma-separated; default: detected GPU backend then cpu")
    args = ap.parse_args(argv)

    models_dir = config.paths().ensure().models_dir
    model_ids = args.models.split(",") if args.models else [m.id for m in models.CANDIDATES if m.path(models_dir).exists()]
    detected = hardware.choose_backend(hardware.detect())
    backends = args.backends.split(",") if args.backends else ([detected, "cpu"] if detected != "cpu" else ["cpu"])

    docs: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id, backend, tier in plan(model_ids, backends):
        doc = run_one(model_id, backend, tier)
        if doc is not None:
            docs[(model_id, backend)] = doc

    write_report()
    return 0


def write_report() -> Path:
    """Tables over every result on disk, not just this invocation, plus the rule's picks."""
    all_docs = {(d["model"], d["backend"]): d for d in (regression.load(f) for f in BENCH_DIR.glob("*.json"))}
    order = {m.id: i for i, m in enumerate(models.CANDIDATES)}
    model_ids = sorted({m for m, _ in all_docs}, key=lambda m: order.get(m, len(order)))
    backends = sorted({b for _, b in all_docs}, key=lambda b: (b == "cpu", b))
    quality_backend = next((b for b in backends if b != "cpu"), "cpu")
    defaults = select_defaults(all_docs, quality_backend)
    rule = (
        f"\n### Defaults by the selection rule (D3)\n\n"
        f"- CPU default (quality from {quality_backend}; CPU tool decision <= {CPU_TOOL_CALL_LIMIT_S:.0f} s, memory <= {RAM_LIMIT_GB:.0f} GB): **{defaults['cpu'] or 'none eligible yet'}**\n"
        f"- GPU default (fits {VRAM_LIMIT_GB:.0f} GB VRAM): **{defaults['gpu'] or 'none eligible yet'}**\n"
    )
    tables = markdown_tables(all_docs, model_ids, backends) + rule
    report = BENCH_DIR / "tables.md"
    report.write_text(tables, encoding="utf-8")
    sync_design_doc(tables)
    console.print(f"wrote {report}")
    return report


DESIGN_DOC = Path(__file__).resolve().parents[1] / "docs" / "DESIGN.md"
TABLES_START = "<!-- bench-tables:start -->"
TABLES_END = "<!-- bench-tables:end -->"


def sync_design_doc(tables: str, doc: Path = DESIGN_DOC) -> bool:
    """Replace the text between the table markers so the design doc cannot drift from the results."""
    if not doc.exists():
        return False
    text = doc.read_text(encoding="utf-8")
    start, end = text.find(TABLES_START), text.find(TABLES_END)
    if start < 0 or end < start:
        return False
    updated = text[: start + len(TABLES_START)] + "\n\n" + tables.strip() + "\n\n" + text[end:]
    doc.write_text(updated, encoding="utf-8")
    return True


if __name__ == "__main__":
    sys.exit(main())
