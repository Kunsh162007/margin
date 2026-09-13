"""Run the evaluation suite against one model on one backend.

    uv run python -m evals.run --model qwen3.5-4b                  # smoke tier, auto backend
    uv run python -m evals.run --model qwen3.5-4b --tier full --freeze
    uv run python -m evals.run --model qwen3.5-4b --backend cpu

Text suites and the vision suite run in separate server processes. Loading a
vision projector costs memory, and text-only speed numbers must stay comparable
between models that have one and models that do not.

Exit code 1 when the run FAILs against its baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from evals import regression
from evals.spec import quality_score
from evals.suites import agentic, grounded, speed, structured, vision
from margin import config, hardware
from margin.runtime import models
from margin.runtime.binaries import LLAMA_CPP_BUILD, ensure_llama_server
from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer, ServerConfig

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
BASELINES_DIR = ROOT / "baselines"
IMAGES_DIR = regression.DATASETS_DIR / "images"
SUITES = ("agentic", "grounded", "structured", "speed", "vision")
SMOKE_EVERY = 3
console = Console()

Metrics = dict[str, float | None]
Raw = dict[str, list[dict[str, Any]]]


def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = regression.DATASETS_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_tier(cases: list[dict[str, Any]], tier: str, key: str) -> list[dict[str, Any]]:
    """Smoke keeps every third case *within each group*, so every group stays represented."""
    if tier == "full":
        return cases
    seen: dict[str, int] = {}
    kept = []
    for case in cases:
        group = str(case.get(key, ""))
        if seen.get(group, 0) % SMOKE_EVERY == 0:
            kept.append(case)
        seen[group] = seen.get(group, 0) + 1
    return kept


def _text_suites(client: LlamaClient, server: LlamaServer, suites: tuple[str, ...], tier: str, passages: dict[str, Any], vram_before: float | None) -> tuple[Metrics, Raw]:
    metrics: Metrics = {}
    raw: Raw = {}
    if "speed" in suites:
        m, raw["speed"] = speed.run(client, server, passages, vram_before)
        metrics |= m
    if "agentic" in suites:
        m, raw["agentic"] = agentic.run(client, select_tier(load_jsonl("tool_calls.jsonl"), tier, "category"))
        metrics |= m
    if "grounded" in suites:
        m, raw["grounded"] = grounded.run(client, select_tier(load_jsonl("grounded_qa.jsonl"), tier, "answerable"), passages)
        metrics |= m
    if "structured" in suites:
        m, raw["structured"] = structured.run(client, select_tier(load_jsonl("structured.jsonl"), tier, "task"), passages)
        metrics |= m
    return metrics, raw


def _vision_suite(cfg: ServerConfig, log: Path, tier: str, passages: dict[str, Any]) -> tuple[Metrics, Raw]:
    cases = select_tier(load_jsonl("images.jsonl"), tier, "style")
    with LlamaServer(cfg, log) as server:
        client = LlamaClient(server.base_url)
        try:
            metrics, rows = vision.run(client, cases, passages, IMAGES_DIR)
        finally:
            client.close()
        metrics |= {"vision.load_s": server.load_seconds, "vision.rss_gb": server.rss_gb()}
    return metrics, {"vision": rows}


def evaluate(model_id: str, backend: str | None, tier: str, suites: tuple[str, ...] = SUITES) -> tuple[dict[str, Any], Raw]:
    hw = hardware.detect()
    chosen = hardware.choose_backend(hw, backend)
    paths = config.paths().ensure()
    spec = models.get(model_id)
    model_path = spec.path(paths.models_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"model file missing: {model_path}")
    exe = ensure_llama_server(paths.bin_dir, hw.os, hw.arch, chosen)
    cfg = ServerConfig(exe=exe, model=model_path, gpu=chosen != "cpu", threads=hardware.default_threads(hw), chat_template_file=spec.chat_template_path())
    passages = {p["id"]: p for p in load_jsonl("passages.jsonl")}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = paths.logs_dir / f"{model_id}-{chosen}-{stamp}"

    metrics: Metrics = {}
    raw: Raw = {}
    text_suites = tuple(s for s in suites if s != "vision")
    if text_suites:
        vram_before = speed.gpu_memory_used_gb() if chosen != "cpu" else None
        # with_name, not with_suffix: model ids contain dots ("qwen3.5-4b") that with_suffix would cut
        with LlamaServer(cfg, log_base.with_name(log_base.name + ".log")) as server:
            client = LlamaClient(server.base_url, tools_template_kwarg=spec.tools_template_kwarg)
            try:
                metrics, raw = _text_suites(client, server, text_suites, tier, passages, vram_before)
            finally:
                client.close()

    mmproj = spec.mmproj_path(paths.models_dir)
    if "vision" in suites and mmproj is not None and mmproj.exists():
        m, r = _vision_suite(replace(cfg, mmproj=mmproj), log_base.with_name(log_base.name + "-vision.log"), tier, passages)
        metrics |= m
        raw |= r

    doc = {
        "model": model_id, "backend": chosen, "tier": tier, "suites": list(suites), "run_at": stamp,
        "llama_cpp_build": LLAMA_CPP_BUILD, "fingerprint": regression.fingerprint(),
        "hardware": {"os": hw.os, "cores": hw.physical_cores, "ram_gb": hw.ram_gb, "gpu": hw.gpu.name if hw.gpu else None},
        "quality_score": quality_score(metrics), "metrics": metrics,
    }
    return doc, raw


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def print_comparison(rows: list[regression.Row], verdict: str) -> None:
    table = Table(title=f"vs baseline: {verdict}")
    for col in ("metric", "baseline", "current", "status"):
        table.add_column(col)
    for r in rows:
        table.add_row(r.metric, _fmt(r.baseline), _fmt(r.current), r.status)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(models.BY_ID))
    ap.add_argument("--backend", choices=hardware.BACKENDS)
    ap.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--freeze", action="store_true", help="write this run as the baseline")
    args = ap.parse_args(argv)

    doc, raw = evaluate(args.model, args.backend, args.tier, tuple(args.suites.split(",")))
    name = f"{doc['model']}-{doc['backend']}-{doc['tier']}"
    regression.save(RESULTS_DIR / f"{name}-{doc['run_at']}.json", doc)
    regression.save(RESULTS_DIR / "raw" / f"{name}-{doc['run_at']}.json", raw)
    console.print_json(data={"quality_score": doc["quality_score"], **doc["metrics"]})

    baseline_path = BASELINES_DIR / f"{name}.json"
    if args.freeze:
        regression.save(baseline_path, doc)
        console.print(f"[bold]baseline frozen[/] {baseline_path}")
        return 0
    if not baseline_path.exists():
        console.print(f"no baseline at {baseline_path}; run with --freeze to create one")
        return 0
    rows, verdict = regression.compare(doc, regression.load(baseline_path))
    print_comparison(rows, verdict)
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
