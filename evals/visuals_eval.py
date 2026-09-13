"""Measure diagram repair and rendering on real model output.

    uv run python -m evals.visuals_eval

The model benchmark already asked all nine models for flowcharts, tables, mind
maps, formula sheets, a timeline and a glossary (``evals/datasets/structured.jsonl``)
and saved their raw JSON. This re-uses those outputs, so it costs no model time
and tests the renderer on what small models really produce, not on tidy
examples.

Reported per model and overall: flowcharts whose edges all resolved *before*
repair, flowcharts with usable edges *after* repair, how many edges were
re-linked or dropped, how often the chain fallback was needed, and the share of
outputs that pass the structural render check. Outputs stored truncated
(the benchmark kept the first 1,500 characters) are counted and excluded.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from margin.visuals.render import RENDERERS, problems, render
from margin.visuals.repair import repair_flowchart

RAW_DIR = Path(__file__).parent / "results" / "bench" / "raw"
OUT_DIR = Path(__file__).parent / "results" / "visuals"


def _edges_resolve(data: dict[str, Any]) -> bool:
    ids = {str(s.get("id")) for s in data.get("steps", []) if isinstance(s, dict)}
    edges = [e for e in data.get("edges", []) if isinstance(e, dict)]
    return bool(edges) and all(str(e.get("source")) in ids and str(e.get("target")) in ids for e in edges)


def score_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"outputs": 0, "truncated_or_invalid": 0, "rendered": 0, "render_ok": 0, "flowcharts": 0,
              "edges_valid_before": 0, "edges_usable_after": 0, "edges_relinked": 0, "edges_dropped": 0, "chain_fallbacks": 0}
    failures: list[str] = []
    for row in rows:
        if row.get("task") not in RENDERERS:
            continue
        counts["outputs"] += 1
        try:
            data = json.loads(row.get("output", ""))
        except json.JSONDecodeError:
            counts["truncated_or_invalid"] += 1
            continue
        if row["task"] == "flowchart":
            counts["flowcharts"] += 1
            counts["edges_valid_before"] += int(_edges_resolve(data))
            fixed, report = repair_flowchart(data)
            counts["edges_usable_after"] += int(bool(fixed["edges"]))
            counts["edges_relinked"] += report.edges_relinked
            counts["edges_dropped"] += report.edges_dropped
            counts["chain_fallbacks"] += int(report.chained)
        rendered = render(row["task"], data)
        counts["rendered"] += 1
        found = problems(rendered)
        counts["render_ok"] += int(not found)
        failures += [f"{row['id']}: {p}" for p in found]
    return counts | {"failures": failures}


def main() -> int:
    files = sorted(RAW_DIR.glob("*-cuda-full.json"))
    if not files:
        print(f"no benchmark outputs in {RAW_DIR}; run evals.bench_models first")
        return 1
    per_model = {f.name.removesuffix("-cuda-full.json"): score_model(json.loads(f.read_text(encoding="utf-8")).get("structured", [])) for f in files}
    totals = {k: sum(m[k] for m in per_model.values()) for k in per_model[next(iter(per_model))] if k != "failures"}
    summary = {
        "models": len(per_model),
        **totals,
        "edges_valid_before_rate": round(totals["edges_valid_before"] / max(1, totals["flowcharts"]), 4),
        "edges_usable_after_rate": round(totals["edges_usable_after"] / max(1, totals["flowcharts"]), 4),
        "render_ok_rate": round(totals["render_ok"] / max(1, totals["rendered"]), 4),
    }
    for name, m in per_model.items():
        print(f"{name:>14}: flowcharts {m['flowcharts']}, valid before {m['edges_valid_before']}, usable after {m['edges_usable_after']}, "
              f"relinked {m['edges_relinked']}, dropped {m['edges_dropped']}, chained {m['chain_fallbacks']}, render ok {m['render_ok']}/{m['rendered']}, excluded {m['truncated_or_invalid']}")
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps({"summary": summary, "per_model": per_model, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S")}, indent=2), encoding="utf-8")
    return 0 if summary["render_ok_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
