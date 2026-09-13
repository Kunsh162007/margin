"""Compare a run with a frozen baseline.

A baseline is only comparable with a run over the same datasets and the same
runtime build. Both are hashed into a fingerprint, and a mismatch refuses the
comparison instead of silently comparing different things.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.spec import BY_ID, MetricSpec
from margin.runtime.binaries import LLAMA_CPP_BUILD

DATASETS_DIR = Path(__file__).parent / "datasets"


class FingerprintMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Row:
    metric: str
    baseline: float | None
    current: float | None
    status: str  # PASS | SUSPECT | FAIL | NEW | MISSING | INFO


def fingerprint(datasets_dir: Path = DATASETS_DIR, build: str = LLAMA_CPP_BUILD) -> str:
    h = hashlib.sha256(build.encode())
    for f in sorted([*datasets_dir.glob("*.jsonl"), *datasets_dir.glob("images/*")]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def classify(spec: MetricSpec, baseline: float | None, current: float | None) -> str:
    if baseline is None:
        return "NEW"
    if current is None:
        return "MISSING"
    worse_by = (baseline - current) if spec.direction == "higher" else (current - baseline)
    tolerance = spec.band * abs(baseline) if spec.relative else spec.band
    if worse_by <= tolerance:
        status = "PASS"
    elif worse_by <= 2 * tolerance:
        status = "SUSPECT"
    else:
        status = "FAIL"
    return status if spec.gated else ("INFO" if status == "PASS" else f"INFO-{status}")


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[list[Row], str]:
    if current["fingerprint"] != baseline["fingerprint"]:
        raise FingerprintMismatch(
            f"baseline fingerprint {baseline['fingerprint']} != run {current['fingerprint']}: "
            "datasets or runtime changed, so the numbers are not comparable. Re-baseline with --freeze."
        )
    rows = []
    for metric_id, spec in BY_ID.items():
        base, cur = baseline["metrics"].get(metric_id), current["metrics"].get(metric_id)
        if base is None and cur is None:
            continue
        rows.append(Row(metric_id, base, cur, classify(spec, base, cur)))
    statuses = {r.status for r in rows}
    verdict = "FAIL" if "FAIL" in statuses or "MISSING" in statuses else "SUSPECT" if "SUSPECT" in statuses else "PASS"
    return rows, verdict


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
