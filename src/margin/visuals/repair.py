"""Repair flowcharts the model got almost right.

The benchmark showed small models filling the flowchart schema correctly but
pointing arrows at a step's *label* instead of its *id* ("Client" -> "Server"
where the steps are "1", "2"). A JSON schema guarantees shape, not that
references resolve, so the fix belongs in code:

1. an edge end that is a step id is kept;
2. one that matches exactly one step label (ignoring case and punctuation), or
   is contained in exactly one label, is re-linked to that step's id;
3. anything else, self-loops and duplicates are dropped;
4. if no usable edge is left, the steps are chained in the order listed — the
   model almost always lists a process in order — and the report says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RepairReport:
    edges_in: int
    edges_kept: int
    edges_relinked: int
    edges_dropped: int
    chained: bool

    @property
    def was_valid(self) -> bool:
        return self.edges_in > 0 and self.edges_relinked == 0 and self.edges_dropped == 0


def _squash(text: str) -> str:
    return _NON_ALNUM.sub("", str(text).lower())


def _resolver(steps: list[dict[str, Any]]):
    ids = {str(s.get("id")) for s in steps}
    labels = [(_squash(s.get("label", "")), str(s.get("id"))) for s in steps]

    def resolve(ref: Any) -> str | None:
        ref = str(ref)
        if ref in ids:
            return ref
        key = _squash(ref)
        if not key:
            return None
        exact = [sid for label, sid in labels if label == key]
        if len(exact) == 1:
            return exact[0]
        partial = [sid for label, sid in labels if label and (key in label or label in key)]
        return partial[0] if len(partial) == 1 else None

    return resolve


def repair_flowchart(data: dict[str, Any]) -> tuple[dict[str, Any], RepairReport]:
    steps = [s for s in data.get("steps", []) if isinstance(s, dict) and "id" in s]
    raw_edges = [e for e in data.get("edges", []) if isinstance(e, dict)]
    resolve = _resolver(steps)
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    relinked = dropped = 0
    for edge in raw_edges:
        source, target = resolve(edge.get("source")), resolve(edge.get("target"))
        if source is None or target is None or source == target or (source, target) in seen:
            dropped += 1
            continue
        relinked += int(source != str(edge.get("source")) or target != str(edge.get("target")))
        seen.add((source, target))
        edges.append({"source": source, "target": target, **({"label": str(edge["label"])} if edge.get("label") else {})})

    chained = not edges and len(steps) > 1
    if chained:
        edges = [{"source": str(a["id"]), "target": str(b["id"])} for a, b in zip(steps, steps[1:])]
    report = RepairReport(len(raw_edges), len(raw_edges) - dropped, relinked, dropped, chained)
    return {**data, "steps": steps, "edges": edges}, report
