"""Render structured study material as Mermaid diagrams or Markdown.

Model text never reaches Mermaid unescaped: every label is quoted and its
quotes, brackets and line breaks are neutralised, and node ids are generated,
not taken from the model. Flowcharts are repaired (``repair.py``) before they
are drawn.

``problems()`` is a *structural* check — diagram type, every referenced node
declared, balanced brackets, table rows matching the header. It is not the
Mermaid parser, which needs Node.js; the exported HTML renders diagrams in the
browser, where a real parse happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from margin.visuals.repair import repair_flowchart

_LATEX_HINT = re.compile(r"[\\^_{}]")
_MERMAID_TYPES = ("flowchart", "mindmap", "timeline")
_NODE_REF = re.compile(r"\b(n\d+)\b")
_NODE_DECL = re.compile(r"\b(n\d+)\[")


@dataclass(frozen=True)
class Rendered:
    kind: str
    format: str  # "mermaid" or "markdown"
    text: str


def _mermaid_text(text: Any) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned.replace('"', "#quot;").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def flowchart(data: dict[str, Any]) -> str:
    fixed, _ = repair_flowchart(data)
    node = {str(step["id"]): f"n{i}" for i, step in enumerate(fixed["steps"])}
    lines = ["flowchart TD"]
    lines += [f'    {node[str(s["id"])]}["{_mermaid_text(s.get("label", s["id"]))}"]' for s in fixed["steps"]]
    for edge in fixed["edges"]:
        arrow = f' -->|"{_mermaid_text(edge["label"])}"| ' if edge.get("label") else " --> "
        lines.append(f"    {node[edge['source']]}{arrow}{node[edge['target']]}")
    return "\n".join(lines)


def mindmap(data: dict[str, Any]) -> str:
    counter = iter(range(1, 10_000))
    lines = ["mindmap", f'  n0["{_mermaid_text(data.get("root", "Topic"))}"]']
    for branch in data.get("branches", []):
        lines.append(f'    n{next(counter)}["{_mermaid_text(branch.get("label", ""))}"]')
        lines += [f'      n{next(counter)}["{_mermaid_text(child)}"]' for child in branch.get("children", [])]
    return "\n".join(lines)


def timeline(data: dict[str, Any]) -> str:
    lines = ["timeline", f"    title {_mermaid_text(data.get('title', 'Timeline')).replace(':', ' -')}"]
    for event in data.get("events", []):
        when = _mermaid_text(event.get("when", "")).replace(":", " -")
        what = _mermaid_text(event.get("what", "")).replace(":", " -")
        lines.append(f"    {when} : {what}")
    return "\n".join(lines)


def table(data: dict[str, Any]) -> str:
    columns = [_cell(c) for c in data.get("columns", [])] or ["Item"]
    lines = [f"**{_cell(data.get('title', ''))}**", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in data.get("rows", []):
        cells = [_cell(c) for c in (row if isinstance(row, list) else [row])][: len(columns)]
        cells += [""] * (len(columns) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def formula_sheet(data: dict[str, Any]) -> str:
    lines = [f"**{_cell(data.get('title', 'Formulas'))}**", "", "| Name | Formula | Symbols |", "|---|---|---|"]
    for f in data.get("formulas", []):
        expr = str(f.get("expression", ""))
        shown = f"${expr}$" if _LATEX_HINT.search(expr) else f"`{expr}`"
        lines.append(f"| {_cell(f.get('name', ''))} | {_cell(shown)} | {_cell(f.get('variables', ''))} |")
    return "\n".join(lines)


def glossary(data: dict[str, Any]) -> str:
    return "\n".join(f"- **{_cell(t.get('term', ''))}** — {_cell(t.get('definition', ''))}" for t in data.get("terms", []))


RENDERERS: dict[str, tuple[str, Callable[[dict[str, Any]], str]]] = {
    "flowchart": ("mermaid", flowchart),
    "mindmap": ("mermaid", mindmap),
    "timeline": ("mermaid", timeline),
    "table": ("markdown", table),
    "formula": ("markdown", formula_sheet),
    "glossary": ("markdown", glossary),
}


def render(kind: str, data: dict[str, Any]) -> Rendered:
    if kind not in RENDERERS:
        raise ValueError(f"no renderer for {kind!r}; known: {', '.join(RENDERERS)}")
    fmt, fn = RENDERERS[kind]
    return Rendered(kind, fmt, fn(data))


def problems(rendered: Rendered) -> list[str]:
    """Structural problems that would stop a diagram or table from rendering. Empty means it passes."""
    found: list[str] = []
    lines = rendered.text.splitlines()
    if rendered.format == "mermaid":
        if not lines or not lines[0].startswith(_MERMAID_TYPES):
            found.append("unknown diagram type")
        declared = set(_NODE_DECL.findall(rendered.text))
        undeclared = {ref for ref in _NODE_REF.findall(rendered.text) if ref not in declared}
        if undeclared and rendered.kind == "flowchart":
            found.append(f"undeclared nodes: {sorted(undeclared)}")
        for i, line in enumerate(lines[1:], start=2):
            if line.count("[") != line.count("]") or line.count('"') % 2:
                found.append(f"unbalanced brackets or quotes on line {i}")
        if rendered.kind == "flowchart" and len(lines) < 2:
            found.append("empty flowchart")
    else:
        header = next((ln for ln in lines if ln.startswith("| ")), None)
        if header is not None:
            width = header.count(" | ") + 1
            bad = [ln for ln in lines if ln.startswith("| ") and ln.replace("\\|", "").count(" | ") + 1 != width]
            if bad:
                found.append(f"{len(bad)} table row(s) with the wrong number of cells")
    return found
