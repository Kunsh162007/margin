"""Run the tools the model calls.

Arguments are validated against the same pydantic models that produced the
schema the model saw (D7). An invalid call is not an exception: the model gets
a message naming what to fix, which small models act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from margin.agent.calc import CalcError, calculate
from margin.agent.tools import validate_arguments
from margin.blueprint.weights import Blueprint, chapter_shares
from margin.retrieve.search import Searcher
from margin.runtime.client import ToolCall
from margin.store.db import Workspace
from margin.visuals.render import Rendered, problems, render

VISUAL_KINDS = {"make_flowchart": "flowchart", "make_table": "table", "make_mindmap": "mindmap",
                "make_timeline": "timeline", "make_formula_sheet": "formula", "make_glossary": "glossary"}
LATER_PHASE = {"generate_questions", "grade_answer", "create_flashcards", "export_notes"}
PASSAGE_CHARS = 600


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    artifact: Rendered | None = None


class ToolExecutor:
    def __init__(self, ws: Workspace | None = None, searcher: Searcher | None = None, blueprint: Blueprint | None = None):
        self._ws = ws
        self._searcher = searcher
        self._blueprint = blueprint
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "search_book": self._search, "read_section": self._read, "get_exam_blueprint": self._blueprint_tool, "calculate": self._calculate,
            **{name: self._visual for name in VISUAL_KINDS},
        }

    def run(self, call: ToolCall) -> ToolResult:
        problem = validate_arguments(call.name, call.arguments)
        if problem:
            return ToolResult(call.name, False, f"Invalid arguments for {call.name}: {problem}. Fix them and call the tool again.")
        if call.name in LATER_PHASE:
            return ToolResult(call.name, False, f"{call.name} is not available yet.")
        args = dict(call.arguments or {}) | {"__name__": call.name}
        try:
            return self._handlers[call.name](args)
        except (CalcError, ValueError, KeyError, LookupError) as exc:
            return ToolResult(call.name, False, f"{call.name} failed: {exc}")

    def _search(self, args: dict[str, Any]) -> ToolResult:
        if self._searcher is None:
            return ToolResult("search_book", False, "No library is open.")
        hits = self._searcher.search(args["query"], top_k=args.get("top_k", 5), scope=args.get("scope"))
        if not hits:
            return ToolResult("search_book", True, "No passages found.")
        passages = [f"[{h.row.sid} {h.row.section_title}, p.{h.row.page_start}] {' '.join(h.row.text.split())[:PASSAGE_CHARS]}" for h in hits]
        return ToolResult("search_book", True, "\n\n".join(passages))

    def _read(self, args: dict[str, Any]) -> ToolResult:
        if self._ws is None:
            return ToolResult("read_section", False, "No library is open.")
        found = self._ws.find_section(args["section_id"])
        if found is None:
            return ToolResult("read_section", False, f"No section {args['section_id']!r} in the library.")
        doc_id, title = found
        return ToolResult("read_section", True, f"[{args['section_id']} {title}]\n{self._ws.section_text(doc_id, args['section_id']) or ''}")

    def _blueprint_tool(self, args: dict[str, Any]) -> ToolResult:
        if self._blueprint is None or not self._blueprint.sections:
            return ToolResult("get_exam_blueprint", False, "No past papers have been added. Run: margin blueprint <papers>")
        top_n = args.get("top_n", 10)
        chapters = ", ".join(f"chapter {c} {s:.0%}" for c, s in chapter_shares(self._blueprint)[:top_n])
        sections = "\n".join(f"{i + 1}. {r.sid} {r.title}: {r.share:.0%}" for i, r in enumerate(self._blueprint.top(top_n)))
        return ToolResult("get_exam_blueprint", True, f"By chapter: {chapters}\nBy section:\n{sections}")

    def _calculate(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult("calculate", True, f"{calculate(args['expression']):.10g}")

    def _visual(self, args: dict[str, Any]) -> ToolResult:
        name = args.pop("__name__")
        rendered = render(VISUAL_KINDS[name], args)
        found = problems(rendered)
        if found:
            return ToolResult(name, False, f"The {rendered.kind} could not be drawn: {'; '.join(found)}", rendered)
        return ToolResult(name, True, f"{rendered.kind.capitalize()} created.", rendered)
