"""Run the tools the model calls.

Arguments are validated against the same pydantic models that produced the
schema the model saw. An invalid call is not an exception: the model gets
a message naming what to fix, which small models act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from margin.agent.calc import CalcError, calculate
from margin.agent.tools import validate_arguments
from margin.blueprint.weights import Blueprint, chapter_shares
from margin.practice.flashcards import cards_from_notes
from margin.practice.grading import grade_answer
from margin.practice.questions import Question, generate_questions
from margin.retrieve.search import Searcher
from margin.runtime.client import ToolCall
from margin.store.db import Workspace
from margin.visuals.render import Rendered, problems, render

VISUAL_KINDS = {"make_flowchart": "flowchart", "make_table": "table", "make_mindmap": "mindmap",
                "make_timeline": "timeline", "make_formula_sheet": "formula", "make_glossary": "glossary"}
PASSAGE_CHARS = 600
SECTION_CHARS = 6000


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    artifact: Rendered | None = None


class ToolExecutor:
    """Runs tool calls. ``client`` is needed by the tools that ask the model (questions, marking);
    ``export_dir`` is where exports are written. Questions it generates are remembered, because
    marking uses the marking scheme written with the question."""

    def __init__(self, ws: Workspace | None = None, searcher: Searcher | None = None, blueprint: Blueprint | None = None,
                 client: Any = None, export_dir: Path | None = None):
        self._ws = ws
        self._searcher = searcher
        self._blueprint = blueprint
        self._client = client
        self._export_dir = export_dir
        self._questions: dict[str, Question] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "search_book": self._search, "read_section": self._read, "get_exam_blueprint": self._blueprint_tool, "calculate": self._calculate,
            "generate_questions": self._generate, "grade_answer": self._grade, "create_flashcards": self._flashcards, "export_notes": self._export,
            **{name: self._visual for name in VISUAL_KINDS},
        }

    def run(self, call: ToolCall) -> ToolResult:
        problem = validate_arguments(call.name, call.arguments)
        if problem:
            return ToolResult(call.name, False, f"Invalid arguments for {call.name}: {problem}. Fix them and call the tool again.")
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

    def _sections(self, topic: str | None) -> list[tuple[str, str, str]]:
        """Sections a topic names: a chapter or section id, else the best search hit."""
        from margin.notes.writer import sections_in_scope  # the writer imports this module

        found = sections_in_scope(self._ws, topic) if topic else self._ws.teaching_sections()
        if found or self._searcher is None or not topic:
            return found
        hits = self._searcher.search(topic, top_k=1)
        return [(h.row.doc_id, h.row.sid, h.row.section_title) for h in hits]

    def _generate(self, args: dict[str, Any]) -> ToolResult:
        if self._ws is None or self._client is None:
            return ToolResult("generate_questions", False, "No library is open.")
        sections = self._sections(args["topic"])
        if not sections:
            return ToolResult("generate_questions", False, f"Nothing in the library matches {args['topic']!r}.")
        doc_id, sid, title = sections[0]
        text = (self._ws.section_text(doc_id, sid, include_exercises=False) or "")[:SECTION_CHARS]
        checked = generate_questions(self._client, text, sid, args["count"], args["marks"], args["question_type"])
        accepted = [c.question for c in checked if c.accepted]
        self._questions |= {q.question: q for q in accepted}
        lines = [f"{i + 1}. ({q.marks} marks) {q.question}" for i, q in enumerate(accepted)]
        rejected = [f"- {c.question.question[:80]}: {'; '.join(c.reasons)}" for c in checked if not c.accepted]
        body = f"Questions from {sid} {title}:\n" + "\n".join(lines) if accepted else f"No question from {sid} passed the checks."
        return ToolResult("generate_questions", bool(accepted), body + ("\nLeft out:\n" + "\n".join(rejected) if rejected else ""))

    def _grade(self, args: dict[str, Any]) -> ToolResult:
        question = self._questions.get(args["question"].strip())
        if question is None:
            return ToolResult("grade_answer", False, "Only questions written by generate_questions can be marked, because marking uses their marking scheme. Generate questions first and pass one back exactly.")
        if self._client is None:
            return ToolResult("grade_answer", False, "No model is running.")
        grade = grade_answer(self._client, question.question, question.marking_points, args["student_answer"], question.marks)
        missing = "".join(f"\nMissing: {p}" for p in grade.missing)
        return ToolResult("grade_answer", True, f"{grade.marks_awarded}/{grade.max_marks} marks. {grade.feedback}{missing}")

    def _flashcards(self, args: dict[str, Any]) -> ToolResult:
        if self._ws is None:
            return ToolResult("create_flashcards", False, "No library is open.")
        cards = [card for doc_id, sid, _ in self._sections(args["topic"]) if (saved := self._ws.load_notes(doc_id, sid)) for card in cards_from_notes(saved["markdown"], sid)]
        if not cards:
            return ToolResult("create_flashcards", False, f"No saved notes with key terms for {args['topic']!r}. Write notes first.")
        return ToolResult("create_flashcards", True, "\n".join(f"{c.front} — {c.back}" for c in cards[: args["count"]]))

    def _export(self, args: dict[str, Any]) -> ToolResult:
        from margin.export.assets import mermaid_script  # the exporter imports the notes writer, which imports this module
        from margin.export.files import export_notes, saved_notes

        fmt = args["format"]
        if fmt == "pdf":
            return ToolResult("export_notes", False, "PDF export is not available. Export html instead; it prints cleanly to PDF from a browser.")
        if self._ws is None or self._export_dir is None:
            return ToolResult("export_notes", False, "No library is open.")
        topic = args.get("topic")
        notes = saved_notes(self._ws, self._sections(topic))
        if not notes:
            return ToolResult("export_notes", False, "There are no saved notes to export. Write notes first.")
        done = export_notes(notes, fmt, self._export_dir, topic or "all", mermaid_script() if fmt == "html" else None)
        what = f"{done.cards} cards" if fmt == "anki" else f"notes for {done.sections} section(s)"
        return ToolResult("export_notes", True, f"Saved {what} to {done.path}.")

    def _calculate(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult("calculate", True, f"{calculate(args['expression']):.10g}")

    def _visual(self, args: dict[str, Any]) -> ToolResult:
        name = args.pop("__name__")
        rendered = render(VISUAL_KINDS[name], args)
        found = problems(rendered)
        if found:
            return ToolResult(name, False, f"The {rendered.kind} could not be drawn: {'; '.join(found)}", rendered)
        return ToolResult(name, True, f"{rendered.kind.capitalize()} created.", rendered)
