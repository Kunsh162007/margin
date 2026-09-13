"""Everything the study interface does, without the interface.

The Textual app (``tui/app.py``) only draws and forwards what the student does;
each action is a blocking method here that the app runs on a worker thread. Every
method opens its own workspace connection, because a SQLite connection belongs to
the thread that opened it. The model and the embedder are passed in, so tests and
the offline check drive the same flow with a scripted model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from margin.agent.executor import ToolExecutor
from margin.agent.loop import ClientLike, LoopResult, run_agent
from margin.agent.tools import openai_tools
from margin.blueprint.paper import paper_text, parse_paper
from margin.blueprint.weights import Blueprint, build_blueprint
from margin.export.assets import mermaid_script
from margin.export.files import Exported, export_notes, saved_notes
from margin.ingest.detect import UnsupportedFile
from margin.notes.writer import SectionNotes, sections_in_scope, write_notes
from margin.practice.grading import Grade, grade_answer
from margin.practice.questions import Question, QuestionType, generate_for_sections
from margin.retrieve.search import EmbedderLike, IndexResult, Searcher, index_document
from margin.store.db import Workspace

SECTION_CHARS = 6000
HISTORY_TURNS = 6
SYSTEM_PROMPT = (
    "You are Margin, a study assistant working from the student's own library. Before answering a question about the "
    "material, search the library or read the relevant section, and say which section the answer comes from. When a "
    "table, flowchart, mind map, timeline, formula sheet or glossary would make something clearer, make one with the "
    "tool. Use calculate for arithmetic. If the library does not cover something, say so plainly."
)


class Study:
    def __init__(self, workspace: Path, export_dir: Path, client_factory: Callable[[], ClientLike],
                 embedder: EmbedderLike | None = None, mermaid_js: Callable[[], str | None] = mermaid_script):
        self.workspace = workspace
        self.export_dir = export_dir
        self._client_factory = client_factory
        self._embedder = embedder
        self._mermaid_js = mermaid_js
        self.blueprint: Blueprint | None = None
        self.questions: list[Question] = []
        self.history: list[dict[str, Any]] = []

    def embedder(self) -> EmbedderLike:
        if self._embedder is None:
            from margin.retrieve.embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    # library -------------------------------------------------------------------

    def documents(self) -> list[dict[str, object]]:
        with Workspace.open(self.workspace) as ws:
            return ws.documents()

    def add(self, paths: list[Path], on_file: Callable[[Path, str], None] | None = None) -> list[IndexResult]:
        added: list[IndexResult] = []
        with Workspace.open(self.workspace) as ws:
            for path in paths:
                try:
                    result = index_document(ws, path, self.embedder())
                except (FileNotFoundError, UnsupportedFile) as exc:
                    if on_file:
                        on_file(path, f"skipped: {exc}")
                    continue
                added.append(result)
                if on_file:
                    on_file(path, "already in the library" if result.skipped else f"{result.title}: {result.sections} sections, {result.chunks} passages")
        return added

    def read_papers(self, paths: list[Path]) -> Blueprint:
        questions = []
        for path in paths:
            try:
                questions += parse_paper(paper_text(path), source=path.stem)
            except (FileNotFoundError, UnsupportedFile):
                continue
        if not questions:
            raise ValueError("No questions were found in those papers.")
        with Workspace.open(self.workspace) as ws:
            if ws.chunk_count() == 0:
                raise ValueError("Add your books to the library first.")
            self.blueprint = build_blueprint(Searcher(ws, self.embedder()), questions)
        return self.blueprint

    def sections(self, scope: str | None) -> list[tuple[str, str, str]]:
        with Workspace.open(self.workspace) as ws:
            return sections_in_scope(ws, scope)

    # notes -----------------------------------------------------------------------

    def write_notes(self, scope: str, visuals: bool = True,
                    on_progress: Callable[[str, SectionNotes, bool], None] | None = None) -> list[SectionNotes]:
        client = self._client_factory()
        with Workspace.open(self.workspace) as ws:
            executor = ToolExecutor(ws, Searcher(ws, self.embedder()), self.blueprint)
            return write_notes(client, ws, scope=scope, executor=executor, on_progress=on_progress, visuals=visuals)

    def export(self, scope: str, fmt: str) -> Exported:
        with Workspace.open(self.workspace) as ws:
            notes = saved_notes(ws, sections_in_scope(ws, scope))
        if not notes:
            raise ValueError(f"There are no saved notes for {scope!r} yet. Write them first.")
        return export_notes(notes, fmt, self.export_dir, scope, self._mermaid_js() if fmt == "html" else None)

    # practice ----------------------------------------------------------------------

    def generate_questions(self, scope: str, count: int, marks: int, kind: QuestionType = "short") -> tuple[list[Question], int]:
        with Workspace.open(self.workspace) as ws:
            texts = [(sid, (ws.section_text(doc_id, sid, include_exercises=False) or "")[:SECTION_CHARS]) for doc_id, sid, _ in sections_in_scope(ws, scope)]
        if not texts:
            raise ValueError(f"No sections match {scope!r}.")
        checked = generate_for_sections(self._client_factory(), texts, count, marks, kind)
        self.questions = [c.question for c in checked if c.accepted]
        return self.questions, sum(not c.accepted for c in checked)

    def mark(self, index: int, answer: str) -> Grade:
        q = self.questions[index]
        return grade_answer(self._client_factory(), q.question, q.marking_points, answer, q.marks)

    # ask ------------------------------------------------------------------------------

    def ask(self, text: str) -> LoopResult:
        client = self._client_factory()
        with Workspace.open(self.workspace) as ws:
            executor = ToolExecutor(ws, Searcher(ws, self.embedder()), self.blueprint, client, self.export_dir)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history[-2 * HISTORY_TURNS:], {"role": "user", "content": text}]
            result = run_agent(client, executor, messages, openai_tools())
        self.history = [*self.history, {"role": "user", "content": text}, {"role": "assistant", "content": result.final or "I stopped after the step limit."}]
        return result
