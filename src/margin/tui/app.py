"""The Margin study interface.

Five tabs in the order a student works: Library, Blueprint, Notes, Practice, Ask.
The app only draws. ``margin.study.Study`` does the work on a worker thread, one
job at a time, and reports back through ``call_from_thread``. Progress shows work
already finished — sections written, files read — never a predicted finish time,
because on a CPU the time per section varies too much to promise one.

Text from books and from the model is always inserted as plain text or Markdown,
never as console markup, so a stray ``[bold]`` in a document cannot restyle the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, Markdown, OptionList, ProgressBar,
                             RichLog, Static, TabbedContent, TabPane, TextArea)

from margin.agent.loop import LoopResult
from margin.blueprint.weights import Blueprint, chapter_shares
from margin.notes.writer import SectionNotes, to_markdown
from margin.practice.grading import Grade
from margin.practice.questions import Question
from margin.study import Study

GOLD = "#d4a72c"
TOP_ROWS = 15
MARGIN_THEME = Theme(
    name="margin", primary=GOLD, secondary="#8f6a10", accent="#e8c766", foreground="#ebe5d6", background="#14120e",
    surface="#1c1914", panel="#262118", success="#8fb573", warning="#e0a03a", error="#d9674e", dark=True,
)

CSS = """
#tabs, #tabs ContentSwitcher { height: 1fr; }
TabPane { height: 1fr; padding: 1 2; }
.hint { color: $text-muted; margin-bottom: 1; }
.row { height: auto; margin-bottom: 1; }
.row Input { width: 1fr; }
.row .field { padding: 1 0 0 2; color: $text-muted; width: auto; }
.row Button, .row Checkbox { margin-left: 1; }
#count, #marks { width: 12; }
#documents { height: 1fr; }
#library-log { height: 8; border-top: solid $primary 40%; }
#blueprint-tables { height: 1fr; }
#blueprint-tables DataTable { width: 1fr; margin-right: 1; }
#notes-progress { margin-bottom: 1; }
#notes-scroll { height: 1fr; border: round $primary 40%; padding: 0 1; }
#practice-body { height: 1fr; }
#questions { width: 40%; border: round $primary 40%; }
#question-pane { padding-left: 2; }
#question-text { height: auto; min-height: 4; margin-bottom: 1; }
#answer { height: 8; margin-bottom: 1; }
#feedback { height: auto; margin-top: 1; }
#ask-log { height: 1fr; border: round $primary 40%; padding: 0 1; }
#status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
"""


def split_paths(value: str) -> list[Path]:
    """Paths typed or pasted into one box: separated by ';' or new lines, quotes allowed."""
    parts = (p.strip().strip("\"'") for p in value.replace("\n", ";").split(";"))
    return [Path(p).expanduser() for p in parts if p]


def _positive_int(value: str, default: int) -> int:
    try:
        return max(1, int(value))
    except ValueError:
        return default


class MarginApp(App[None]):
    TITLE = "Margin"
    CSS = CSS
    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    def __init__(self, study: Study, model_label: str = ""):
        super().__init__()
        self.study = study
        self.model_label = model_label
        self._busy: str | None = None
        self._model_ready = False
        self._current: int | None = None
        self._notes_parts: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="library", id="tabs"):
            with TabPane("Library", id="library"):
                yield Label("Add books, lecture notes, slides or photos. Separate several paths with ;", classes="hint")
                with Horizontal(classes="row"):
                    yield Input(placeholder="biology-textbook.pdf; lecture-3.pptx; whiteboard.jpg", id="add-paths")
                    yield Button("Add", id="add", variant="primary")
                yield DataTable(id="documents", cursor_type="row", zebra_stripes=True)
                yield RichLog(id="library-log", wrap=True, max_lines=200)
            with TabPane("Blueprint", id="blueprint"):
                yield Label("Past papers show which chapters the exam weighs. PDFs, Word files, photos and text files all work.", classes="hint")
                with Horizontal(classes="row"):
                    yield Input(placeholder="paper-2023.pdf; paper-2024.jpg", id="papers")
                    yield Button("Read papers", id="read-papers", variant="primary")
                with Horizontal(id="blueprint-tables"):
                    yield DataTable(id="chapters", cursor_type="row")
                    yield DataTable(id="sections", cursor_type="row")
            with TabPane("Notes", id="notes"):
                with Horizontal(classes="row"):
                    yield Input(placeholder="chapter or section, e.g. ch4 or 4.2", id="notes-scope")
                    yield Checkbox("Diagrams", value=True, id="visuals")
                    yield Button("Write notes", id="write-notes", variant="primary")
                yield ProgressBar(total=1, show_eta=False, id="notes-progress")
                with VerticalScroll(id="notes-scroll"):
                    yield Markdown("Notes are checked sentence by sentence against the book and saved as each section finishes.", id="notes-view")
                with Horizontal(classes="row"):
                    yield Button("Export HTML", id="export-html")
                    yield Button("Export Markdown", id="export-markdown")
                    yield Button("Anki deck", id="export-anki")
            with TabPane("Practice", id="practice"):
                with Horizontal(classes="row"):
                    yield Input(placeholder="chapter or section, e.g. ch4 or 4.2", id="practice-scope")
                    yield Label("Questions", classes="field")
                    yield Input("5", type="integer", id="count")
                    yield Label("Marks each", classes="field")
                    yield Input("2", type="integer", id="marks")
                    yield Button("Write questions", id="generate", variant="primary")
                    yield Button("Retry mistakes", id="retry")
                with Horizontal(id="practice-body"):
                    yield OptionList(id="questions")
                    with Vertical(id="question-pane"):
                        yield Static("Questions appear here. Each one passed three checks before you see it.", id="question-text")
                        yield TextArea(id="answer", soft_wrap=True, placeholder="Write your answer, then press Mark.")
                        yield Button("Mark", id="mark", variant="primary")
                        yield Static(id="feedback")
            with TabPane("Ask", id="ask"):
                yield Label("Ask about your library, or ask for a table, flowchart, questions or an export.", classes="hint")
                yield RichLog(id="ask-log", wrap=True, max_lines=2000)
                yield Input(placeholder="Compare static and kinetic friction in a table", id="ask-input")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(MARGIN_THEME)
        self.theme = "margin"
        self.sub_title = self.model_label
        self._refresh_documents()
        self._set_status("Ready.")

    # jobs --------------------------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(Text(message))

    def _run(self, label: str, work: Callable[[], Any], done: Callable[[Any], None], uses_model: bool = False) -> bool:
        if self._busy:
            self.notify(f"Still {self._busy}. Wait for it to finish.", severity="warning")
            return False
        self._busy = label
        loading = " The model loads first, which takes a little while." if uses_model and not self._model_ready else ""
        self._set_status(f"{label.capitalize()}…{loading}")

        def job() -> None:
            try:
                result = work()
            except Exception as exc:  # the job boundary: every failure is shown to the student
                self.call_from_thread(self._job_failed, label, exc)
                return
            self.call_from_thread(self._job_done, done, result, uses_model)

        self.run_worker(job, name=label, group="jobs", thread=True, exit_on_error=False)
        return True

    def _job_done(self, done: Callable[[Any], None], result: Any, uses_model: bool) -> None:
        self._busy = None
        self._model_ready = self._model_ready or uses_model
        self._set_status("Ready.")
        done(result)

    def _job_failed(self, label: str, exc: Exception) -> None:
        self._busy = None
        self._set_status(f"{label.capitalize()} failed.")
        self.notify(f"{label.capitalize()} failed: {exc}", severity="error", timeout=10)

    # library ----------------------------------------------------------------------

    def _refresh_documents(self) -> None:
        table = self.query_one("#documents", DataTable)
        table.clear(columns=True)
        table.add_columns("Title", "Type", "Pages", "Read by OCR", "Added")
        for d in self.study.documents():
            table.add_row(Text(str(d["title"])), str(d["kind"]), str(d["pages"]), str(d["ocr_pages"]), str(d["added_at"])[:10])

    @on(Button.Pressed, "#add")
    @on(Input.Submitted, "#add-paths")
    def add_files(self) -> None:
        box = self.query_one("#add-paths", Input)
        paths = split_paths(box.value)
        if not paths:
            self.notify("Type the path of a file to add.")
            return
        log = self.query_one("#library-log", RichLog)

        def report(path: Path, message: str) -> None:
            self.call_from_thread(log.write, Text(f"{path.name}: {message}"))

        if self._run(f"adding {len(paths)} file(s)", lambda: self.study.add(paths, report), lambda _: self._refresh_documents()):
            box.clear()

    # blueprint --------------------------------------------------------------------

    @on(Button.Pressed, "#read-papers")
    @on(Input.Submitted, "#papers")
    def read_papers(self) -> None:
        paths = split_paths(self.query_one("#papers", Input).value)
        if not paths:
            self.notify("Type the path of at least one past paper.")
            return
        self._run("reading past papers", lambda: self.study.read_papers(paths), self._show_blueprint)

    def _show_blueprint(self, blueprint: Blueprint) -> None:
        chapters = self.query_one("#chapters", DataTable)
        chapters.clear(columns=True)
        chapters.add_columns("Chapter", "Exam weight")
        for chapter, share in chapter_shares(blueprint)[:TOP_ROWS]:
            chapters.add_row(chapter, f"{share:.0%}")
        sections = self.query_one("#sections", DataTable)
        sections.clear(columns=True)
        sections.add_columns("Section", "Title", "Weight", "Questions")
        for row in blueprint.top(TOP_ROWS):
            sections.add_row(row.sid, Text(row.title[:40]), f"{row.share:.0%}", Text(", ".join(row.questions[:3])))
        unmatched = f"; {blueprint.unmatched} matched nothing in your library" if blueprint.unmatched else ""
        self.notify(f"Read {blueprint.questions} questions{unmatched}. Trust the chapter ranking more than the section one.", timeout=8)

    # notes ------------------------------------------------------------------------

    @on(Button.Pressed, "#write-notes")
    @on(Input.Submitted, "#notes-scope")
    def write_notes(self) -> None:
        scope = self.query_one("#notes-scope", Input).value.strip()
        if not scope:
            self.notify("Type a chapter or section, for example ch4 or 4.2.")
            return
        targets = self.study.sections(scope)
        if not targets:
            self.notify(f"No sections match {scope!r}. Check the ids in the Library or with margin search.", severity="error")
            return
        visuals = self.query_one("#visuals", Checkbox).value

        def report(sid: str, notes: SectionNotes, resumed: bool) -> None:
            self.call_from_thread(self._section_written, notes, resumed)

        work = lambda: self.study.write_notes(scope, visuals, report)  # noqa: E731
        if self._run(f"writing notes for {scope}", work, self._notes_done, uses_model=True):
            self._notes_parts = []
            self.query_one("#notes-progress", ProgressBar).update(total=len(targets), progress=0)
            self.query_one("#notes-view", Markdown).update(f"Writing notes for {len(targets)} section(s). Finished sections are kept if you stop.")

    def _section_written(self, notes: SectionNotes, resumed: bool) -> None:
        self.query_one("#notes-progress", ProgressBar).advance(1)
        self._notes_parts.append(to_markdown([notes]))
        self.query_one("#notes-view", Markdown).update("\n\n---\n\n".join(self._notes_parts))
        state = "already written" if resumed else f"written in {notes.seconds:.0f} s, {notes.dropped} unsupported sentence(s) removed"
        self._set_status(f"{notes.sid} {notes.title}: {state}")

    def _notes_done(self, notes: list[SectionNotes]) -> None:
        self.notify(f"Notes saved for {len(notes)} section(s). Export HTML to see the diagrams drawn.")

    @on(Button.Pressed, "#export-html, #export-markdown, #export-anki")
    def export(self, event: Button.Pressed) -> None:
        fmt = (event.button.id or "").removeprefix("export-")
        scope = self.query_one("#notes-scope", Input).value.strip()
        if not scope:
            self.notify("Type the chapter or section to export.")
            return
        self._run(f"exporting {fmt}", lambda: self.study.export(scope, fmt), lambda done: self.notify(f"Saved {done.path}", timeout=10))

    # practice ---------------------------------------------------------------------

    @on(Button.Pressed, "#generate")
    @on(Input.Submitted, "#practice-scope")
    def generate(self) -> None:
        scope = self.query_one("#practice-scope", Input).value.strip()
        if not scope:
            self.notify("Type a chapter or section, for example ch4 or 4.2.")
            return
        count = _positive_int(self.query_one("#count", Input).value, 5)
        marks = _positive_int(self.query_one("#marks", Input).value, 2)
        self._run("writing questions", lambda: self.study.generate_questions(scope, count, marks), self._show_questions, uses_model=True)

    def _fill_questions(self, questions: list[Question]) -> None:
        options = self.query_one("#questions", OptionList)
        options.clear_options()
        options.add_options([Text(f"Q{i + 1}  {q.question}") for i, q in enumerate(questions)])
        if questions:
            options.highlighted = 0
            self._choose(0)
        else:
            self._current = None

    def _show_questions(self, result: tuple[list[Question], int]) -> None:
        questions, rejected = result
        self._fill_questions(questions)
        left_out = f"; {rejected} failed the checks and were left out" if rejected else ""
        self.notify(f"{len(questions)} question(s) ready{left_out}.")

    @on(Button.Pressed, "#retry")
    def retry(self) -> None:
        self._run("finding mistakes to retry", lambda: (self.study.load_retries(), self.study.weak_topics(3)), self._show_retries)

    def _show_retries(self, result: tuple[list[Question], list[Any]]) -> None:
        questions, weak = result
        self._fill_questions(questions)
        weakest = "; weakest: " + ", ".join(f"{t.sid} ({t.score:.0%} of marks)" for t in weak) if weak else ""
        found = f"{len(questions)} question(s) you lost marks on are due again" if questions else "No mistakes are due for retry"
        self.notify(f"{found}{weakest}.", timeout=8)

    @on(OptionList.OptionHighlighted, "#questions")
    def question_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._choose(event.option_index)

    def _choose(self, index: int) -> None:
        if index >= len(self.study.questions):
            return
        self._current = index
        q = self.study.questions[index]
        text = Text.assemble((f"Q{index + 1} · {q.marks} marks · section {q.section}\n\n", "bold"), q.question)
        last = self.study.missed_before.get(q.question)
        if last is not None:
            cause = f", marked as {last.cause}" if last.cause else ""
            text.append(f"\n\nmissed before: {last.marks_awarded}/{last.max_marks}, {len(last.missing)} point(s) missing{cause}", style="italic")
        self.query_one("#question-text", Static).update(text)
        self.query_one("#answer", TextArea).load_text("")
        self.query_one("#feedback", Static).update("")

    @on(Button.Pressed, "#mark")
    def mark(self) -> None:
        if self._current is None:
            self.notify("Write some questions first.")
            return
        index, answer = self._current, self.query_one("#answer", TextArea).text
        self._run("marking your answer", lambda: self.study.mark(index, answer), self._show_grade, uses_model=True)

    def _show_grade(self, grade: Grade) -> None:
        text = Text.assemble((f"{grade.marks_awarded}/{grade.max_marks} marks", f"bold {GOLD}"), f"\n{grade.feedback}")
        for point in grade.missing:
            text.append(f"\n• missing: {point}")
        self.query_one("#feedback", Static).update(text)

    # ask ---------------------------------------------------------------------------

    @on(Input.Submitted, "#ask-input")
    def ask(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text and self._run("thinking", lambda: self.study.ask(text), self._show_answer, uses_model=True):
            event.input.clear()
            self.query_one("#ask-log", RichLog).write(Text(f"You: {text}", style=f"bold {GOLD}"))

    def _show_answer(self, result: LoopResult) -> None:
        log = self.query_one("#ask-log", RichLog)
        for r in result.results:
            detail = "" if r.ok else f" — {r.content[:160]}"
            log.write(Text(f"  {r.name} {'done' if r.ok else 'failed'}{detail}", style="dim"))
            if r.ok and r.artifact is not None:
                log.write(Text(r.artifact.text) if r.artifact.format == "mermaid" else RichMarkdown(r.artifact.text))
        log.write(RichMarkdown(result.final) if result.final else Text("I stopped after the step limit.", style="italic"))
