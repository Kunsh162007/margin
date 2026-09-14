"""The `margin` command.

    margin            set up on first run, then start
    margin setup      download the runtime and the model for this machine
    margin doctor     show what Margin detected and what is installed
    margin models     list the models Margin can run
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from margin import __version__, config, hardware
from margin.runtime import models
from margin.runtime.binaries import LLAMA_CPP_BUILD, UnsupportedPlatform, ensure_llama_server, find_server
from margin.runtime.download import DownloadError, download

console = Console()
GOLD = "#d4a72c"


def _selection(args: argparse.Namespace) -> tuple[hardware.Hardware, str, models.ModelSpec]:
    hw = hardware.detect()
    backend = hardware.choose_backend(hw, getattr(args, "backend", None))
    model_id = getattr(args, "model", None)
    spec = models.get(model_id) if model_id else models.recommended(backend, hw.ram_gb, hw.gpu.vram_gb if hw.gpu else None)
    return hw, backend, spec


def cmd_doctor(args: argparse.Namespace) -> int:
    hw, backend, spec = _selection(args)
    paths = config.paths()
    server = find_server(paths.bin_dir / f"{LLAMA_CPP_BUILD}-{backend}") if (paths.bin_dir / f"{LLAMA_CPP_BUILD}-{backend}").exists() else None
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style=GOLD)
    table.add_column()
    table.add_row("version", __version__)
    table.add_row("system", f"{hw.os} {hw.arch}, {hw.physical_cores} cores, {hw.ram_gb} GB RAM")
    table.add_row("gpu", f"{hw.gpu.name} ({hw.gpu.vram_gb} GB)" if hw.gpu else "none found — running on CPU")
    table.add_row("backend", backend)
    table.add_row("home", str(paths.home))
    table.add_row("runtime", f"llama.cpp {LLAMA_CPP_BUILD} " + ("installed" if server else "not installed"))
    table.add_row("model", f"{spec.id} ({spec.size_gb} GB) " + ("installed" if spec.path(paths.models_dir).exists() else "not installed"))
    from margin.ingest.formulas import installed_reader

    table.add_row("formula reader", "installed" if installed_reader(paths.models_dir) else "not installed — run margin setup")
    console.print(table)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    models_dir = config.paths().models_dir
    table = Table(title="Models")
    for col in ("id", "family", "params", "download", "vision", "license", "installed"):
        table.add_column(col)
    for m in models.CANDIDATES:
        table.add_row(m.id, m.family, f"{m.params_b}B", f"{m.size_gb} GB", "yes" if m.vision else "", m.license, "yes" if m.path(models_dir).exists() else "")
    console.print(table)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    hw, backend, spec = _selection(args)
    paths = config.paths().ensure()
    size = "already downloaded" if spec.path(paths.models_dir).exists() else f"{spec.size_gb} GB download"
    console.print(f"[{GOLD}]Setting up Margin[/] for {backend.upper()} with {spec.id} ({size})")
    try:
        exe = ensure_llama_server(paths.bin_dir, hw.os, hw.arch, backend)
        download(spec.url, spec.path(paths.models_dir), label=spec.id)
    except (DownloadError, UnsupportedPlatform) as exc:
        console.print(f"[red]Setup failed:[/] {exc}")
        return 1
    if not _fetch_search_models():
        return 1
    try:
        from margin.export.assets import ensure_mermaid
        from margin.ingest.formulas import ensure_model

        ensure_mermaid()
        ensure_model(paths.models_dir)
    except DownloadError as exc:
        console.print(f"[red]Could not fetch the diagram renderer or the formula reader:[/] {exc}")
        return 1
    console.print(f"[{GOLD}]Ready.[/] runtime {exe.name}, model {spec.file}, search models cached")
    return 0


def _fetch_search_models() -> bool:
    """Download the embedding and reranking models now, so searching later needs no network."""
    from margin.retrieve.embed import Embedder, Reranker

    try:
        Embedder().passages(["warm-up"])
        Reranker().scores("warm-up", ["warm-up"])
    except Exception as exc:  # fastembed surfaces network and file errors as several types
        console.print(f"[red]Could not fetch search models:[/] {exc}")
        return False
    return True


def _workspace_path():
    return config.paths().ensure().workspaces_dir / "default.db"


def cmd_add(args: argparse.Namespace) -> int:
    """Index files into the default workspace."""
    from pathlib import Path

    from margin.ingest.detect import UnsupportedFile
    from margin.retrieve import embed
    from margin.ingest.formulas import installed_reader
    from margin.retrieve.search import index_document
    from margin.store.db import Workspace

    embedder, failures = embed.Embedder(), 0
    formulas = None if args.no_formulas else installed_reader(config.paths().models_dir)
    with Workspace.open(_workspace_path()) as ws:
        for name in args.files:
            try:
                result = index_document(ws, Path(name), embedder, formulas=formulas)
            except (FileNotFoundError, UnsupportedFile) as exc:
                console.print(f"[red]skipped[/] {name}: {exc}")
                failures += 1
                continue
            status = "already added" if result.skipped else f"{result.sections} sections, {result.chunks} chunks"
            console.print(f"[{GOLD}]{result.title}[/]  {status}")
    return 1 if failures else 0


def _server_config(args: argparse.Namespace):
    """(spec, backend, ServerConfig) for the installed model, or None after telling the user to run setup."""
    from margin.runtime.session import NotInstalled, server_config

    try:
        return server_config(getattr(args, "backend", None), getattr(args, "model", None))
    except NotInstalled as exc:
        console.print(str(exc))
        return None


def cmd_notes(args: argparse.Namespace) -> int:
    """Write verified notes for a chapter or section, resuming any unfinished run."""
    from pathlib import Path

    from margin.agent.executor import ToolExecutor
    from margin.notes.writer import sections_in_scope, to_markdown, write_notes
    from margin.retrieve import embed
    from margin.retrieve.search import Searcher
    from margin.runtime.client import ClientError, LlamaClient
    from margin.runtime.server import LlamaServer, ServerError
    from margin.store.db import Workspace

    selected = _server_config(args)
    if selected is None:
        return 1
    spec, backend, cfg = selected
    paths = config.paths()
    with Workspace.open(_workspace_path()) as ws:
        targets = sections_in_scope(ws, args.scope)
        if not targets:
            console.print(f"No sections match {args.scope!r}. Add books with margin add, and check ids with margin search.")
            return 1
        console.print(f"[{GOLD}]Writing notes[/] for {len(targets)} section(s) with {spec.id} on {backend.upper()} — finished sections are kept if you stop.")
        try:
            with LlamaServer(cfg, paths.logs_dir / "notes.log") as server:
                client = LlamaClient(server.base_url, tools_template_kwarg=spec.tools_template_kwarg)
                progress = lambda sid, n, resumed: console.print(f"  {sid} {n.title[:50]}: " + ("already written" if resumed else f"{n.seconds:.0f} s, {len(n.visuals)} diagram(s), {n.dropped} unsupported sentence(s) removed"))
                notes = write_notes(client, ws, scope=args.scope, executor=ToolExecutor(ws, Searcher(ws, embed.Embedder())), on_progress=progress, visuals=not args.no_visuals)
                client.close()
        except (ServerError, ClientError) as exc:
            console.print(f"[red]Stopped:[/] {exc}. Run the same command again to continue.")
            return 1
    out = Path(args.out)
    out.write_text(to_markdown(notes), encoding="utf-8")
    console.print(f"[{GOLD}]Saved[/] {out}")
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    """Write checked exam questions with marking schemes; --quiz asks them and marks your answers."""
    from pathlib import Path

    from margin.notes.writer import sections_in_scope
    from margin.practice.questions import generate_for_sections
    from margin.runtime.client import ClientError, LlamaClient
    from margin.runtime.server import LlamaServer, ServerError
    from margin.store.db import Workspace

    selected = _server_config(args)
    if selected is None:
        return 1
    spec, backend, cfg = selected
    with Workspace.open(_workspace_path()) as ws:
        targets = sections_in_scope(ws, args.scope)
        if not targets:
            console.print(f"No sections match {args.scope!r}. Add books with margin add, and check ids with margin search.")
            return 1
        texts = [(sid, (ws.section_text(doc_id, sid, include_exercises=False) or "")[:6000]) for doc_id, sid, _ in targets]
    lines: list[str] = []
    console.print(f"[{GOLD}]Writing questions[/] from {len(texts)} section(s) with {spec.id} on {backend.upper()}")
    try:
        with LlamaServer(cfg, config.paths().logs_dir / "questions.log") as server:
            client = LlamaClient(server.base_url)
            checked = generate_for_sections(client, texts, args.count, args.marks, args.type)
            accepted = [c.question for c in checked if c.accepted]
            rejected = len(checked) - len(accepted)
            if args.quiz:
                _quiz(client, accepted)
            client.close()
    except (ServerError, ClientError) as exc:
        console.print(f"[red]Stopped:[/] {exc}")
        return 1
    for n, q in enumerate(accepted, 1):
        points = "\n".join(f"   - {p}" for p in q.marking_points)
        lines.append(f"**Q{n}** ({q.marks} marks, section {q.section}) {q.question}\n\n<details><summary>Model answer and marking scheme</summary>\n\n{q.answer}\n\n{points}\n\n</details>")
    out = Path(args.out)
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[{GOLD}]Saved[/] {len(accepted)} question(s) to {out}" + (f"; {rejected} failed the checks and were left out" if rejected else ""))
    return 0 if accepted else 1


def _quiz(client, questions) -> None:
    """Ask each question, mark the typed answer, and remember it so lost marks come back for retry."""
    from rich.markup import escape

    from margin.practice.grading import grade_answer
    from margin.practice.progress import record_attempt
    from margin.store.db import Workspace

    total = scored = 0
    for n, q in enumerate(questions, 1):
        console.print(f"\n[{GOLD}]Q{n}[/] ({q.marks} marks) {escape(q.question)}")
        answer = console.input("your answer (blank to skip) > ")
        grade = grade_answer(client, q.question, q.marking_points, answer, q.marks)
        with Workspace.open(_workspace_path()) as ws:
            record_attempt(ws, q, grade)
        total, scored = total + q.marks, scored + grade.marks_awarded
        console.print(f"  {grade.marks_awarded}/{q.marks}. {grade.feedback}")
        for point in grade.missing:
            console.print(f"  missing: {point}")
    if questions:
        console.print(f"\n[{GOLD}]Score[/] {scored}/{total}")


def cmd_mistakes(args: argparse.Namespace) -> int:
    """Questions you lost marks on and when each is due again, your weakest sections, and why answers lost marks."""
    from datetime import datetime, timezone

    from rich.text import Text

    from margin.blueprint.weights import blueprint_from_json
    from margin.practice import progress
    from margin.store.db import Workspace

    with Workspace.open(_workspace_path()) as ws:
        if args.cause:
            return _note_cause(ws, *args.cause)
        now = datetime.now(timezone.utc)
        open_mistakes = sorted((a for a in progress.latest_attempts(ws) if a.lost_marks), key=lambda a: (a.due_at, a.id))
        saved = ws.load_blueprint_json()
        weak = progress.weak_topics(ws, blueprint_from_json(saved) if saved else None, limit=args.top)
    if args.retry:
        return _retry([a.question for a in open_mistakes if a.due_at <= now], args)
    if not open_mistakes:
        console.print("No mistakes recorded. Answer questions with: margin questions <scope> --quiz")
        return 0
    table = Table(title="Questions that lost marks", box=None)
    for col in ("id", "section", "marks", "due", "question", "cause"):
        table.add_column(col)
    for a in open_mistakes:
        due = "now" if a.due_at <= now else a.due_at.date().isoformat()
        table.add_row(str(a.id), a.question.section, f"{a.marks_awarded}/{a.max_marks}", due, Text(a.question.question[:70]), a.cause or "")
    console.print(table)
    if weak:
        console.print("Weakest sections: " + ", ".join(f"{t.sid} ({t.score:.0%} of marks)" for t in weak))
    console.print(f"Retry the ones due now with: margin mistakes --retry. Note why one lost marks with: margin mistakes --cause ID {'|'.join(progress.CAUSES)}")
    return 0


def _note_cause(ws, attempt_id: str, cause: str) -> int:
    from margin.practice import progress

    if not attempt_id.isdigit():
        console.print(f"[red]The attempt id must be a number, not {attempt_id!r}.[/]")
        return 1
    try:
        found = progress.set_cause(ws, int(attempt_id), cause)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    console.print(f"Noted: attempt {attempt_id} lost marks through {cause}." if found else f"There is no attempt {attempt_id}.")
    return 0 if found else 1


def _retry(questions, args: argparse.Namespace) -> int:
    from margin.runtime.client import ClientError, LlamaClient
    from margin.runtime.server import LlamaServer, ServerError

    if not questions:
        console.print("Nothing is due for another try yet.")
        return 0
    selected = _server_config(args)
    if selected is None:
        return 1
    _, _, cfg = selected
    try:
        with LlamaServer(cfg, config.paths().logs_dir / "questions.log") as server:
            client = LlamaClient(server.base_url)
            _quiz(client, questions)
            client.close()
    except (ServerError, ClientError) as exc:
        console.print(f"[red]Stopped:[/] {exc}")
        return 1
    return 0


def cmd_reread(args: argparse.Namespace) -> int:
    """Read again the PDFs in your library that were added before their drawn equations could be read."""
    from margin.ingest import formulas as formula_reading
    from margin.retrieve import embed
    from margin.retrieve.search import reread_library
    from margin.store.db import Workspace

    reader = formula_reading.installed_reader(config.paths().models_dir)
    if reader is None:
        console.print("The formula reader is not installed. Run: margin setup")
        return 1
    with Workspace.open(_workspace_path()) as ws:
        if not any(d["kind"] == "pdf" and not d["formulas_read"] for d in ws.documents()):
            console.print("Nothing to re-read: every PDF in your library already has its equations read.")
            return 0
        console.print("Re-reading the PDFs added before their equations could be read…")
        done, missing = reread_library(ws, embed.Embedder(), reader)
    console.print(f"[{GOLD}]Done:[/] re-read {len(done)} PDF(s); notes you already wrote are kept.")
    for path in missing:
        console.print(f"not found — add it again from where it is now: {path}")
    return 1 if missing else 0


def cmd_cards(args: argparse.Namespace) -> int:
    """Turn the key terms in saved notes into an Anki deck."""
    from pathlib import Path

    from margin.notes.writer import sections_in_scope
    from margin.practice.flashcards import cards_from_notes, export_anki
    from margin.store.db import Workspace

    cards = []
    with Workspace.open(_workspace_path()) as ws:
        for doc_id, sid, _ in sections_in_scope(ws, args.scope):
            saved = ws.load_notes(doc_id, sid)
            if saved:
                cards += cards_from_notes(saved["markdown"], sid)
    if not cards:
        console.print(f"No saved notes with key terms for {args.scope!r}. Write them first with: margin notes {args.scope}")
        return 1
    path = export_anki(cards, f"Margin::{args.scope}", Path(args.out))
    console.print(f"[{GOLD}]Saved[/] {len(cards)} card(s) to {path} — import it in Anki with File > Import.")
    return 0


def cmd_blueprint(args: argparse.Namespace) -> int:
    """Read past papers and show which chapters and sections of the library they weigh most."""
    from pathlib import Path

    from margin.blueprint.paper import paper_text, parse_paper
    from margin.blueprint.weights import blueprint_to_json, build_blueprint, chapter_shares
    from margin.ingest.detect import UnsupportedFile
    from margin.retrieve import embed
    from margin.retrieve.search import Searcher
    from margin.store.db import Workspace

    questions = []
    for name in args.files:
        path = Path(name)
        try:
            found = parse_paper(paper_text(path), source=path.stem)
        except (FileNotFoundError, UnsupportedFile) as exc:
            console.print(f"[red]skipped[/] {name}: {exc}")
            continue
        console.print(f"{path.name}: {len(found)} questions" + (f", {found[0].year}" if found and found[0].year else ""))
        questions.extend(found)
    if not questions:
        console.print("No questions found in those papers.")
        return 1
    with Workspace.open(_workspace_path()) as ws:
        if ws.chunk_count() == 0:
            console.print("Add your books first with: margin add <file> ...")
            return 1
        blueprint = build_blueprint(Searcher(ws, embed.Embedder()), questions)
        ws.save_blueprint_json(blueprint_to_json(blueprint))

    chapters = Table(title="Chapters by exam weight", box=None)
    chapters.add_column("chapter")
    chapters.add_column("share", justify="right")
    for chapter, share in chapter_shares(blueprint)[: args.top]:
        chapters.add_row(chapter, f"{share:.0%}")
    console.print(chapters)
    sections = Table(title="Sections (less certain than chapters)", box=None)
    for col in ("section", "title", "share", "questions"):
        sections.add_column(col)
    for row in blueprint.top(args.top):
        sections.add_row(row.sid, row.title[:45], f"{row.share:.0%}", ", ".join(row.questions[:4]))
    console.print(sections)
    if blueprint.unmatched:
        console.print(f"{blueprint.unmatched} question(s) matched nothing in your library.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the default workspace."""
    from margin.retrieve import embed
    from margin.retrieve.search import Searcher
    from margin.store.db import Workspace

    with Workspace.open(_workspace_path()) as ws:
        if ws.chunk_count() == 0:
            console.print("Nothing to search yet. Add files with: margin add <file> ...")
            return 1
        searcher = Searcher(ws, embed.Embedder(), embed.Reranker() if args.rerank else None)
        hits = searcher.search(args.query, top_k=args.top_k, scope=args.scope, rerank=args.rerank)
    if not hits:
        console.print("No matches.")
        return 0
    table = Table(box=None)
    for col in ("section", "title", "pages", "passage"):
        table.add_column(col)
    for hit in hits:
        r = hit.row
        pages = "" if r.page_start is None else (f"{r.page_start}" if r.page_start == r.page_end else f"{r.page_start}-{r.page_end}")
        table.add_row(r.sid, r.section_title[:40], pages, " ".join(r.text.split())[:160])
    console.print(table)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show what Margin reads from a file: type, pages, OCR pages and the section tree."""
    from pathlib import Path

    from margin.ingest.detect import UnsupportedFile
    from margin.ingest.pipeline import ingest_file

    try:
        doc, sections = ingest_file(Path(args.file))
    except (FileNotFoundError, UnsupportedFile) as exc:
        console.print(f"[red]Cannot read:[/] {exc}")
        return 1
    console.print(f"[{GOLD}]{doc.title}[/]  {doc.kind}, {doc.page_count} pages, {doc.char_count:,} characters, {len(sections)} sections")
    if doc.ocr_pages:
        console.print(f"text from OCR on {len(doc.ocr_pages)} page(s): {', '.join(map(str, doc.ocr_pages[:20]))}")
    table = Table(box=None)
    for col in ("id", "section", "pages", "chars"):
        table.add_column(col)
    for s in sections:
        pages = "" if s.page_start is None else (f"{s.page_start}" if s.page_start == s.page_end else f"{s.page_start}-{s.page_end}")
        table.add_row(s.id, "  " * (s.level - 1) + s.title[:70], pages, f"{len(s.text):,}")
    console.print(table)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export saved notes as Markdown, self-contained HTML or an Anki deck."""
    from pathlib import Path

    from margin.export.assets import mermaid_script
    from margin.export.files import export_notes, saved_notes
    from margin.notes.writer import sections_in_scope
    from margin.store.db import Workspace

    with Workspace.open(_workspace_path()) as ws:
        notes = saved_notes(ws, sections_in_scope(ws, args.scope))
    if not notes:
        console.print(f"No saved notes for {args.scope!r}. Write them first with: margin notes {args.scope}")
        return 1
    renderer = mermaid_script() if args.format == "html" else None
    if args.format == "html" and renderer is None:
        console.print("Diagrams will show as source text; run margin setup once to draw them.")
    done = export_notes(notes, args.format, Path(args.out_dir), args.scope, renderer)
    what = f"{done.cards} card(s)" if args.format == "anki" else f"{done.sections} section(s)"
    console.print(f"[{GOLD}]Saved[/] {what} to {done.path}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Set up on first run, then open the study interface."""
    from pathlib import Path

    from margin.runtime.session import ModelSession, NotInstalled, server_config
    from margin.study import Study
    from margin.tui.app import MarginApp

    try:
        spec, backend, cfg = server_config(getattr(args, "backend", None), getattr(args, "model", None))
    except NotInstalled:
        if cmd_setup(args) != 0:
            return 1
        spec, backend, cfg = server_config(getattr(args, "backend", None), getattr(args, "model", None))
    session = ModelSession(spec, backend, cfg, config.paths().logs_dir / "session.log")
    study = Study(_workspace_path(), Path.cwd() / "margin-exports", session.client)
    try:
        MarginApp(study, model_label=f"{spec.id} on {backend.upper()}").run()
    finally:
        session.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="margin", description="Offline study assistant.")
    parser.add_argument("--version", action="version", version=f"margin {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, fn, help_text in (
        ("setup", cmd_setup, "download the runtime and model"),
        ("doctor", cmd_doctor, "show detected hardware and install status"),
        ("models", cmd_models, "list available models"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)
        if name in ("setup", "doctor"):
            p.add_argument("--model", choices=list(models.BY_ID))
            p.add_argument("--backend", choices=hardware.BACKENDS)
    inspect = sub.add_parser("inspect", help="show how a file is read and split into sections")
    inspect.add_argument("file")
    inspect.set_defaults(func=cmd_inspect)
    add = sub.add_parser("add", help="add books, notes, slides or photos to your library")
    add.add_argument("files", nargs="+")
    add.add_argument("--no-formulas", action="store_true", help="do not read equations a PDF draws as pictures (faster on a CPU; formulas then cannot be checked against the book)")
    add.set_defaults(func=cmd_add)
    search = sub.add_parser("search", help="search your library")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--scope", help="chapter or section, e.g. ch4 or 4.2")
    search.add_argument("--rerank", action="store_true", help="add a cross-encoder second pass (slower; did not help on the textbook benchmark)")
    search.set_defaults(func=cmd_search)
    blueprint = sub.add_parser("blueprint", help="read past papers and rank chapters by exam weight")
    blueprint.add_argument("files", nargs="+", help="past papers: PDF, Word, photos or text")
    blueprint.add_argument("--top", type=int, default=10, help="how many chapters and sections to show")
    blueprint.set_defaults(func=cmd_blueprint)
    notes = sub.add_parser("notes", help="write verified notes for a chapter or section (resumes if stopped)")
    notes.add_argument("scope", help="chapter or section, e.g. ch4 or 4.2")
    notes.add_argument("--out", default="notes.md", help="Markdown file to write")
    notes.add_argument("--no-visuals", action="store_true", help="skip diagrams and tables (about 2.3x faster; useful on a CPU)")
    notes.add_argument("--model", choices=list(models.BY_ID))
    notes.add_argument("--backend", choices=hardware.BACKENDS)
    notes.set_defaults(func=cmd_notes)
    questions = sub.add_parser("questions", help="write exam questions that pass three checks, with marking schemes")
    questions.add_argument("scope", help="chapter or section, e.g. ch4 or 4.2")
    questions.add_argument("--count", type=int, default=5)
    questions.add_argument("--marks", type=int, default=2, help="marks per question")
    questions.add_argument("--type", choices=["short", "long", "numerical", "mcq"], default="short")
    questions.add_argument("--quiz", action="store_true", help="ask each question now and mark your typed answer")
    questions.add_argument("--out", default="questions.md", help="Markdown file to write")
    questions.add_argument("--model", choices=list(models.BY_ID))
    questions.add_argument("--backend", choices=hardware.BACKENDS)
    questions.set_defaults(func=cmd_questions)
    mistakes = sub.add_parser("mistakes", help="questions you lost marks on, when each is due again, and your weakest sections")
    mistakes.add_argument("--retry", action="store_true", help="ask the questions due now and mark your answers")
    mistakes.add_argument("--cause", nargs=2, metavar=("ID", "CAUSE"), help="note why an attempt lost marks: concept, misread, calculation, keyword or no-attempt")
    mistakes.add_argument("--top", type=int, default=5, help="how many weak sections to show")
    mistakes.add_argument("--model", choices=list(models.BY_ID))
    mistakes.add_argument("--backend", choices=hardware.BACKENDS)
    mistakes.set_defaults(func=cmd_mistakes)
    reread = sub.add_parser("reread", help="read again the PDFs added before their equations could be read (keeps your notes)")
    reread.set_defaults(func=cmd_reread)
    cards = sub.add_parser("cards", help="export the key terms from saved notes as an Anki deck")
    cards.add_argument("scope", help="chapter or section, e.g. ch4 or 4.2")
    cards.add_argument("--out", default="margin.apkg", help="Anki package to write")
    cards.set_defaults(func=cmd_cards)
    export = sub.add_parser("export", help="export saved notes as Markdown, self-contained HTML or an Anki deck")
    export.add_argument("scope", help="chapter or section, e.g. ch4 or 4.2")
    export.add_argument("--format", choices=["markdown", "html", "anki"], default="html")
    export.add_argument("--out-dir", default="margin-exports", help="folder to write into")
    export.set_defaults(func=cmd_export)
    parser.set_defaults(func=cmd_start)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\nstopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
