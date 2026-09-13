"""Write notes section by section, verify them, and checkpoint every section.

For each section in scope: the section text goes to the model with the visual
tools available; the loop runs until the model answers with its notes; the
notes are verified against the section; the result is saved at once. A run that
is killed resumes at the first unfinished section, so a long chapter is never
started over.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from margin.agent.executor import ToolExecutor
from margin.agent.loop import ClientLike, fill_arguments
from margin.agent.tools import openai_tools
from margin.notes.verify import verify_notes
from margin.retrieve.search import in_scope
from margin.store.db import ChunkRow, Workspace
from margin.visuals.render import Rendered

MAX_SECTION_CHARS = 6000
VISUAL_TOOLS = ["make_flowchart", "make_table", "make_mindmap", "make_timeline", "make_formula_sheet", "make_glossary"]
# Two passes per section, not one prompt with optional tools. Asked to "write notes and call a
# visual tool if it helps", Gemma 4 E4B made no diagram for 6 of 6 sections, yet it called
# make_table natively when that was the only job (evals/notes_eval.py). The loop's shape is fixed in code.
VISUAL_PROMPT = (
    "You choose one visual aid for revising a textbook section. For a process or sequence of stages call make_flowchart; "
    "for a comparison or classification call make_table; for equations call make_formula_sheet; for dated events call make_timeline; "
    "for a topic with several subtopics call make_mindmap. Use only content from the section. Call exactly one tool. "
    "If no visual would help, reply with the single word none."
)
NOTES_PROMPT = (
    "You write concise, exam-focused revision notes from one textbook section. Use only facts stated in the section. "
    "Write Markdown: a heading with the section title, then key points as bullets, key terms in bold, and equations in LaTeX between $ signs."
)
VISUAL_MAX_STEPS = 2
NOTES_MAX_TOKENS = 2048


@dataclass(frozen=True)
class SectionNotes:
    doc_id: str
    sid: str
    title: str
    markdown: str
    visuals: tuple[Rendered, ...]
    kept: int
    dropped: int
    seconds: float


def sections_in_scope(ws: Workspace, scope: str | None = None, sids: list[str] | None = None) -> list[tuple[str, str, str]]:
    rows = ws.teaching_sections()
    if sids is not None:
        wanted = set(sids)
        return [r for r in rows if r[1] in wanted]
    return [r for r in rows if in_scope(ChunkRow(0, r[0], "", r[1], r[2], "", None, None), scope)]


def _visual_for(client: ClientLike, executor: ToolExecutor, section: dict[str, str]) -> tuple[Rendered, ...]:
    """Pick a visual with a tool call, then fill its arguments under that tool's JSON schema.

    Gemma 4 E4B chose sensible visuals but filled nested schemas from memory: tables
    without a title, rows as strings, mind maps with invented field names — every call
    failed validation, and a retry repeated the mistake. Generating the arguments
    with the schema enforced by llama.cpp's grammar makes them valid by construction.
    """
    pick = client.chat([{"role": "system", "content": VISUAL_PROMPT}, section], tools=openai_tools(VISUAL_TOOLS), max_tokens=NOTES_MAX_TOKENS)
    name = next((c.name for c in pick.tool_calls if c.name in VISUAL_TOOLS), None)
    if name is None:
        return ()
    fill_request = {"role": "user", "content": f"Fill in the {name.removeprefix('make_').replace('_', ' ')} using only content from the section."}
    filled = fill_arguments(client, [{"role": "system", "content": VISUAL_PROMPT}, section, fill_request], name, NOTES_MAX_TOKENS)
    if filled is None:
        return ()
    outcome = executor.run(filled)
    return (outcome.artifact,) if outcome.ok and outcome.artifact is not None else ()


def write_section(client: ClientLike, executor: ToolExecutor, ws: Workspace, doc_id: str, sid: str, title: str, visuals_wanted: bool = True) -> SectionNotes:
    """visuals_wanted=False skips the two visual calls: about 2.3x faster, which matters on a CPU."""
    text = (ws.section_text(doc_id, sid, include_exercises=False) or "")[:MAX_SECTION_CHARS]
    section = {"role": "user", "content": f"Section {sid} {title}\n\n{text}"}
    started = time.perf_counter()
    visuals = _visual_for(client, executor, section) if visuals_wanted else ()
    notes_reply = client.chat([{"role": "system", "content": NOTES_PROMPT}, section], max_tokens=NOTES_MAX_TOKENS)
    verified = verify_notes(notes_reply.content, text)
    return SectionNotes(doc_id, sid, title, verified.markdown, visuals, verified.kept, verified.dropped, round(time.perf_counter() - started, 2))


def load_saved(ws: Workspace, doc_id: str, sid: str, title: str) -> SectionNotes | None:
    saved = ws.load_notes(doc_id, sid)
    if saved is None:
        return None
    visuals = tuple(Rendered(**v) for v in json.loads(saved["visuals_json"]))
    return SectionNotes(doc_id, sid, title, saved["markdown"], visuals, saved["kept"], saved["dropped"], saved["seconds"])


def write_notes(client: ClientLike, ws: Workspace, scope: str | None = None, sids: list[str] | None = None,
                executor: ToolExecutor | None = None, on_progress: Callable[[str, SectionNotes, bool], None] | None = None,
                visuals: bool = True) -> list[SectionNotes]:
    executor = executor or ToolExecutor(ws)
    written: list[SectionNotes] = []
    for doc_id, sid, title in sections_in_scope(ws, scope, sids):
        notes = load_saved(ws, doc_id, sid, title)
        resumed = notes is not None
        if notes is None:
            notes = write_section(client, executor, ws, doc_id, sid, title, visuals_wanted=visuals)
            visuals = json.dumps([{"kind": v.kind, "format": v.format, "text": v.text} for v in notes.visuals])
            ws.save_notes(doc_id, sid, notes.markdown, visuals, notes.kept, notes.dropped, notes.seconds)
        written.append(notes)
        if on_progress:
            on_progress(sid, notes, resumed)
    return written


def to_markdown(notes: list[SectionNotes]) -> str:
    parts = []
    for n in notes:
        blocks = [n.markdown]
        blocks += [f"```mermaid\n{v.text}\n```" if v.format == "mermaid" else v.text for v in n.visuals]
        parts.append("\n\n".join(b for b in blocks if b.strip()))
    return "\n\n---\n\n".join(parts) + "\n"
