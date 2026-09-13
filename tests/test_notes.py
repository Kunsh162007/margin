import json

import numpy as np
import pytest

from margin.ingest.types import Block, Document, Section
from margin.notes.verify import verify_notes
from margin.notes.writer import sections_in_scope, to_markdown, write_notes
from margin.retrieve.chunking import chunk_section, is_exercise_section
from margin.store.db import Workspace
from tests.test_agent import ScriptedClient, reply

SOURCE = "Kirchhoff's voltage law says the voltage around a closed loop sums to zero. Kirchhoff's current law says current entering a junction equals current leaving it."


def test_verify_keeps_supported_sentences_headings_and_formulas():
    notes = "# 4.1 Kirchhoff's laws\n- **Voltage law**: voltage around a closed loop sums to zero.\n- Quantum chromodynamics explains gluon confinement inside hadrons.\n- $V = IR$\n\n```mermaid\nflowchart TD\n```"
    v = verify_notes(notes, SOURCE)
    assert "Voltage law" in v.markdown and "gluon" not in v.markdown and "$V = IR$" in v.markdown and "```mermaid" in v.markdown
    assert (v.kept, v.dropped) == (2, 1) and v.dropped_sentences[0].startswith("Quantum")


def _workspace(tmp_path):
    doc = Document("d1", "physics.pdf", "pdf", "Physics", 10, (Block("x", 1),))
    sections = [
        Section("4.1", "Kirchhoff's laws", 2, ("Chapter 4",), 1, 1, SOURCE),
        Section("4.2", "Resistors", 2, ("Chapter 4",), 2, 2, "Resistors in series add their resistances together along a single path."),
        Section("5.1", "Magnetism", 2, ("Chapter 5",), 3, 3, "Magnets have north and south poles that attract opposite poles."),
        Section("s9", "Review Questions", 2, ("Chapter 4",), 4, 4, "1 . What does the voltage law state about a closed loop?"),
    ]
    chunks = [c for s in sections for c in chunk_section(s)]
    ws = Workspace.open(tmp_path / "ws.db")
    ws.add_document(doc, sections, chunks, np.zeros((len(chunks), 4), dtype=np.float32), "fake", {s.id for s in sections if is_exercise_section(s.title)})
    return ws


def test_scope_selects_teaching_sections_only(tmp_path):
    with _workspace(tmp_path) as ws:
        assert [s for _, s, _ in sections_in_scope(ws, "ch4")] == ["4.1", "4.2"]
        assert [s for _, s, _ in sections_in_scope(ws, sids=["5.1", "s9"])] == ["5.1"]


def test_notes_are_written_verified_checkpointed_and_resumed(tmp_path):
    flow_call = ("make_flowchart", {"title": "KVL", "steps": [{"id": "1", "label": "Loop"}, {"id": "2", "label": "Sum is zero"}], "edges": [{"source": "Loop", "target": "Sum is zero"}]})
    client = ScriptedClient([
        reply(calls=(("make_flowchart", {"title": "unused"}),)),  # pick: only the tool's name is used
        reply(json.dumps(flow_call[1])),  # fill: arguments generated under the tool's JSON schema
        reply("# 4.1 Kirchhoff's laws\n- Voltage around a closed loop sums to zero.\n- Quantum chromodynamics explains gluon confinement in hadrons."),  # notes pass
    ])
    progress = []
    with _workspace(tmp_path) as ws:
        notes = write_notes(client, ws, sids=["4.1"], on_progress=lambda sid, n, resumed: progress.append((sid, resumed)))
        assert progress == [("4.1", False)] and notes[0].dropped == 1 and notes[0].kept == 1
        assert len(notes[0].visuals) == 1 and "gluon" not in notes[0].markdown

        never_called = ScriptedClient([])  # would raise IndexError if asked for anything
        again = write_notes(never_called, ws, sids=["4.1"], on_progress=lambda sid, n, resumed: progress.append((sid, resumed)))
        assert progress[-1] == ("4.1", True) and again[0].markdown == notes[0].markdown and again[0].visuals == notes[0].visuals

        exported = to_markdown(again)
        assert exported.startswith("# 4.1 Kirchhoff's laws") and "```mermaid\nflowchart TD" in exported


def test_clearing_notes_forces_a_rewrite(tmp_path):
    with _workspace(tmp_path) as ws:
        ws.save_notes("d1", "4.1", "# old", "[]", 1, 0, 1.0)
        assert ws.clear_notes(["4.1"]) == 1 and ws.load_notes("d1", "4.1") is None
