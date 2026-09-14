import numpy as np
import pytest

from margin.ingest.types import Block, Document, Section
from margin.retrieve.chunking import MAX_CHARS, chunk_section, is_exercise_section
from margin.retrieve.search import Searcher, in_scope, rrf
from margin.store.db import ChunkRow, Workspace, fts_query

VOCAB = ["kirchhoff", "voltage", "loop", "current", "junction", "photosynthesis", "chloroplast", "light", "question", "what"]


class FakeEmbedder:
    name = "fake-bow"

    def _vec(self, text: str) -> np.ndarray:
        v = np.array([text.lower().count(w) for w in VOCAB], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def passages(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def _section(sid, title, text, page=1, level=2):
    return Section(sid, title, level, ("Chapter",), page, page, text)


def _workspace(tmp_path):
    doc = Document("doc1", "book.pdf", "pdf", "Physics", 3, (Block("x", 1),))
    sections = [
        _section("4.1", "Kirchhoff's laws", "Kirchhoff's voltage law says the voltage around a loop sums to zero."),
        _section("4.2", "Junctions", "Kirchhoff's current law: current into a junction equals current out."),
        _section("5.1", "Photosynthesis", "Photosynthesis in the chloroplast turns light into sugar."),
        _section("5.9", "Review Questions", "1 . What does Kirchhoff's voltage law say about a loop voltage question?"),
    ]
    from margin.retrieve.chunking import is_exercise_section as flag

    chunks = [c for s in sections for c in chunk_section(s)]
    embedder = FakeEmbedder()
    ws = Workspace.open(tmp_path / "ws.db")
    ws.add_document(doc, sections, chunks, embedder.passages([c.text for c in chunks]), embedder.name, {s.id for s in sections if flag(s.title)})
    return ws, embedder


def test_chunks_respect_limits_and_keep_every_word():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(12))
    chunks = chunk_section(_section("1.1", "Long", text))
    assert len(chunks) > 1 and all(len(c.text) <= MAX_CHARS * 2 for c in chunks)
    assert sum(c.text.count("word") for c in chunks) == 12 * 120
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_a_single_huge_paragraph_is_split():
    chunks = chunk_section(_section("1.1", "Wall of text", "Sentence number one is here. " * 400))
    assert all(len(c.text) <= 2 * MAX_CHARS for c in chunks) and len(chunks) > 3


@pytest.mark.parametrize(("title", "expected"), [("Review Questions", True), ("Conceptual Questions", True), ("Challenge Problems", True), ("Problems", True), ("Key Terms", False), ("4.2 Kirchhoff's laws", False)])
def test_exercise_sections_are_recognised(title, expected):
    assert is_exercise_section(title) is expected


def test_question_lists_inside_a_review_section_are_flagged_but_summary_is_not():
    text = (
        "Key Terms\n\nvector: a quantity with magnitude and direction.\n\n"
        "Summary\n\n2.1 Scalars and Vectors • A vector has magnitude and direction. Conceptual Questions\n\n"
        "2.1 Scalars and Vectors\n\n1 . Is a temperature forecast a vector or a scalar quantity? Explain.\n\n"
        "Problems\n\n2.1 Scalars and Vectors\n\n12 . A scuba diver makes a slow descent into the ocean."
    )
    chunks = chunk_section(_section("s9", "Chapter Review", text))
    teaching = " ".join(c.text for c in chunks if not c.is_exercise)
    questions = " ".join(c.text for c in chunks if c.is_exercise)
    assert "magnitude and direction" in teaching and "temperature forecast" not in teaching
    assert "temperature forecast" in questions and "scuba diver" in questions


def test_prose_mentioning_problems_does_not_start_an_exercise_tail():
    from margin.retrieve.chunking import split_exercise_tail

    text = "This chapter teaches strategies for solving Problems in physics. Problems are solved in three stages."
    assert split_exercise_tail(text) == (text, "")


def test_leak_guard_flags_one_title_dominating_first_place():
    from collections import Counter

    from evals.retrieval_eval import leak_check

    leaked = leak_check(Counter({"Chapter Review": 291, "1.2 Units and Standards": 4}), 295)
    assert leaked["leak_warning"] and leaked["top1_title"] == "Chapter Review"
    spread = leak_check(Counter({f"{i}.1 Section": 10 for i in range(1, 18)}), 170)
    assert not spread["leak_warning"] and spread["top1_share"] < 0.2
    assert leak_check(Counter(), 0)["leak_warning"] is False


def test_fts_query_survives_punctuation():
    assert fts_query("Kirchhoff's law?") == '"Kirchhoff" OR "s" OR "law"'
    assert fts_query("??") is None


def test_rrf_rewards_agreement():
    assert [i for i, _ in rrf([1, 2, 3], [3, 1, 4])][:2] == [1, 3]


def test_scope_matching():
    row = ChunkRow(1, "doc1", "Physics", "4.2", "Junctions", "t", 1, 1)
    assert in_scope(row, "ch4") and in_scope(row, "chapter 4") and in_scope(row, "4.2") and in_scope(row, "doc1")
    assert not in_scope(row, "4.1") and not in_scope(row, "ch42")


def test_hybrid_search_finds_the_right_section_and_skips_exercises(tmp_path):
    ws, embedder = _workspace(tmp_path)
    with ws:
        hits = Searcher(ws, embedder).search("what does the voltage law say about a loop", top_k=3)
        assert hits[0].row.sid == "4.1"
        assert all(h.row.sid != "5.9" for h in hits)
        scoped = Searcher(ws, embedder).search("light", top_k=3, scope="ch5")
        assert {h.row.sid for h in scoped} == {"5.1"}
        assert ws.section_text("doc1", "4.2").startswith("Kirchhoff's current law")


def test_removing_a_document_clears_its_index(tmp_path):
    ws, embedder = _workspace(tmp_path)
    with ws:
        assert ws.remove_document("doc1")
        assert ws.fts("kirchhoff", 10) == [] and ws.chunk_count(include_exercises=True) == 0
        assert Searcher(ws, embedder).search("kirchhoff") == []


def test_an_old_workspace_gains_the_formulas_column_and_keeps_its_documents(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, pages INTEGER NOT NULL, "
        "ocr_pages INTEGER NOT NULL, added_at TEXT NOT NULL); INSERT INTO documents VALUES ('d1', 'book.pdf', 'pdf', 'Book', 1, 0, '2026-09-01T00:00:00');"
    )
    con.commit()
    con.close()
    with Workspace.open(path) as ws:
        assert [(d["id"], d["formulas_read"]) for d in ws.documents()] == [("d1", 0)]


def _library_text(ws):
    return "\n".join(ws.section_text(doc_id, sid) or "" for doc_id, sid, _ in ws.teaching_sections())


def test_adding_a_pdf_again_rereads_it_with_formulas_and_keeps_its_notes(tmp_path):
    from margin.retrieve.search import index_document
    from tests.test_formulas import _FakeReader, _pdf_with_drawn_equation
    from tests.test_study import FakeEmbedder

    pdf = tmp_path / "book.pdf"
    _pdf_with_drawn_equation(pdf)
    with Workspace.open(tmp_path / "ws.db") as ws:
        first = index_document(ws, pdf, FakeEmbedder())
        ws.save_notes(first.doc_id, "s1", "# notes written earlier", "[]", 1, 0, 1.0)
        assert "$$" not in _library_text(ws)

        again = index_document(ws, pdf, FakeEmbedder(), formulas=_FakeReader())
        assert not again.skipped and again.reread and "$$g=9.8$$" in _library_text(ws)
        assert ws.load_notes(first.doc_id, "s1")["markdown"] == "# notes written earlier"

        reader = _FakeReader()
        assert index_document(ws, pdf, FakeEmbedder(), formulas=reader).skipped and reader.crops == []  # read already
        assert index_document(ws, pdf, FakeEmbedder()).skipped


def test_cli_reread_upgrades_pdfs_added_without_formulas(tmp_path, monkeypatch, capsys):
    import margin.ingest.formulas as formulas
    import margin.retrieve.embed as embed
    from margin.cli import main
    from tests.test_formulas import _FakeReader, _pdf_with_drawn_equation
    from tests.test_study import FakeEmbedder

    monkeypatch.setenv("MARGIN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    pdf = tmp_path / "book.pdf"
    _pdf_with_drawn_equation(pdf)
    assert main(["add", str(pdf), "--no-formulas"]) == 0
    monkeypatch.setattr(formulas, "installed_reader", lambda models_dir: _FakeReader())
    capsys.readouterr()
    assert main(["reread"]) == 0 and "re-read 1" in capsys.readouterr().out
    assert main(["reread"]) == 0 and "Nothing to re-read" in capsys.readouterr().out


def test_cli_add_then_search_uses_the_default_workspace(tmp_path, monkeypatch, capsys):
    docx = pytest.importorskip("docx")
    import margin.retrieve.embed as embed
    from margin.cli import main

    monkeypatch.setenv("MARGIN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    d = docx.Document()
    d.add_heading("Kirchhoff's laws", level=1)
    d.add_paragraph("Kirchhoff's voltage law says the voltage around a loop sums to zero.")
    d.add_heading("Photosynthesis", level=1)
    d.add_paragraph("Photosynthesis in the chloroplast turns light into sugar.")
    path = tmp_path / "notes.docx"
    d.save(path)

    assert main(["search", "voltage"]) == 1  # empty workspace says how to add files
    assert main(["add", str(path), str(tmp_path / "missing.pdf")]) == 1  # one bad file reported, the good one added
    assert main(["add", str(path)]) == 0 and "already added" in capsys.readouterr().out
    assert main(["search", "voltage around a loop", "--top-k", "1"]) == 0
    assert "Kirchhoff" in capsys.readouterr().out


def test_gold_questions_are_split_on_their_numbers():
    from evals.datasets.build_retrieval_gold import extract, split_questions

    text = "1 . What is physics? Explain in your own words please. 2 . Some have described physics as a search for simplicity."
    assert [n for n, _ in split_questions(text)] == [1, 2]
    blocks = (
        Block("Conceptual Questions", 47),
        Block("1.2 Units and Standards", 47, "heading", 2),
        Block("7 . Identify some advantages of metric units in everyday life.", 47),
        Block("Additional Problems", 49),
        Block("99 . This problem is not under a section heading at all.", 49),
    )
    rows = extract(blocks, "uphys1")
    assert [(r["gold_section"], r["kind"]) for r in rows] == [("1.2", "conceptual")]


def test_answer_key_never_replaces_questions_and_first_copy_wins():
    from evals.datasets.build_retrieval_gold import before_answer_key, extract, first_occurrences
    from margin.ingest.types import TocEntry

    blocks = (
        Block("Conceptual Questions", 47),
        Block("1.1 The Scope and Scale of Physics", 47, "heading", 2),
        Block("3 . If two different theories describe observations equally well, is one more valid?", 47),
        Block("3 . If two different theories describe observations equally well, is one more valid?", 48),
        Block("Answer Key", 895, "heading", 1),
        Block("Conceptual Questions", 895),
        Block("1.1 The Scope and Scale of Physics", 895, "heading", 2),
        Block("3 . No, neither of these two theories is more valid than the other one here.", 895),
    )
    doc = Document("d", "p.pdf", "pdf", "Physics", 900, blocks, (TocEntry("Answer Key", 1, 895),))
    rows = first_occurrences(extract(before_answer_key(doc), "uphys1"))
    assert len(rows) == 1 and rows[0]["question"].startswith("If two different theories")
    assert before_answer_key(Document("d", "p.pdf", "pdf", "T", 1, blocks)) == blocks  # no bookmark: nothing cut


def test_gold_markers_glued_to_the_previous_block_are_recognised():
    from evals.datasets.build_retrieval_gold import extract

    blocks = (
        Block("Useful strategies for solving problems", 98),
        Block("1 . This line is prose in the chapter and must not become a question.", 98),
        Block("• The cross product of two vectors is a vector. Conceptual Questions", 99),
        Block("2.1 Scalars and Vectors", 99, "heading", 2),
        Block("1 . Is a temperature forecast a vector or a scalar quantity? 2 . Which of these is a vector quantity here? Problems", 99),
        Block("2.1 Scalars and Vectors", 100, "heading", 2),
        Block("12 . A scuba diver makes a slow descent into the depths of the ocean. Additional Problems", 100),
        Block("50 . Not under a section heading, so it must be ignored entirely.", 101),
    )
    rows = extract(blocks, "uphys1")
    assert [(r["id"], r["kind"], r["gold_section"]) for r in rows] == [
        ("uphys1-ch2-q1", "conceptual", "2.1"),
        ("uphys1-ch2-q2", "conceptual", "2.1"),
        ("uphys1-ch2-q12", "problem", "2.1"),
    ]
    assert not rows[1]["question"].endswith("Problems")
