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
