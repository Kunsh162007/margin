import random

import pytest

from margin.blueprint.paper import PaperQuestion, detect_year, parse_paper
from margin.blueprint.weights import RANK_SPLIT, build_blueprint, question_weight
from margin.retrieve.search import Hit
from margin.store.db import ChunkRow

PAPER = """UNIVERSITY EXAMINATION 2023
MECHANICS — PAPER 2
Answer ALL questions.

Q1. State Newton's second law of motion and give its units. [4 marks]

2) A 2 kg block slides down a frictionless incline of 30 degrees.
(a) Find the acceleration of the block. [3 marks]
(b) Find its speed after travelling 5 m. (4 marks)

Page 1 of 2
Question 3 Explain why the moment of inertia depends on the axis of rotation. (5)
END OF PAPER
"""


def test_parse_paper_reads_numbers_parts_marks_and_year():
    qs = parse_paper(PAPER, source="2023")
    assert [q.number for q in qs] == ["1", "2(a)", "2(b)", "3"]
    assert [q.marks for q in qs] == [4, 3, 4, 5]
    assert {q.year for q in qs} == {2023}
    assert qs[1].text.startswith("A 2 kg block slides") and qs[1].text.endswith("acceleration of the block.")
    assert "Page 1" not in qs[2].text and qs[0].key == "2023:1"


def test_numbers_inside_question_text_do_not_start_questions():
    text = "1. A car of mass\n1200 kg moves at\n3. m/s along a road; find its momentum. [2 marks]"
    qs = parse_paper(text)
    assert [q.number for q in qs] == ["1"]  # after question 1 only 1 or 2 can follow, so "3." stays text
    assert "3. m/s" in qs[0].text and qs[0].marks == 2
    no_separator = parse_paper("1. A block of\n2 kg rests on a table. Find the normal force. [2 marks]")
    assert [q.number for q in no_separator] == ["1"] and "2 kg rests" in no_separator[0].text


def test_detect_year_uses_the_header():
    assert detect_year("Physics Final Examination, May 2019\nQ1 ...") == 2019
    assert detect_year("no year here") is None


def test_question_weight_uses_marks_and_recency():
    old = PaperQuestion("1", "q", 10, 2016)
    new = PaperQuestion("1", "q", 10, 2026)
    assert question_weight(old, (2016, 2026)) == 10.0
    assert question_weight(new, (2016, 2026)) == 15.0
    assert question_weight(PaperQuestion("1", "q", None, None), None) == 1.0


def _row(sid, title):
    return ChunkRow(hash(sid) % 1000, "book", "Physics", sid, title, "text", 1, 1)


class FakeSearcher:
    """Routes a question to sections by keyword; the same section twice checks de-duplication."""

    ROUTES = {
        "newton": [("5.3", "Newton's second law"), ("5.3", "Newton's second law"), ("5.4", "Mass and weight"), ("6.1", "Friction")],
        "inertia": [("10.5", "Moments of inertia"), ("10.6", "Torque")],
    }

    def search(self, query, top_k=5, scope=None, rerank=False):
        for word, sections in self.ROUTES.items():
            if word in query.lower():
                return [Hit(_row(sid, title), 1.0) for sid, title in sections]
        return []


def test_blueprint_splits_weight_over_distinct_top_sections():
    questions = [
        PaperQuestion("1", "State Newton's second law", 10, 2020, "p1"),
        PaperQuestion("2", "Explain moment of inertia", 4, 2020, "p1"),
        PaperQuestion("3", "Something the book never covers", 5, 2020, "p1"),
    ]
    bp = build_blueprint(FakeSearcher(), questions, split=RANK_SPLIT)
    by_sid = {r.sid: r for r in bp.sections}
    assert bp.unmatched == 1 and bp.questions == 3
    assert by_sid["5.3"].weight == pytest.approx(6.0) and by_sid["5.4"].weight == pytest.approx(2.5) and by_sid["6.1"].weight == pytest.approx(1.5)
    assert by_sid["10.5"].weight == pytest.approx(4 * 0.6 / 0.85, abs=1e-3)  # weights are stored to 4 decimals
    assert sum(r.share for r in bp.sections) == pytest.approx(1.0, abs=1e-3)
    assert by_sid["5.3"].questions == ("p1:1",) and by_sid["5.3"].marks == 10 and bp.sections[0].sid == "5.3"


def test_best_match_only_gives_everything_to_the_first_section():
    bp = build_blueprint(FakeSearcher(), [PaperQuestion("1", "Newton", 10, None)], split=(1.0,))
    assert [(r.sid, r.weight) for r in bp.sections] == [("5.3", 10.0)]
    default = build_blueprint(FakeSearcher(), [PaperQuestion("1", "Newton", 10, None)])
    assert [(r.sid, r.weight) for r in default.sections] == [("5.3", 10.0)]  # best match is the default


def test_chapter_shares_roll_sections_up():
    from margin.blueprint.weights import chapter_shares

    bp = build_blueprint(FakeSearcher(), [PaperQuestion("1", "Newton", 6, None), PaperQuestion("2", "inertia", 4, None)], split=RANK_SPLIT)
    shares = dict(chapter_shares(bp))
    # Newton (6 marks): 85% to chapter 5 (5.3 + 5.4), 15% to chapter 6; inertia (4 marks): all to chapter 10
    assert shares["5"] == pytest.approx(0.51, abs=1e-3) and shares["6"] == pytest.approx(0.09, abs=1e-3) and shares["10"] == pytest.approx(0.4, abs=1e-3)
    assert [chapter for chapter, _ in chapter_shares(bp)] == ["5", "10", "6"]


class _BagEmbedder:
    name = "bag"
    VOCAB = ["newton", "force", "acceleration", "inertia", "rotation", "torque", "friction", "surface"]

    def _vec(self, text):
        import numpy as np

        v = np.array([text.lower().count(w) for w in self.VOCAB], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def passages(self, texts):
        import numpy as np

        return np.vstack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def test_cli_blueprint_ranks_chapters_from_a_paper(tmp_path, monkeypatch, capsys):
    docx = pytest.importorskip("docx")
    import margin.retrieve.embed as embed
    from margin.cli import main

    monkeypatch.setenv("MARGIN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(embed, "Embedder", _BagEmbedder)
    book = docx.Document()
    book.add_heading("5.3 Newton's second law", level=2)
    book.add_paragraph("Newton's second law: force equals mass times acceleration.")
    book.add_heading("10.6 Torque", level=2)
    book.add_paragraph("Torque causes rotation; inertia resists changes in rotation.")
    book_path = tmp_path / "physics.docx"
    book.save(book_path)
    paper = tmp_path / "paper2024.txt"
    paper.write_text("EXAMINATION 2024\n1. State Newton's law relating force and acceleration. [10 marks]\n2. Explain torque and rotation. [2 marks]\n", encoding="utf-8")

    assert main(["add", str(book_path)]) == 0
    capsys.readouterr()
    assert main(["blueprint", str(paper), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "2 questions, 2024" in out  # table titles wrap at pytest's capture width, so check rows, not titles
    rows = [line.split() for line in out.splitlines()]
    assert ["5", "83%"] in rows and ["10", "17%"] in rows  # chapter 5 (10 marks) outweighs chapter 10 (2 marks)
    assert rows.index(["5", "83%"]) < rows.index(["10", "17%"])


def test_synthetic_papers_round_trip_through_the_parser():
    from evals.blueprint_eval import make_papers, render_paper, score_parsing

    gold = [
        {"id": f"g{i}", "book": "b", "kind": "problem" if i % 2 else "conceptual", "chapter": 1 + i % 8, "gold_section": f"{1 + i % 8}.1",
         "question": f"Question number {i} asks how the quantity {i} changes when the force doubles and the mass halves."}
        for i in range(160)
    ]
    papers, favourites = make_papers(gold, random.Random(1))
    parsed = [parse_paper(render_paper(p)) for p in papers]
    scores = score_parsing(papers, parsed)
    assert len(favourites) == 6
    assert scores["question_recall"] == 1.0 and scores["marks_accuracy"] == 1.0 and scores["year_accuracy"] == 1.0
