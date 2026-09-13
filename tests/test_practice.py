import json
import zipfile

import pytest

from margin.practice.flashcards import cards_from_notes, export_anki, new_schedule, review
from margin.practice.grading import grade_answer, marks_for
from margin.practice.questions import Question, check, generate_questions
from tests.test_agent import ScriptedClient, reply

SECTION = "Static friction acts between surfaces that are not sliding and can grow up to a maximum value. Kinetic friction acts once surfaces slide and is usually smaller than the maximum static friction."


def _q(answer, question="How does kinetic friction compare with static friction?", marks=2, kind="short"):
    return Question("6.2", kind, question, answer, marks, ("kinetic friction acts when sliding", "it is smaller than maximum static friction"))


def test_check_accepts_a_supported_answer_and_rejects_invented_or_copied_ones():
    good = check(_q("Kinetic friction acts once surfaces slide and is usually smaller than the maximum static friction."), SECTION, (), 2)
    assert good.accepted and good.reasons == ()
    invented = check(_q("Friction arises from quantum electromagnetic interactions between atomic lattices."), SECTION, (), 2)
    assert not invented.accepted and "not supported" in invented.reasons[0]
    book = ("How does kinetic friction compare with static friction for sliding surfaces?",)
    copied = check(_q(SECTION, question="How does kinetic friction compare with static friction for sliding surfaces?"), SECTION, book, 2)
    assert "near-copy" in " ".join(copied.reasons)
    assert "3 marks" in " ".join(check(_q(SECTION, marks=3), SECTION, (), 2).reasons)


def test_generate_questions_uses_the_schema_and_checks_each_question():
    payload = {"questions": [
        {"question": "When does kinetic friction act?", "answer": "Kinetic friction acts once surfaces slide.", "marks": 2, "marking_points": ["acts", "when sliding"]},
        {"question": "Why do tyres grip?", "answer": "Rubber molecules form transient covalent bonds with asphalt.", "marks": 2, "marking_points": ["bonds"]},
    ]}
    client = ScriptedClient([reply(json.dumps(payload))])
    checked = generate_questions(client, SECTION, "6.2", count=2, marks=2)
    assert [c.accepted for c in checked] == [True, False]
    assert client.calls[0][1] == 2048
    assert generate_questions(ScriptedClient([reply("not json")]), SECTION, "6.2", 1, 2) == []


def test_marks_are_computed_from_points_not_chosen_by_the_model():
    assert marks_for((True, False, True, True), 4) == 3 and marks_for((), 4) == 0
    client = ScriptedClient([reply(json.dumps({"points_met": [True, False], "feedback": "Say it is smaller."}))])
    grade = grade_answer(client, "Compare kinetic and static friction.", ("acts when sliding", "smaller than static"), "It acts when sliding.", 2)
    assert (grade.marks_awarded, grade.points_met, grade.missing) == (1, (True, False), ("smaller than static",))
    short_list = ScriptedClient([reply(json.dumps({"points_met": [True], "feedback": ""}))])
    assert grade_answer(short_list, "q", ("a", "b"), "answer", 2).points_met == (True, False)  # padded, never over-credited


def test_blank_answer_scores_zero_without_asking_the_model():
    client = ScriptedClient([])
    assert grade_answer(client, "q", ("a",), "   ", 3).marks_awarded == 0 and client.calls == []


def test_flashcards_come_from_bold_terms_in_notes():
    notes = "# 6.2 Friction\n- **Static friction**: acts between surfaces that are not sliding.\n- **Kinetic friction** — acts once surfaces slide.\n- **Static friction**: duplicate line.\n- Plain bullet without a term."
    cards = cards_from_notes(notes, "6.2")
    assert [(c.front, c.back) for c in cards] == [("Static friction", "acts between surfaces that are not sliding."), ("Kinetic friction", "acts once surfaces slide.")]


def test_fsrs_review_moves_the_due_date_forward():
    first, due_good = review(new_schedule(), "good")
    _, due_easy = review(first, "easy")
    assert due_easy > due_good
    with pytest.raises(KeyError):
        review(new_schedule(), "perfect")


def test_executor_binds_questions_grading_cards_and_export(tmp_path):
    from margin.agent.executor import ToolExecutor
    from margin.runtime.client import ToolCall
    from tests.test_notes import _workspace

    with _workspace(tmp_path) as ws:
        _run_practice_tools(ws, "4.1", tmp_path, ToolExecutor, ToolCall)


def _run_practice_tools(ws, sid, tmp_path, ToolExecutor, ToolCall):
    text = ws.section_text(ws.find_section(sid)[0], sid, include_exercises=False)
    answer = " ".join(text.split()[:25])
    payload = {"questions": [{"question": "Explain the idea of this section.", "answer": answer, "marks": 2, "marking_points": ["first idea", "second idea"]}]}
    client = ScriptedClient([reply(json.dumps(payload)), reply(json.dumps({"points_met": [True, True], "feedback": "Complete."}))])
    ex = ToolExecutor(ws, client=client, export_dir=tmp_path / "out")

    made = ex.run(ToolCall("generate_questions", {"topic": sid, "count": 1, "marks": 2, "question_type": "short"}, "{}"))
    assert made.ok and "Explain the idea of this section." in made.content and answer not in made.content  # answers stay hidden
    graded = ex.run(ToolCall("grade_answer", {"question": "Explain the idea of this section.", "student_answer": "both ideas", "max_marks": 2}, "{}"))
    assert graded.ok and graded.content.startswith("2/2 marks")
    unknown = ex.run(ToolCall("grade_answer", {"question": "Something else?", "student_answer": "x", "max_marks": 2}, "{}"))
    assert not unknown.ok and "generate_questions" in unknown.content

    assert "Write notes first" in ex.run(ToolCall("create_flashcards", {"topic": sid, "count": 5}, "{}")).content
    doc_id = ws.find_section(sid)[0]
    ws.save_notes(doc_id, sid, "- **Term one**: the first definition here.", "[]", 1, 0, 1.0)
    assert ex.run(ToolCall("create_flashcards", {"topic": sid, "count": 5}, "{}")).content == "Term one — the first definition here."
    md = ex.run(ToolCall("export_notes", {"format": "markdown", "topic": sid}, "{}"))
    anki = ex.run(ToolCall("export_notes", {"format": "anki", "topic": sid}, "{}"))
    assert md.ok and anki.ok and "1 cards" in anki.content
    assert "Term one" in next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    html_export = ex.run(ToolCall("export_notes", {"format": "html", "topic": sid}, "{}"))
    assert html_export.ok and "Term one" in next((tmp_path / "out").glob("*.html")).read_text(encoding="utf-8")
    assert "print" in ex.run(ToolCall("export_notes", {"format": "pdf"}, "{}")).content


def test_anki_export_writes_a_valid_package(tmp_path):
    cards = cards_from_notes("- **Torque**: the rotational equivalent of force.", "10.6")
    path = export_anki(cards, "Physics::Chapter 10", tmp_path / "deck.apkg")
    with zipfile.ZipFile(path) as zf:
        assert "collection.anki2" in zf.namelist()
