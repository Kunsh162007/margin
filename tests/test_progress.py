from datetime import datetime, timedelta, timezone

import pytest

from margin.blueprint.weights import Blueprint, SectionWeight
from margin.practice.grading import Grade
from margin.practice.progress import CAUSES, due_retries, mastery, record_attempt, set_cause, weak_topics
from margin.practice.questions import Question
from margin.store.db import Workspace

NOW = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
POINTS = ("names the closed loop", "says the voltages sum to zero")


def _q(sid="4.1", text="State Kirchhoff's voltage law.", marks=2):
    return Question(sid, "short", text, "Around a closed loop the voltages sum to zero.", marks, POINTS)


def _grade(awarded, max_marks=2):
    return Grade(awarded, max_marks, (), POINTS[awarded:] if awarded < max_marks else (), "")


def test_an_attempt_is_stored_with_its_question_and_comes_back_when_due(tmp_path):
    with Workspace.open(tmp_path / "ws.db") as ws:
        attempt = record_attempt(ws, _q(), _grade(1), NOW)
        assert attempt.id > 0 and attempt.missing == POINTS[1:] and attempt.question == _q()
        assert attempt.due_at >= NOW + timedelta(days=1)  # never straight back in the same sitting
        assert due_retries(ws, NOW) == []
        assert [a.question.question for a in due_retries(ws, attempt.due_at)] == ["State Kirchhoff's voltage law."]


def test_full_marks_on_a_retry_clears_the_mistake(tmp_path):
    with Workspace.open(tmp_path / "ws.db") as ws:
        first = record_attempt(ws, _q(), _grade(0), NOW)
        record_attempt(ws, _q(), _grade(2), first.due_at)
        assert due_retries(ws, NOW + timedelta(days=400)) == []


def test_a_question_with_no_marks_comes_back_no_later_than_a_near_miss(tmp_path):
    with Workspace.open(tmp_path / "ws.db") as ws:
        missed = record_attempt(ws, _q(text="Define current."), _grade(0), NOW)
        near = record_attempt(ws, _q(text="Define voltage."), _grade(1), NOW)
        assert missed.due_at <= near.due_at


def test_a_cause_must_be_one_of_the_named_kinds(tmp_path):
    with Workspace.open(tmp_path / "ws.db") as ws:
        attempt = record_attempt(ws, _q(), _grade(1), NOW)
        assert "calculation" in CAUSES and set_cause(ws, attempt.id, "calculation")
        assert due_retries(ws, attempt.due_at)[0].cause == "calculation"
        with pytest.raises(ValueError, match="concept"):
            set_cause(ws, attempt.id, "bad luck")
        assert not set_cause(ws, 999, "concept")


def test_cli_lists_mistakes_due_and_records_a_cause(tmp_path, monkeypatch, capsys):
    from margin.cli import _workspace_path, main

    monkeypatch.setenv("MARGIN_HOME", str(tmp_path / "home"))
    with Workspace.open(_workspace_path()) as ws:
        attempt = record_attempt(ws, _q(), _grade(1), NOW)
    assert main(["mistakes"]) == 0
    out = capsys.readouterr().out
    assert "Kirchhoff" in out and "4.1" in out and "1/2" in out
    assert main(["mistakes", "--cause", str(attempt.id), "keyword"]) == 0
    capsys.readouterr()
    main(["mistakes"])
    assert "keyword" in capsys.readouterr().out
    assert main(["mistakes", "--cause", str(attempt.id), "bad luck"]) == 1
    assert "concept" in capsys.readouterr().out


def test_weak_topics_rank_sections_by_exam_weight_times_marks_lost(tmp_path):
    with Workspace.open(tmp_path / "ws.db") as ws:
        record_attempt(ws, _q("4.1", "q1"), _grade(2), NOW)
        record_attempt(ws, _q("4.2", "q2"), _grade(0), NOW)
        record_attempt(ws, _q("5.1", "q3"), _grade(1), NOW)
        assert {sid: m.score for sid, m in mastery(ws).items()} == {"4.1": 1.0, "4.2": 0.0, "5.1": 0.5}
        blueprint = Blueprint((SectionWeight("d1", "5.1", "Magnetism", 3.0, 0.6, (), 3), SectionWeight("d1", "4.2", "Resistors", 2.0, 0.4, (), 2)), 5, 0)
        assert [(t.sid, t.priority) for t in weak_topics(ws, blueprint)] == [("4.2", 0.4), ("5.1", 0.3)]
        assert [(t.sid, t.priority) for t in weak_topics(ws)] == [("4.2", 1.0), ("5.1", 0.5)]  # no papers read: marks lost alone
