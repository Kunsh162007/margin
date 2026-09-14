import json

import docx
import numpy as np
import pytest

from margin.study import SYSTEM_PROMPT, Study
from tests.test_agent import ScriptedClient, reply
from tests.test_notes import _workspace

QUESTIONS = {"questions": [{"question": "What does Kirchhoff's voltage law say?", "answer": "The voltage around a closed loop sums to zero.",
                            "marks": 2, "marking_points": ["about a closed loop", "the voltages sum to zero"]}]}


class FakeEmbedder:
    name = "fake"

    def passages(self, texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    def query(self, text):
        return np.zeros(4, dtype=np.float32)


def make_docx(path):
    document = docx.Document()
    document.add_heading("Ohm's law", level=1)
    document.add_paragraph("The current through a resistor is proportional to the voltage across it, and the constant is the resistance.")
    document.save(str(path))
    return path


def _study(tmp_path, script=(), **kwargs):
    _workspace(tmp_path).close()
    client = ScriptedClient(list(script))
    return Study(tmp_path / "ws.db", tmp_path / "out", lambda: client, FakeEmbedder(), **kwargs), client


def test_add_indexes_files_and_reports_the_ones_it_cannot_read(tmp_path):
    study, _ = _study(tmp_path)
    bogus = tmp_path / "notes.bin"
    bogus.write_bytes(b"\x00\x01\x02 not a document")
    messages = []
    added = study.add([make_docx(tmp_path / "ohm.docx"), bogus, tmp_path / "missing.pdf"], lambda p, m: messages.append((p.name, m)))
    assert len(added) == 1 and added[0].chunks > 0
    assert [name for name, _ in messages] == ["ohm.docx", "notes.bin", "missing.pdf"]
    assert messages[1][1].startswith("skipped") and messages[2][1].startswith("skipped")
    assert len(study.documents()) == 2
    assert study.add([tmp_path / "ohm.docx"])[0].skipped


def test_blueprint_reads_a_text_paper_and_refuses_an_empty_one(tmp_path):
    study, _ = _study(tmp_path)
    paper = tmp_path / "2024.txt"
    paper.write_text("EXAMINATION 2024\nQ1. State Kirchhoff's voltage law for a closed loop. [4 marks]\nQ2. What poles do magnets have? [2 marks]\n", encoding="utf-8")
    blueprint = study.read_papers([paper])
    assert blueprint.questions == 2 and study.blueprint is blueprint
    with pytest.raises(ValueError):
        study.read_papers([tmp_path / "missing.txt"])


def test_the_blueprint_is_saved_with_the_library(tmp_path):
    study, _ = _study(tmp_path)
    paper = tmp_path / "2024.txt"
    paper.write_text("EXAMINATION 2024\nQ1. State Kirchhoff's voltage law for a closed loop. [4 marks]\nQ2. What poles do magnets have? [2 marks]\n", encoding="utf-8")
    saved = study.read_papers([paper])
    next_session = Study(tmp_path / "ws.db", tmp_path / "out", lambda: None, FakeEmbedder())
    assert next_session.blueprint == saved and saved.sections


def test_marked_answers_are_remembered_and_mistakes_come_back_for_retry(tmp_path):
    from datetime import datetime, timedelta, timezone

    now = [datetime.now(timezone.utc)]
    script = [reply(json.dumps(QUESTIONS)), reply(json.dumps({"points_met": [True, False], "feedback": "Say they sum to zero."})),
              reply(json.dumps({"points_met": [True, True], "feedback": "Complete."}))]
    study, _ = _study(tmp_path, script, clock=lambda: now[0])
    study.generate_questions("4.1", count=1, marks=2)
    assert study.mark(0, "It is about a closed loop.").marks_awarded == 1
    assert study.load_retries() == [] and [t.sid for t in study.weak_topics()] == ["4.1"]  # not due in the same sitting

    now[0] += timedelta(days=2)
    [mistake] = study.mistakes()
    assert study.set_cause(mistake.id, "keyword") and study.mistakes()[0].cause == "keyword"
    assert [q.question for q in study.load_retries()] == [QUESTIONS["questions"][0]["question"]] == [q.question for q in study.questions]
    assert study.mark(0, "Around a closed loop the voltages sum to zero.").marks_awarded == 2
    now[0] += timedelta(days=400)
    assert study.load_retries() == [] and study.weak_topics() == []


def test_questions_marking_and_ask_keep_their_state(tmp_path):
    script = [reply(json.dumps(QUESTIONS)), reply(json.dumps({"points_met": [True, True], "feedback": "Complete."})), reply("It is in section 4.1.")]
    study, client = _study(tmp_path, script)
    questions, rejected = study.generate_questions("4.1", count=1, marks=2)
    assert len(questions) == 1 and rejected == 0
    assert study.mark(0, "Around a closed loop the voltages sum to zero.").marks_awarded == 2
    result = study.ask("Where is the voltage law?")
    assert result.final == "It is in section 4.1." and [m["role"] for m in study.history] == ["user", "assistant"]
    assert client.calls[-1][0][0] == {"role": "system", "content": SYSTEM_PROMPT}
    with pytest.raises(ValueError):
        study.export("5.1", "html")  # no saved notes yet
