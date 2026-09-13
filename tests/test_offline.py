"""D15: "offline" is something these tests can fail on.

The ``no_network`` fixture refuses every connection and DNS lookup that is not
to this machine. The first test proves the guard bites; the rest run the study
flow and the cached search models under it.
"""

import json

import httpx
import pytest

from tests.test_agent import reply
from tests.test_study import QUESTIONS, _study, make_docx

NOTES = "# 4.1 Kirchhoff's laws\n- **Voltage law**: the voltage around a closed loop sums to zero."


def test_the_guard_really_blocks_the_internet(no_network):
    with pytest.raises((no_network, httpx.HTTPError)):
        httpx.get("https://example.com", timeout=5)


def test_the_study_flow_needs_no_network(tmp_path, no_network):
    script = [
        reply(NOTES),
        reply(json.dumps(QUESTIONS)),
        reply(json.dumps({"points_met": [True, True], "feedback": "Complete."})),
        reply(calls=(("export_notes", {"format": "html", "topic": "4.1"}),)),
        reply("Your notes are exported."),
    ]
    study, client = _study(tmp_path, script, mermaid_js=lambda: "window.mermaid={initialize(){}};")
    assert study.add([make_docx(tmp_path / "ohm.docx")])[0].chunks > 0
    paper = tmp_path / "2024.txt"
    paper.write_text("EXAMINATION 2024\nQ1. State Kirchhoff's voltage law for a closed loop. [4 marks]\n", encoding="utf-8")
    assert study.read_papers([paper]).questions == 1
    assert study.write_notes("4.1", visuals=False)[0].kept == 1
    assert study.export("4.1", "html").path.exists() and study.export("4.1", "anki").cards == 1
    study.generate_questions("4.1", count=1, marks=2)
    assert study.mark(0, "The voltages around a closed loop sum to zero.").marks_awarded == 2
    assert study.ask("Export my notes for 4.1 as html").results[0].ok
    assert client.script == []


def test_cached_search_models_load_without_network(no_network):
    from margin.retrieve.embed import Embedder, cache_dir

    if not (cache_dir() / "models--qdrant--bge-small-en-v1.5-onnx-q").exists():
        pytest.skip("search models are not downloaded on this machine; run margin setup")
    assert Embedder().query("voltage around a closed loop").shape == (384,)
