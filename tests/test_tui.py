import asyncio
import json
from pathlib import Path

from textual.widgets import Checkbox, DataTable, Input, OptionList, ProgressBar, RichLog, Static, TabbedContent, TextArea

from margin.tui.app import MarginApp, split_paths
from tests.test_agent import reply
from tests.test_study import QUESTIONS, _study

NOTES = "# 4.1 Kirchhoff's laws\n- **Voltage law**: the voltage around a closed loop sums to zero."


async def _settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _open(app, pilot, tab):
    # click the tab in the tab bar, as a student would
    target = next(t for t in app.screen.query("ContentTab") if t.id == f"--content-tab-{tab}")
    assert await pilot.click(target)
    await pilot.pause()
    assert app.query_one("#tabs", TabbedContent).active == tab


def _text(widget) -> str:
    return str(widget.render())


def test_split_paths_accepts_quotes_semicolons_and_new_lines():
    assert split_paths('"C:/books/a b.pdf"; notes.docx\n\n slides.pptx ') == [Path("C:/books/a b.pdf"), Path("notes.docx"), Path("slides.pptx")]


def test_whole_flow_runs_from_the_interface(tmp_path):
    script = [
        reply(NOTES),
        reply(json.dumps(QUESTIONS)),
        reply(json.dumps({"points_met": [True, False], "feedback": "Say what the voltages add up to."})),
        reply(calls=(("calculate", {"expression": "2 * 3"}),)),
        reply("It is **6**."),
    ]
    study, client = _study(tmp_path, script, mermaid_js=lambda: None)
    app = MarginApp(study, model_label="scripted model")

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            assert app.query_one("#documents", DataTable).row_count == 1

            await _open(app, pilot, "notes")
            app.query_one("#notes-scope", Input).value = "4.1"
            app.query_one("#visuals", Checkbox).value = False
            await pilot.click("#write-notes")
            await _settle(app, pilot)
            bar = app.query_one("#notes-progress", ProgressBar)
            assert (bar.total, bar.progress) == (1, 1)
            await pilot.click("#export-html")
            await _settle(app, pilot)
            assert "Voltage law" in (tmp_path / "out" / "margin-4-1.html").read_text(encoding="utf-8")

            await _open(app, pilot, "practice")
            app.query_one("#practice-scope", Input).value = "4.1"
            app.query_one("#count", Input).value = "1"
            await pilot.click("#generate")
            await _settle(app, pilot)
            assert app.query_one("#questions", OptionList).option_count == 1
            assert "Kirchhoff's voltage law" in _text(app.query_one("#question-text", Static))
            app.query_one("#answer", TextArea).load_text("It is about a closed loop.")
            await pilot.click("#mark")
            await _settle(app, pilot)
            assert "1/2 marks" in _text(app.query_one("#feedback", Static))

            await _open(app, pilot, "ask")
            box = app.query_one("#ask-input", Input)
            box.value = "What is 2 times 3?"
            box.focus()
            await pilot.press("enter")
            await _settle(app, pilot)
            log = "\n".join(line.text for line in app.query_one("#ask-log", RichLog).lines)
            assert "calculate done" in log and "6" in log

    asyncio.run(scenario())
    assert client.script == []  # every scripted reply was used, in order


def test_the_library_rereads_books_added_before_their_equations_could_be_read(tmp_path, monkeypatch):
    import margin.ingest.formulas as formulas
    from tests.test_formulas import _FakeReader, _pdf_with_drawn_equation

    study, _ = _study(tmp_path)
    pdf = tmp_path / "book.pdf"
    _pdf_with_drawn_equation(pdf)
    monkeypatch.setattr(formulas, "installed_reader", lambda models_dir: None)
    study.add([pdf])
    monkeypatch.setattr(formulas, "installed_reader", lambda models_dir: _FakeReader())
    app = MarginApp(study)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.click("#reread")
            await _settle(app, pilot)
            log = "\n".join(line.text for line in app.query_one("#library-log", RichLog).lines)
            assert "re-read with its equations" in log and "not found" in log

    asyncio.run(scenario())


def test_mistakes_come_back_through_the_retry_button(tmp_path):
    from datetime import datetime, timedelta, timezone

    now = [datetime.now(timezone.utc)]
    script = [reply(json.dumps(QUESTIONS)), reply(json.dumps({"points_met": [False, False], "feedback": "Name the loop."})),
              reply(json.dumps({"points_met": [True, True], "feedback": "Complete."}))]
    study, client = _study(tmp_path, script, clock=lambda: now[0])
    app = MarginApp(study)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await _open(app, pilot, "practice")
            await pilot.click("#retry")
            await _settle(app, pilot)
            assert app.query_one("#questions", OptionList).option_count == 0  # nothing answered yet

            app.query_one("#practice-scope", Input).value = "4.1"
            app.query_one("#count", Input).value = "1"
            await pilot.click("#generate")
            await _settle(app, pilot)
            app.query_one("#answer", TextArea).load_text("No idea.")
            await pilot.click("#mark")
            await _settle(app, pilot)

            now[0] += timedelta(days=2)
            await pilot.click("#retry")
            await _settle(app, pilot)
            assert app.query_one("#questions", OptionList).option_count == 1
            assert "missed before" in _text(app.query_one("#question-text", Static))
            app.query_one("#answer", TextArea).load_text("Around a closed loop the voltages sum to zero.")
            await pilot.click("#mark")
            await _settle(app, pilot)
            assert "2/2 marks" in _text(app.query_one("#feedback", Static))

    asyncio.run(scenario())
    assert client.script == []


def test_a_failed_job_is_shown_and_frees_the_app(tmp_path):
    study, _ = _study(tmp_path, script=[])  # the model has nothing to say: the first call raises
    app = MarginApp(study)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await _open(app, pilot, "notes")
            app.query_one("#notes-scope", Input).value = "4.1"
            await pilot.click("#write-notes")
            await _settle(app, pilot)
            assert "failed" in _text(app.query_one("#status", Static)) and app._busy is None

    asyncio.run(scenario())
