import json

import pytest

from margin.agent.calc import CalcError, calculate
from margin.agent.executor import ToolExecutor
from margin.agent.loop import RETRY_MESSAGE, TOOL_TURN_TOKENS, run_agent
from margin.runtime.client import ClientError, ToolCall, to_result


def reply(content: str = "", calls: tuple = ()):
    message: dict = {"content": content}
    if calls:
        message["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return to_result({"choices": [{"message": message, "finish_reason": "stop"}]}, 0.01)


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.kwargs = []

    def chat(self, messages, *, tools=None, max_tokens=1024, **kwargs):
        self.calls.append((list(messages), max_tokens))
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.parametrize(("expr", "value"), [("0.5*1200*25^2", 375000.0), ("sqrt(16) + 2**3", 12.0), ("-(3 % 2)", -1.0), ("round(pi, 2)", 3.14)])
def test_calculate(expr, value):
    assert calculate(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "(1).__class__", "2 ** 100000", "1/0", "x + 1", "9" * 300])
def test_calculate_refuses_code_and_runaway_input(expr):
    with pytest.raises(CalcError):
        calculate(expr)


def test_executor_reports_invalid_arguments_back_to_the_model():
    out = ToolExecutor().run(ToolCall("calculate", {"expr": "1+1"}, "{}"))
    assert not out.ok and "Invalid arguments for calculate" in out.content and "Fix them" in out.content


def test_executor_renders_visuals_and_calculates():
    flow = ToolExecutor().run(ToolCall("make_flowchart", {"title": "t", "steps": [{"id": "1", "label": "A"}, {"id": "2", "label": "B"}], "edges": [{"source": "A", "target": "B"}]}, "{}"))
    assert flow.ok and flow.artifact is not None and flow.artifact.text.startswith("flowchart TD")
    assert ToolExecutor().run(ToolCall("calculate", {"expression": "3.2 * 450"}, "{}")).content == "1440"
    assert "No library" in ToolExecutor().run(ToolCall("create_flashcards", {"topic": "x", "count": 3}, "{}")).content
    assert "No library" in ToolExecutor().run(ToolCall("read_section", {"section_id": "4.2"}, "{}")).content
    assert "margin blueprint" in ToolExecutor().run(ToolCall("get_exam_blueprint", {"top_n": 3}, "{}")).content


def test_loop_runs_tools_then_returns_the_answer_with_the_bigger_budget():
    client = ScriptedClient([reply(calls=(("calculate", {"expression": "2+2"}),)), reply("The answer is 4.")])
    result = run_agent(client, ToolExecutor(), [{"role": "user", "content": "2+2?"}], tools=[])
    assert (result.final, result.stopped, result.steps, result.retries) == ("The answer is 4.", "answered", 2, 0)
    assert result.results[0].content == "4" and client.calls[0][1] == TOOL_TURN_TOKENS
    assert client.calls[1][0][-1] == {"role": "tool", "tool_call_id": "c0", "content": "4"}


def test_loop_retries_once_after_a_truncated_tool_call():
    parse_error = ClientError('llama-server returned 500: {"message":"Failed to parse tool call arguments as JSON"}')
    client = ScriptedClient([parse_error, reply("Done.")])
    result = run_agent(client, ToolExecutor(), [{"role": "user", "content": "table please"}], tools=[])
    assert result.final == "Done." and result.retries == 1
    assert client.calls[1][0][-1] == {"role": "user", "content": RETRY_MESSAGE}
    with pytest.raises(ClientError):
        run_agent(ScriptedClient([parse_error, parse_error]), ToolExecutor(), [{"role": "user", "content": "x"}], tools=[])
    with pytest.raises(ClientError):
        run_agent(ScriptedClient([ClientError("connection refused")]), ToolExecutor(), [{"role": "user", "content": "x"}], tools=[])


TABLE_ROWS_AS_STRINGS = ("make_table", {"title": "Friction", "columns": ["Kind", "Acts when"], "rows": ["static | at rest", "kinetic | sliding"]})
TABLE_FILLED = {"title": "Friction", "columns": ["Kind", "Acts when"], "rows": [["static", "at rest"], ["kinetic", "sliding"]]}


def test_invalid_arguments_are_filled_again_under_the_tools_schema():
    client = ScriptedClient([reply(calls=(TABLE_ROWS_AS_STRINGS,)), reply(json.dumps(TABLE_FILLED)), reply("Here is the table.")])
    result = run_agent(client, ToolExecutor(), [{"role": "user", "content": "compare static and kinetic friction in a table"}], tools=[])
    assert result.results[0].ok and result.results[0].artifact.kind == "table" and result.filled == 1
    assert client.kwargs[1]["json_schema"]["title"] == "MakeTable"  # the fill ran with the tool's schema enforced
    assert "rows.0" in client.calls[1][0][-1]["content"] and result.final == "Here is the table."


def test_a_fill_that_cannot_be_read_leaves_the_error_for_the_model():
    client = ScriptedClient([reply(calls=(TABLE_ROWS_AS_STRINGS,)), reply("not json"), reply("Sorry.")])
    result = run_agent(client, ToolExecutor(), [{"role": "user", "content": "table"}], tools=[])
    assert not result.results[0].ok and "Invalid arguments" in result.results[0].content and result.filled == 0


def test_loop_stops_at_the_step_limit():
    looping = [reply(calls=(("calculate", {"expression": "1+1"}),)) for _ in range(3)]
    result = run_agent(ScriptedClient(looping), ToolExecutor(), [{"role": "user", "content": "x"}], tools=[], max_steps=3)
    assert result.stopped == "max_steps" and result.final == "" and len(result.results) == 3
