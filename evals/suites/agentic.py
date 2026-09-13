"""Agentic suite: does the model pick the right tool, with the right arguments, at the right time?

Two kinds of case:

* **single-turn** (simple, multiple, parallel, irrelevance, multi_turn): one model
  call; the emitted tool calls are compared with the expected ones.
* **task**: a bounded agent loop. Tool results come from fixtures in the case, so
  the score measures planning and argument passing, not retrieval quality.

Calls recovered from reply text rather than parsed by llama-server count the
same as native ones, and their share is reported as ``agent.text_parsed_call_rate``.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from evals.matchers import match_args, match_calls, squash
from margin.agent.tools import BY_NAME, openai_tools, validate_arguments
from margin.runtime.client import ClientError, LlamaClient, ToolCall

SYSTEM_PROMPT = (
    "You are Margin, an offline study assistant. The student's books, notes and past papers are already uploaded. "
    "Call a tool whenever one fits the request. If a request needs several independent things, call several tools at once. "
    "If no tool fits, reply in plain text without calling any tool."
)
SINGLE_TURN = ("simple", "multiple", "parallel", "multi_turn")


def _call_counts(calls: tuple[ToolCall, ...] | list[ToolCall]) -> dict[str, int]:
    return {
        "calls": len(calls),
        "valid_calls": sum(1 for c in calls if validate_arguments(c.name, c.arguments) is None),
        "hallucinated_calls": sum(1 for c in calls if c.name not in BY_NAME),
        "text_calls": sum(1 for c in calls if c.source == "text"),
    }


def _single_turn(client: LlamaClient, case: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *case["messages"]]
    res = client.chat(messages, tools=tools, max_tokens=768)
    actual = [(c.name, c.arguments) for c in res.tool_calls]
    expected = case["expected"]
    if expected:
        selection_ok, args_ok = match_calls(expected, actual)
    else:
        selection_ok = args_ok = not actual
    return {
        "id": case["id"], "category": case["category"], "selection_ok": selection_ok, "args_ok": args_ok,
        **_call_counts(res.tool_calls), "latency_s": res.wall_s,
        "emitted": [{"name": n, "args": a} for n, a in actual], "content": res.content[:300],
    }


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(name in it for name in needle)


def _task(client: LlamaClient, case: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *case["messages"]]
    made: list[ToolCall] = []
    final: str | None = None
    steps = latency = 0.0
    for step in range(case.get("max_steps", 5)):
        res = client.chat(messages, tools=tools, max_tokens=1024)
        steps, latency = step + 1, latency + res.wall_s
        if not res.tool_calls:
            final = res.content
            break
        messages.append(res.message)
        raw_calls = res.message.get("tool_calls") or []
        for i, call in enumerate(res.tool_calls):
            made.append(call)
            call_id = raw_calls[i].get("id") if i < len(raw_calls) else None
            messages.append({"role": "tool", "tool_call_id": call_id or f"call_{step}_{i}", "content": case["mocks"].get(call.name, "Done.")})

    names = [c.name for c in made]
    trajectory_ok = _is_subsequence(case["trajectory"], names)
    args_ok = all(any(c.name == name and match_args(exp, c.arguments) for c in made) for name, exp in case.get("final_args", {}).items())
    wanted = case.get("answer_contains")
    answer_ok = True if not wanted else final is not None and any(squash(w) in squash(final) for w in wanted)
    return {
        "id": case["id"], "category": "task", "trajectory_ok": trajectory_ok, "args_ok": args_ok, "answer_ok": answer_ok,
        "terminated": final is not None, "success": trajectory_ok and args_ok and answer_ok and final is not None,
        "steps": steps, **_call_counts(made), "redundant_calls": max(0, len(made) - len(case["trajectory"])), "latency_s": latency,
        "emitted": [{"name": c.name, "args": c.arguments} for c in made], "final": (final or "")[:300],
    }


def _error_row(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {"id": case["id"], "category": case["category"], "error": str(exc)[:300], "selection_ok": False, "args_ok": False,
            "success": False, "trajectory_ok": False, "calls": 0, "valid_calls": 0, "hallucinated_calls": 0, "text_calls": 0, "latency_s": 0.0}


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return mean(1.0 if r.get(key) else 0.0 for r in rows) if rows else None


def _share(rows: list[dict[str, Any]], key: str) -> float | None:
    total = sum(r["calls"] for r in rows)
    return sum(r.get(key, 0) for r in rows) / total if total else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    single = [r for c in SINGLE_TURN for r in by_cat[c]]
    tasks = by_cat["task"]
    ok_single = [r for r in single if "error" not in r]
    ok_tasks = [r for r in tasks if "error" not in r]
    return {
        "agent.selection_acc": _rate(single, "selection_ok"),
        "agent.args_acc": _rate(single, "args_ok"),
        "agent.parallel_acc": _rate(by_cat["parallel"], "args_ok"),
        "agent.multi_turn_acc": _rate(by_cat["multi_turn"], "args_ok"),
        "agent.irrelevance_acc": _rate(by_cat["irrelevance"], "args_ok"),
        "agent.schema_valid_rate": _share(rows, "valid_calls"),
        "agent.hallucinated_tool_rate": _share(rows, "hallucinated_calls"),
        "agent.text_parsed_call_rate": _share(rows, "text_calls"),
        "agent.task_success_rate": _rate(tasks, "success"),
        "agent.trajectory_acc": _rate(tasks, "trajectory_ok"),
        "agent.redundant_calls_per_task": mean(r.get("redundant_calls", 0) for r in tasks) if tasks else None,
        "agent.steps_mean": mean(r.get("steps", 0) for r in ok_tasks) if ok_tasks else None,
        "agent.error_rate": _rate(rows, "error"),
        "agent.call_latency_mean_s": mean(r["latency_s"] for r in ok_single) if ok_single else None,
        "agent.n_cases": float(len(rows)),
        "agent.n_ok": float(sum(1 for r in rows if "error" not in r)),
    }


def run(client: LlamaClient, cases: list[dict[str, Any]]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    tools = openai_tools()
    rows = []
    for case in cases:
        try:
            rows.append(_task(client, case, tools) if case["category"] == "task" else _single_turn(client, case, tools))
        except ClientError as exc:
            rows.append(_error_row(case, exc))
    return summarise(rows), rows
