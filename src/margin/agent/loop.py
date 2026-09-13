"""The bounded tool loop: model turn, tool calls, results, repeat, stop.

Two fixes from the model benchmark (``evals/bench_models.py``) live here:

* tool turns get a 2,048-token output budget — Qwen3.5 9B's 3,652-character
  table argument was cut off at 1,024 tokens;
* when llama-server rejects a tool call as unparseable JSON, the model is told
  why and gets one retry instead of the whole task failing.

One fix from the notes writer is shared here: when a call names the
right tool but its arguments fail validation — Gemma 4 E4B sends table rows as
strings instead of lists — the tool choice is kept and the arguments are
generated once more with the tool's JSON schema enforced by llama.cpp's grammar.

The loop's shape is code, not the model's choice: at most ``max_steps``
turns, then it stops and says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from margin.agent.executor import ToolExecutor, ToolResult
from margin.agent.tools import BY_NAME, validate_arguments
from margin.runtime.client import ChatResult, ClientError, ToolCall

TOOL_TURN_TOKENS = 2048
MAX_STEPS = 6
PARSE_FAILURE = "Failed to parse tool call"
RETRY_MESSAGE = "Your last tool call could not be read as JSON, probably because it was cut off. Call the tool again with shorter arguments."
FILL_MESSAGE = "Your call to {tool} had invalid arguments ({problem}). Give the arguments for {tool} again, using only content from this conversation."


class ClientLike(Protocol):
    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, max_tokens: int = 1024, **kwargs: Any) -> ChatResult: ...


@dataclass(frozen=True)
class LoopResult:
    final: str
    results: tuple[ToolResult, ...]
    steps: int
    retries: int
    stopped: str  # "answered" or "max_steps"
    filled: int = 0  # calls whose arguments were regenerated under the tool's schema


def fill_arguments(client: ClientLike, messages: list[dict[str, Any]], name: str, max_tokens: int = TOOL_TURN_TOKENS) -> ToolCall | None:
    """Generate ``name``'s arguments with its JSON schema enforced, so their shape is valid by construction."""
    reply = client.chat(messages, json_schema=BY_NAME[name].params.model_json_schema(), max_tokens=max_tokens)
    try:
        arguments = json.loads(reply.content)
    except json.JSONDecodeError:
        return None
    return ToolCall(name, arguments if isinstance(arguments, dict) else None, reply.content, source="schema")


def _repair(client: ClientLike, history: list[dict[str, Any]], call: ToolCall, max_tokens: int) -> ToolCall | None:
    problem = validate_arguments(call.name, call.arguments)
    if problem is None or call.name not in BY_NAME:
        return None
    request = {"role": "user", "content": FILL_MESSAGE.format(tool=call.name, problem=problem)}
    try:
        filled = fill_arguments(client, [*history, request], call.name, max_tokens)
    except ClientError:
        return None  # the original call runs and the model sees the validation message instead
    if filled is None or validate_arguments(filled.name, filled.arguments) is not None:
        return None
    return filled


def run_agent(client: ClientLike, executor: ToolExecutor, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
              max_steps: int = MAX_STEPS, max_tokens: int = TOOL_TURN_TOKENS) -> LoopResult:
    history = list(messages)
    results: list[ToolResult] = []
    retries = 0
    fills = 0
    for step in range(max_steps):
        try:
            reply = client.chat(history, tools=tools, max_tokens=max_tokens)
        except ClientError as exc:
            if PARSE_FAILURE in str(exc) and retries == 0:
                retries += 1
                history = [*history, {"role": "user", "content": RETRY_MESSAGE}]
                continue
            raise
        if not reply.tool_calls:
            return LoopResult(reply.content, tuple(results), step + 1, retries, "answered", fills)
        raw_calls = reply.message.get("tool_calls") or []
        turn = [reply.message]
        for i, call in enumerate(reply.tool_calls):
            repaired = _repair(client, history, call, max_tokens)
            if repaired is not None:
                call, fills = repaired, fills + 1
            outcome = executor.run(call)
            results.append(outcome)
            call_id = (raw_calls[i].get("id") if i < len(raw_calls) else None) or f"call_{step}_{i}"
            turn.append({"role": "tool", "tool_call_id": call_id, "content": outcome.content})
        history = [*history, *turn]
    return LoopResult("", tuple(results), max_steps, retries, "max_steps", fills)
