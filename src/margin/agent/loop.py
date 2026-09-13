"""The bounded tool loop: model turn, tool calls, results, repeat, stop.

Two fixes from the model benchmark live here (DESIGN.md §6):

* tool turns get a 2,048-token output budget — Qwen3.5 9B's 3,652-character
  table argument was cut off at 1,024 tokens;
* when llama-server rejects a tool call as unparseable JSON, the model is told
  why and gets one retry instead of the whole task failing.

The loop's shape is code, not the model's choice (D6): at most ``max_steps``
turns, then it stops and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from margin.agent.executor import ToolExecutor, ToolResult
from margin.runtime.client import ChatResult, ClientError

TOOL_TURN_TOKENS = 2048
MAX_STEPS = 6
PARSE_FAILURE = "Failed to parse tool call"
RETRY_MESSAGE = "Your last tool call could not be read as JSON, probably because it was cut off. Call the tool again with shorter arguments."


class ClientLike(Protocol):
    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, max_tokens: int = 1024, **kwargs: Any) -> ChatResult: ...


@dataclass(frozen=True)
class LoopResult:
    final: str
    results: tuple[ToolResult, ...]
    steps: int
    retries: int
    stopped: str  # "answered" or "max_steps"


def run_agent(client: ClientLike, executor: ToolExecutor, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
              max_steps: int = MAX_STEPS, max_tokens: int = TOOL_TURN_TOKENS) -> LoopResult:
    history = list(messages)
    results: list[ToolResult] = []
    retries = 0
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
            return LoopResult(reply.content, tuple(results), step + 1, retries, "answered")
        raw_calls = reply.message.get("tool_calls") or []
        turn = [reply.message]
        for i, call in enumerate(reply.tool_calls):
            outcome = executor.run(call)
            results.append(outcome)
            call_id = (raw_calls[i].get("id") if i < len(raw_calls) else None) or f"call_{step}_{i}"
            turn.append({"role": "tool", "tool_call_id": call_id, "content": outcome.content})
        history = [*history, *turn]
    return LoopResult("", tuple(results), max_steps, retries, "max_steps")
