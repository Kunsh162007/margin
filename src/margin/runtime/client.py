"""Chat client for llama-server's OpenAI-compatible endpoint.

Requests are deterministic by default (temperature 0, fixed seed) so that an
evaluation run can be repeated. Thinking is off by default: for tool routing
and note drafting on a CPU, reasoning tokens cost seconds per call.

llama-server turns a model's tool-call text into structured calls only for chat
formats it recognises. When it returns none, the reply text is searched for the
common written formats, and any call recovered that way is marked
``source="text"`` so evaluations can report how often that happened.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_SEED = 42
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
TEXT_CALL_PATTERNS = (
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),  # Hermes style: SmolLM3 and others
    re.compile(r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>", re.DOTALL),  # Phi-4-mini
    re.compile(r"functools(\[.*\])", re.DOTALL),  # Phi-4-mini, older format
)


class ClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] | None  # None when the model emitted unparseable JSON
    raw_arguments: str
    id: str = ""
    source: str = "native"  # "native": parsed by llama-server; "text": recovered from reply text


@dataclass(frozen=True)
class ChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    prompt_ms: float
    predicted_ms: float
    wall_s: float
    message: dict[str, Any]  # assistant message to append to history

    @property
    def gen_tps(self) -> float:
        return self.completion_tokens / (self.predicted_ms / 1000) if self.predicted_ms else 0.0

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / (self.prompt_ms / 1000) if self.prompt_ms else 0.0


def _decode_arguments(value: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(value, dict):
        return value, json.dumps(value, ensure_ascii=False)
    if not isinstance(value, str) or not value:
        return ({}, "{}") if value in (None, "") else (None, str(value))
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None, value
    return (parsed if isinstance(parsed, dict) else None), value


def parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args, args_raw = _decode_arguments(fn.get("arguments", ""))
        calls.append(ToolCall(name=fn.get("name", ""), arguments=args, raw_arguments=args_raw, id=raw.get("id", "")))
    return tuple(calls)


def _calls_from_blob(blob: str, start_index: int) -> list[ToolCall]:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return [ToolCall(name="", arguments=None, raw_arguments=blob, id=f"text_{start_index}", source="text")]
    calls = []
    for item in data if isinstance(data, list) else [data]:
        if not isinstance(item, dict):
            continue
        args, args_raw = _decode_arguments(item.get("arguments", item.get("parameters", {})))
        calls.append(ToolCall(name=str(item.get("name", "")), arguments=args, raw_arguments=args_raw, id=f"text_{start_index + len(calls)}", source="text"))
    return calls


def parse_text_tool_calls(content: str) -> tuple[tuple[ToolCall, ...], str]:
    """Tool calls written into the reply text, and the text with those calls removed."""
    calls: list[ToolCall] = []
    remaining = content
    for pattern in TEXT_CALL_PATTERNS:
        for match in pattern.finditer(remaining):
            calls.extend(_calls_from_blob(match.group(1), len(calls)))
        remaining = pattern.sub("", remaining)
    return tuple(calls), remaining.strip()


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def build_payload(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    thinking: bool = False,
    cache_prompt: bool = True,
    tools_template_kwarg: str | None = None,
) -> dict[str, Any]:
    template_kwargs: dict[str, Any] = {"enable_thinking": thinking}
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "seed": DEFAULT_SEED,
        "max_tokens": max_tokens,
        "cache_prompt": cache_prompt,
        "chat_template_kwargs": template_kwargs,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True
        if tools_template_kwarg:
            template_kwargs[tools_template_kwarg] = tools
    if json_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema, "strict": True}}
    return payload


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = 900.0, tools_template_kwarg: str | None = None):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._tools_template_kwarg = tools_template_kwarg

    def close(self) -> None:
        self._http.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: bool = False,
        cache_prompt: bool = True,
    ) -> ChatResult:
        payload = build_payload(
            messages, tools=tools, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature,
            thinking=thinking, cache_prompt=cache_prompt, tools_template_kwarg=self._tools_template_kwarg,
        )
        started = time.perf_counter()
        try:
            resp = self._http.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ClientError(f"request to llama-server failed: {exc}") from exc
        wall = time.perf_counter() - started
        if resp.status_code != 200:
            raise ClientError(f"llama-server returned {resp.status_code}: {resp.text[:500]}")
        return to_result(resp.json(), wall, text_fallback=bool(tools))


def to_result(body: dict[str, Any], wall: float, text_fallback: bool = False) -> ChatResult:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    content = strip_thinking(message.get("content") or "")
    calls = parse_tool_calls(message)
    raw_calls = message.get("tool_calls")
    if not calls and text_fallback and content:
        calls, content = parse_text_tool_calls(content)
        if calls:
            raw_calls = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}} for c in calls]
    history_message: dict[str, Any] = {"role": "assistant", "content": content}
    if raw_calls:
        history_message["tool_calls"] = raw_calls
    return ChatResult(
        content=content,
        tool_calls=calls,
        finish_reason=choice.get("finish_reason") or "",
        prompt_tokens=int(timings.get("prompt_n", usage.get("prompt_tokens", 0))),
        completion_tokens=int(timings.get("predicted_n", usage.get("completion_tokens", 0))),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        predicted_ms=float(timings.get("predicted_ms", 0.0)),
        wall_s=wall,
        message=history_message,
    )
