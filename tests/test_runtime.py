from pathlib import Path

import pytest

from margin.agent.tools import TOOLS, openai_tools, validate_arguments
from margin.hardware import Gpu, Hardware, choose_backend
from margin.runtime.binaries import UnsupportedPlatform, asset_names
from margin.runtime.client import parse_tool_calls, strip_thinking
from margin.runtime.server import ServerConfig, build_args


def _hw(os="windows", arch="x64", gpu=None):
    return Hardware(os=os, arch=arch, physical_cores=8, logical_cores=16, ram_gb=24.0, gpu=gpu)


NVIDIA = Gpu(vendor="nvidia", name="RTX 4060", vram_gb=8.0)


@pytest.mark.parametrize(
    ("hw", "expected"),
    [(_hw(), "cpu"), (_hw(gpu=NVIDIA), "cuda"), (_hw(os="linux", gpu=NVIDIA), "vulkan"), (_hw(os="macos", arch="arm64"), "metal")],
)
def test_choose_backend(hw, expected, monkeypatch):
    monkeypatch.delenv("MARGIN_BACKEND", raising=False)
    assert choose_backend(hw) == expected


def test_backend_override_is_validated():
    assert choose_backend(_hw(gpu=NVIDIA), "cpu") == "cpu"
    with pytest.raises(ValueError):
        choose_backend(_hw(), "tpu")


def test_windows_cuda_needs_runtime_dlls():
    names = asset_names("windows", "x64", "cuda", build="b1")
    assert names == ["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]


def test_linux_has_no_cuda_build():
    with pytest.raises(UnsupportedPlatform):
        asset_names("linux", "x64", "cuda")


def test_cpu_server_offloads_nothing():
    args = build_args(ServerConfig(exe=Path("llama-server"), model=Path("m.gguf"), gpu=False, threads=8), port=5000)
    assert args[args.index("-ngl") + 1] == "0"
    assert "--jinja" in args and args[args.index("--host") + 1] == "127.0.0.1"


def test_parse_tool_calls_handles_strings_dicts_and_bad_json():
    message = {"tool_calls": [
        {"id": "a", "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}},
        {"id": "b", "function": {"name": "calculate", "arguments": {"expression": "3"}}},
        {"id": "c", "function": {"name": "calculate", "arguments": "{not json"}},
    ]}
    calls = parse_tool_calls(message)
    assert calls[0].arguments == {"expression": "2+2"}
    assert calls[1].arguments == {"expression": "3"}
    assert calls[2].arguments is None


def test_payload_is_deterministic_and_passes_tools_where_the_template_reads_them():
    from margin.runtime.client import build_payload

    tools = [{"type": "function", "function": {"name": "calculate"}}]
    plain = build_payload([{"role": "user", "content": "hi"}], tools=tools)
    assert plain["temperature"] == 0.0 and plain["seed"] == 42
    assert plain["tools"] == tools and "xml_tools" not in plain["chat_template_kwargs"]

    smol = build_payload([{"role": "user", "content": "hi"}], tools=tools, tools_template_kwarg="xml_tools")
    assert smol["chat_template_kwargs"]["xml_tools"] == tools and smol["tools"] == tools

    no_tools = build_payload([{"role": "user", "content": "hi"}], tools_template_kwarg="xml_tools")
    assert "tools" not in no_tools and "xml_tools" not in no_tools["chat_template_kwargs"]


def test_text_tool_calls_are_recovered_in_every_written_format():
    from margin.runtime.client import parse_text_tool_calls

    hermes = 'Sure.\n<tool_call>\n{"name": "search_book", "arguments": {"query": "nephron"}}\n</tool_call>'
    calls, rest = parse_text_tool_calls(hermes)
    assert [(c.name, c.arguments, c.source) for c in calls] == [("search_book", {"query": "nephron"}, "text")]
    assert rest == "Sure."

    phi = '<|tool_call|>[{"name": "calculate", "arguments": {"expression": "2**10"}}, {"name": "calculate", "arguments": "{\\"expression\\": \\"3**7\\"}"}]<|/tool_call|>'
    calls, rest = parse_text_tool_calls(phi)
    assert [c.arguments for c in calls] == [{"expression": "2**10"}, {"expression": "3**7"}] and rest == ""

    broken, _ = parse_text_tool_calls("<tool_call>{not json</tool_call>")
    assert broken[0].name == "" and broken[0].arguments is None

    assert parse_text_tool_calls("No tools needed here.") == ((), "No tools needed here.")


def test_text_fallback_only_applies_when_tools_were_offered_and_none_parsed():
    from margin.runtime.client import to_result

    body = {"choices": [{"message": {"content": '<tool_call>{"name": "calculate", "arguments": {"expression": "1+1"}}</tool_call>'}, "finish_reason": "stop"}]}
    with_tools = to_result(body, 0.1, text_fallback=True)
    assert with_tools.tool_calls[0].name == "calculate" and with_tools.message["tool_calls"][0]["function"]["name"] == "calculate"
    assert to_result(body, 0.1, text_fallback=False).tool_calls == ()


def test_chat_template_override_reaches_server_args():
    from margin.runtime.models import get

    spec = get("phi4-mini")
    template = spec.chat_template_path()
    assert template is not None and template.exists()
    args = build_args(ServerConfig(exe=Path("s"), model=Path("m"), gpu=True, threads=4, chat_template_file=template), port=1)
    assert args[args.index("--chat-template-file") + 1] == str(template)
    assert "--chat-template-file" not in build_args(ServerConfig(exe=Path("s"), model=Path("m"), gpu=True, threads=4), port=1)


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>\nAnswer") == "Answer"


def test_every_tool_schema_forbids_extra_arguments():
    schemas = openai_tools()
    assert len(schemas) == len(TOOLS)
    assert all(s["function"]["parameters"].get("additionalProperties") is False for s in schemas)


def test_validate_arguments():
    assert validate_arguments("calculate", {"expression": "1+1"}) is None
    assert validate_arguments("calculate", {"expr": "1+1"}) is not None
    assert validate_arguments("generate_questions", {"topic": "x", "count": 3, "marks": 2, "question_type": "essay"}) is not None
    assert validate_arguments("summon_demon", {}) == "unknown tool 'summon_demon'"
    assert validate_arguments("calculate", None) == "arguments were not valid JSON"
