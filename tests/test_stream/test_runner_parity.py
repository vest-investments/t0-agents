"""Regression tests for the streaming/non-streaming dispatch parity fixes.

Each test pins a behavior where the streaming loop historically diverged
from ``run_agent``'s dispatch contract (``_execute_single_tool_call``):
invalid LLM JSON and hallucinated tool names must produce the error
string for the model WITHOUT dispatching the tool — exactly like the
non-streaming loop, where TensorZero zeroes those fields itself.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway

from t0_agents.stream.events import AgentEvent, ErrorEvent, ToolResultEvent
from t0_agents.stream.runner import StreamingAgentLoop
from t0_agents.wire.frames import ToolCallFrame
from t0_agents.wire.stages import project
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk, tool_call_chunk
from tests.test_parity._scenario import _tool_result

_ADD_TOOL = MCPTool(
    name="add",
    description="Add two integers.",
    inputSchema={"type": "object", "properties": {}},
)


async def _run_one_tool_turn(
    raw_name: str, raw_arguments: str
) -> tuple[FakeT0, FakeMCPClient, list[Any]]:
    """Drive one tool turn + one final text turn; return (t0, client, events)."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([tool_call_chunk(raw_name, raw_arguments, block_id="c1")]),
                final_chunk(),
            ],
            [chat_chunk([text_chunk("done", block_id="t1")]), final_chunk()],
        ],
    )
    client = FakeMCPClient(
        tools=[_ADD_TOOL], call_results={"add": _tool_result("3")}
    )
    runner = StreamingAgentLoop()
    events = [
        e
        async for e in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0), cast(Any, client), "Q?", "agent"
        )
    ]
    return fake_t0, client, events


async def test_invalid_json_args_are_not_dispatched() -> None:
    """Parity fix: invalid LLM JSON must not reach the tool with ``{}`` args."""
    fake_t0, client, events = await _run_one_tool_turn("add", "{not json")

    assert client.call_tool_invocations == []

    (result,) = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result.is_error is True
    assert result.result == "ERROR: invalid JSON arguments from LLM: {not json"

    replayed_assistant = fake_t0.calls[1]["input"]["messages"][1]
    (tool_call,) = replayed_assistant["content"]
    assert tool_call.arguments is None
    assert tool_call.raw_arguments == "{not json"


async def test_non_object_json_args_are_not_dispatched() -> None:
    """Valid-but-non-object JSON follows the same no-dispatch contract."""
    _, client, events = await _run_one_tool_turn("add", '"hello"')

    assert client.call_tool_invocations == []
    (result,) = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result.is_error is True
    assert result.result == 'ERROR: invalid JSON arguments from LLM: "hello"'


async def test_hallucinated_tool_name_is_not_dispatched() -> None:
    """Parity fix: a name outside the advertised schema must not reach fastmcp.

    TensorZero zeroes unknown names on the non-streaming path and the loop
    answers with a schema-error string without dispatching; streaming
    historically forwarded the hallucinated name to the provider.
    """
    fake_t0, client, events = await _run_one_tool_turn("bogus", '{"x": 1}')

    assert client.call_tool_invocations == []

    (result,) = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result.is_error is True
    assert result.result == "ERROR: tool name 'bogus' is not in the schema"

    replayed_assistant = fake_t0.calls[1]["input"]["messages"][1]
    (tool_call,) = replayed_assistant["content"]
    assert tool_call.name is None
    assert tool_call.raw_name == "bogus"


async def test_empty_args_tool_call_still_dispatches() -> None:
    """An empty raw-argument stream is a legitimate no-args call, not an error."""
    _, client, events = await _run_one_tool_turn("add", "")

    assert client.call_tool_invocations == [("add", {}, 30.0)]
    (result,) = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result.is_error is False
    assert result.result == "3"


async def test_salvaged_args_dispatch_with_full_arguments_and_no_error_event() -> None:
    """A complete object plus closing noise is dispatched as if it were clean.

    The recovered dict reaches the tool, no recoverable ``ErrorEvent`` is
    emitted, and the replayed assistant block carries the dict with its
    canonical serialisation — not the noisy raw string.
    """
    fake_t0, client, events = await _run_one_tool_turn("add", '{"a": 2, "b": 3}}')

    assert client.call_tool_invocations == [("add", {"a": 2, "b": 3}, 30.0)]
    assert [e for e in events if isinstance(e, ErrorEvent)] == []

    (result,) = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result.is_error is False
    assert result.result == "3"

    replayed_assistant = fake_t0.calls[1]["input"]["messages"][1]
    (tool_call,) = replayed_assistant["content"]
    assert tool_call.arguments == {"a": 2, "b": 3}
    assert tool_call.raw_arguments == '{"a": 2, "b": 3}'


async def test_salvaged_call_wire_arguments_equal_what_the_tool_received() -> None:
    """Runtime ≡ replay ≡ wire for a salvaged call — the parity this fix exists for.

    The wire's ``arguments_json`` must parse to exactly the dict the tool
    executed with and be byte-identical to the replayed ``raw_arguments``,
    non-ASCII included.
    """
    raw = '{"a": 2, "b": 3, "note": "Añasco"}\n}'
    fake_t0, client, events = await _run_one_tool_turn("add", raw)

    async def _events() -> AsyncIterator[AgentEvent]:
        for event in events:
            yield event

    (frame,) = [f async for f in project(_events()) if isinstance(f, ToolCallFrame)]
    (invocation,) = client.call_tool_invocations
    (tool_call,) = fake_t0.calls[1]["input"]["messages"][1]["content"]

    assert frame.truncated_from is None
    assert json.loads(frame.arguments_json) == invocation[1] == tool_call.arguments
    assert frame.arguments_json == tool_call.raw_arguments == '{"a": 2, "b": 3, "note": "Añasco"}'
