"""Unit tests for :class:`StreamingAgentLoop`."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.stream.events import (
    DoneEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import (
    FakeT0,
    chat_chunk,
    final_chunk,
    text_chunk,
    tool_call_chunk,
)


def _tool_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        data=text,
        is_error=False,
        meta=None,
    )


async def test_streaming_loop_terminates_on_text_only() -> None:
    """One turn producing text → DoneEvent with the joined text."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("hello ", block_id="t1")]),
                chat_chunk([text_chunk("world", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = []
    async for event in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
    ):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in text_deltas] == ["hello ", "world"]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "hello world"
    assert done[0].result.turns == 1
    assert done[0].result.response_source == "text"
    assert len(done[0].result.inference_ids) == 1

    assert events[-1] is done[0]


async def test_streaming_loop_dispatches_tool_calls() -> None:
    """A tool-call turn emits ToolCallEvent, dispatches, feeds result back."""
    fake_t0 = FakeT0(
        responses=[
            # Turn 1: one tool call
            [
                chat_chunk(
                    [
                        tool_call_chunk(
                            raw_name="add",
                            raw_arguments='{"a":1,"b":2}',
                            block_id="c1",
                        )
                    ]
                ),
                final_chunk(),
            ],
            # Turn 2: text response
            [
                chat_chunk([text_chunk("sum is 3", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    fake_client = FakeMCPClient(
        tools=[
            MCPTool(
                name="add",
                description="add two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            )
        ],
        call_results={"add": _tool_result("3")},
    )

    runner = StreamingAgentLoop()
    events = []
    async for event in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "What is 1+2?",
        "agent",
    ):
        events.append(event)

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"a": 1, "b": 2}

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].id == "c1"
    assert tool_results[0].name == "add"

    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert [t.turn for t in turn_ends] == [1, 2]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "sum is 3"
    assert done[0].result.turns == 2


async def test_streaming_loop_max_turns_raises() -> None:
    """Exhausting max_turns surfaces MaxTurnsExceededError."""
    # Each turn emits one tool call that the client can dispatch indefinitely.
    def _one_tool_turn() -> list:
        return [
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="loop",
                        raw_arguments="{}",
                        block_id="loop",
                    )
                ]
            ),
            final_chunk(),
        ]

    fake_t0 = FakeT0(responses=[_one_tool_turn() for _ in range(5)])
    fake_client = FakeMCPClient(
        tools=[
            MCPTool(
                name="loop",
                description="loop forever",
                inputSchema={"type": "object"},
            )
        ],
        call_results={"loop": _tool_result("still looping")},
    )

    runner = StreamingAgentLoop()
    with pytest.raises(MaxTurnsExceededError):
        events: list = []
        async for event in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "loop forever",
            "agent",
            max_turns=3,
        ):
            events.append(event)


async def test_streaming_loop_forwards_config_knobs() -> None:
    """dryrun/tool_choice/tags/namespace must reach the inference call."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
        dryrun=True,
        tags={"scope": "unit-test"},
        namespace="test-ns",
    ):
        pass

    call = fake_t0.calls[0]
    assert call["stream"] is True
    assert call["dryrun"] is True
    assert call["tags"] == {"scope": "unit-test"}
    assert call["namespace"] == "test-ns"


async def test_streaming_loop_system_string_forwarded() -> None:
    """``system="..."`` reaches the streaming inference call as a raw string."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
        system="Reply only in haiku.",
    ):
        pass

    assert fake_t0.calls[0]["input"]["system"] == "Reply only in haiku."


async def test_streaming_loop_system_and_system_arguments_rejected() -> None:
    """Passing both kwargs raises ``ValueError`` before any chunk is yielded."""
    fake_t0 = FakeT0(responses=[])
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    with pytest.raises(ValueError, match="Pass either `system`"):
        async for _ in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "hi",
            "agent",
            system="raw",
            system_arguments={"foo": "bar"},
        ):
            pass
    assert fake_t0.calls == []
