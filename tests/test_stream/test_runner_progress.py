"""E4 end-to-end — ``ToolProgressEvent`` flows into the event stream.

TensorZero is faked (scripted chunks); the tool side is the REAL
in-memory FastMCP server from ``tests/server.py``, so
``ctx.report_progress`` rides the genuine notification path:
server → fastmcp client → loop ``progress_handler`` → runner sink →
``ToolProgressEvent``.  The agent-level test additionally covers
``Agent._chat_streaming_terminal`` wiring the Run buffer as the sink, so
``Run.events()`` consumers observe progress interleaved between the
``ToolCallEvent`` and its ``ToolResultEvent``.
"""

from __future__ import annotations

from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import (
    Agent,
    AgentConfig,
    DoneEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
)
from t0_agents.core.types import ToolProvider
from t0_agents.stream.events import AgentEvent
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk, tool_call_chunk
from tests.server import mcp


def _tool_turn_t0() -> FakeT0:
    return FakeT0(
        responses=[
            [
                chat_chunk(
                    [
                        tool_call_chunk(
                            raw_name="count_with_progress",
                            raw_arguments='{"steps": 2}',
                            block_id="c1",
                        )
                    ]
                ),
                final_chunk(),
            ],
            [
                chat_chunk([text_chunk("counted!", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )


async def test_runner_sink_receives_tool_progress_events() -> None:
    sink_events: list[ToolProgressEvent] = []

    async def sink(event: ToolProgressEvent) -> None:
        sink_events.append(event)

    runner = StreamingAgentLoop()
    events: list[AgentEvent] = []
    async with Client(mcp) as client:
        async for event in runner.stream(
            cast(AsyncTensorZeroGateway, _tool_turn_t0()),
            client,
            "count to 2",
            "agent",
            progress_sink=sink,
        ):
            events.append(event)

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert sink_events == [
        ToolProgressEvent(id="c1", progress=1.0, total=2.0, message="step 1"),
        ToolProgressEvent(id="c1", progress=2.0, total=2.0, message="step 2"),
    ]
    assert sink_events[0].id == tool_calls[0].id
    assert isinstance(events[-1], DoneEvent)


async def test_agent_run_events_interleave_progress() -> None:
    config = AgentConfig(name="test", function_name="agent")
    async with Client(mcp) as client:
        agent = Agent(
            config,
            t0=cast(AsyncTensorZeroGateway, _tool_turn_t0()),
            tool_provider=cast(ToolProvider, client),
        )
        run: Any = agent.invoke("count to 2")
        events = [event async for event in run.events()]

    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert [(e.progress, e.total, e.message) for e in progress] == [
        (1.0, 2.0, "step 1"),
        (2.0, 2.0, "step 2"),
    ]
    call_index = next(i for i, e in enumerate(events) if isinstance(e, ToolCallEvent))
    result_index = next(i for i, e in enumerate(events) if isinstance(e, ToolResultEvent))
    progress_indexes = [i for i, e in enumerate(events) if isinstance(e, ToolProgressEvent)]
    assert all(call_index < i < result_index for i in progress_indexes)
    assert progress[0].id == events[call_index].id
    assert isinstance(events[-1], DoneEvent)
