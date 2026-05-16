"""Concurrent ``result()`` + ``events()`` consumption on a chat Run.

When ``events()`` is subscribed before — or in the same scheduling
tick as — ``result()``, the Run drives via the streaming path and
both consumers see a consistent view of the same invocation.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    TextDeltaEvent,
)
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk


def _stream_script(deltas: list[str]) -> list[Any]:
    chunks: list[Any] = []
    for i, d in enumerate(deltas):
        chunks.append(chat_chunk([text_chunk(d, block_id=f"t{i}")]))
    chunks.append(final_chunk())
    return chunks


def _make_agent(t0: FakeT0) -> Agent:
    return Agent(
        AgentConfig(name="ch", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


class TestTwoChannelConsumption:
    async def test_events_first_then_result_concurrently(self) -> None:
        # Stream produces two text chunks and a final chunk.  The
        # consumer that iterates events() locks the Run into streaming
        # mode; the consumer that awaits result() reads the captured
        # AgentResult.  Both happen concurrently and observe the same
        # Run.
        agent = _make_agent(FakeT0([_stream_script(["alpha", "beta"])]))
        run = agent.invoke("hello")
        events: list[AgentEvent] = []

        async def collect() -> None:
            async for ev in run.events(replay=True):
                events.append(ev)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0)  # ensure events() registers before result()
        result = await run.result()
        await collector

        # The collected events contain at least one TextDeltaEvent and
        # the terminal DoneEvent that carries the AgentResult.
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        dones = [e for e in events if isinstance(e, DoneEvent)]
        assert len(deltas) >= 1
        assert len(dones) == 1
        # The DoneEvent's result matches the result() return value.
        assert dones[0].result is result

    async def test_two_event_consumers_observe_the_same_run(self) -> None:
        agent = _make_agent(FakeT0([_stream_script(["alpha", "beta"])]))
        run = agent.invoke("hello")

        first: list[AgentEvent] = []
        second: list[AgentEvent] = []

        async def collect(out: list[AgentEvent]) -> None:
            async for ev in run.events(replay=True):
                out.append(ev)

        t1 = asyncio.create_task(collect(first))
        t2 = asyncio.create_task(collect(second))
        await asyncio.gather(t1, t2)
        # Both iterators see the same logical stream — including the
        # final DoneEvent.
        assert any(isinstance(e, DoneEvent) for e in first)
        assert any(isinstance(e, DoneEvent) for e in second)
