"""Tests for :meth:`Agent.stream` — memory, guardrails, regression."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import GuardrailLogEntry
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.memory.in_memory import InMemoryMemory
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    TextDeltaEvent,
)
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import (
    FakeT0,
    chat_chunk,
    final_chunk,
    text_chunk,
)


async def _drain_stream(
    agent: Agent,
    message: str,
    episode_id: str | None = None,
) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent.stream(message, episode_id=episode_id):
        events.append(ev)
    return events


def _tool_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        data=text,
        is_error=False,
        meta=None,
    )


async def test_agent_stream_text_only() -> None:
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("hello", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(name="a", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, fake_t0),
        tool_provider=client,
    )
    events = await _drain_stream(agent, "hi")

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].result.response == "hello"
    assert any(isinstance(e, TextDeltaEvent) for e in events)


async def test_agent_stream_persists_memory() -> None:
    memory = InMemoryMemory()
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("remembered", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(name="mem", function_name="agent", memory=memory),
        t0=cast(AsyncTensorZeroGateway, fake_t0),
        tool_provider=client,
    )
    events = await _drain_stream(agent, "hola")
    assert isinstance(events[-1], DoneEvent)

    # Memory persisted under the allocated episode_id.
    saved = await memory.load(events[-1].result.episode_id or "")
    assert len(saved) == 1
    assert saved[0].user_message == "hola"
    assert saved[0].agent_response == "remembered"


async def test_agent_stream_invokes_before_after_run_guardrails() -> None:
    calls: list[str] = []

    class RecorderGuardrail:
        name = "recorder"

        async def before_run(self, ctx: GuardrailContext) -> None:
            calls.append("before_run")

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            calls.append("before_tool")
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            calls.append("after_tool")

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            calls.append("after_run")

    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("done", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(
            name="g",
            function_name="agent",
            guardrails=[RecorderGuardrail()],
        ),
        t0=cast(AsyncTensorZeroGateway, fake_t0),
        tool_provider=client,
    )
    events = await _drain_stream(agent, "gate me")

    assert calls == ["before_run", "after_run"]
    assert isinstance(events[-1], DoneEvent)
    # Guardrail log attached to the final result.
    entries = events[-1].result.guardrail_log
    assert [e.hook for e in entries] == ["before_run", "after_run"]
    assert all(isinstance(e, GuardrailLogEntry) for e in entries)


async def test_agent_stream_fatal_guardrail_propagates() -> None:
    class Blocker:
        name = "blocker"

        async def before_run(self, ctx: GuardrailContext) -> None:
            raise GuardrailBlockedError("forbidden")

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            pass

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            pass

    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("should not reach", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(
            name="g2",
            function_name="agent",
            guardrails=[Blocker()],
        ),
        t0=cast(AsyncTensorZeroGateway, fake_t0),
        tool_provider=client,
    )
    with pytest.raises(GuardrailBlockedError):
        async for _ in agent.stream("nope"):
            pass


async def test_agent_run_unchanged_after_phase_e() -> None:
    """Regression: Phase D-style Agent.run still returns the expected result."""
    from tensorzero import Text

    fake_t0 = FakeT0(
        responses=[
            _response_with_text("phase-d-parity"),
        ]
    )
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(name="regr", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, fake_t0),
        tool_provider=client,
    )
    result = await agent.run("phase d?")
    assert result.response == "phase-d-parity"
    assert result.turns == 1
    assert result.response_source == "text"

    # run path never calls stream=True.
    assert all(call.get("stream") is not True for call in fake_t0.calls)
    _ = Text  # silence unused import warning at static-analyser


def _response_with_text(text: str) -> Any:
    """Helper for Agent.run's FakeChatResponse path."""
    from tensorzero import Text

    from tests._fake_t0 import FakeChatResponse

    return FakeChatResponse(content=[Text(type="text", text=text)])
