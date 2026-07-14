"""Unit tests for the ScriptedAgent test fake."""

from __future__ import annotations

from typing import Any

import pytest

from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolResultEvent,
)
from t0_agents.testing import ScriptedAgent


async def _drain(run: Any) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for event in run.events():
        events.append(event)
    return events


class TestScriptedAgentResult:
    async def test_result_returns_scripted_value(self) -> None:
        agent = ScriptedAgent(result="22C")
        result = await agent.invoke("weather?").result()
        assert result.response == "22C"

    async def test_awaiting_the_run_is_result(self) -> None:
        result = await ScriptedAgent(result="hi").invoke("q")
        assert result.response == "hi"

    async def test_an_agent_result_is_passed_through_unchanged(self) -> None:
        canned = AgentResult(value="typed", turns=3, episode_id="ep-1")
        run = ScriptedAgent(result=canned).invoke("q")
        result = await run.result()
        assert result is canned
        assert run.episode_id == "ep-1"

    async def test_a_none_result_materialises_as_empty_string(self) -> None:
        result = await ScriptedAgent().invoke("q").result()
        assert result.response == ""


class TestScriptedAgentEvents:
    async def test_events_are_replayed_in_order(self) -> None:
        scripted: list[AgentEvent] = [
            TextDeltaEvent(id="t1", delta="two"),
            ToolResultEvent(id="c1", name="lookup", result="ok"),
            TextEndEvent(id="t1"),
        ]
        events = await _drain(ScriptedAgent(result="two", events=scripted).invoke("q"))
        assert [type(e) for e in events[:3]] == [
            TextDeltaEvent,
            ToolResultEvent,
            TextEndEvent,
        ]

    async def test_a_single_terminal_done_event_is_synthesised(self) -> None:
        run = ScriptedAgent(
            result="done",
            events=[TextDeltaEvent(id="t1", delta="done"), TextEndEvent(id="t1")],
        ).invoke("q")
        events = await _drain(run)
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert events[-1] is done[0]
        assert done[0].result.response == "done"

    async def test_no_scripted_events_still_yields_the_terminal(self) -> None:
        events = await _drain(ScriptedAgent(result="x").invoke("q"))
        assert len(events) == 1
        assert isinstance(events[0], DoneEvent)


class TestScriptedAgentRaises:
    async def test_result_propagates_the_scripted_exception(self) -> None:
        run = ScriptedAgent(raises=ValueError("boom")).invoke("q")
        with pytest.raises(ValueError, match="boom"):
            await run.result()

    async def test_events_propagates_the_scripted_exception(self) -> None:
        run = ScriptedAgent(raises=RuntimeError("kaput")).invoke("q")
        with pytest.raises(RuntimeError, match="kaput"):
            await _drain(run)

    async def test_a_raising_run_emits_no_events(self) -> None:
        run = ScriptedAgent(
            raises=ValueError("x"),
            events=[TextDeltaEvent(id="t1", delta="never")],
        ).invoke("q")
        collected: list[AgentEvent] = []
        with pytest.raises(ValueError):
            async for event in run.events():
                collected.append(event)
        assert collected == []


class TestGatewayFree:
    async def test_no_gateway_is_constructed_or_consulted(self) -> None:
        agent = ScriptedAgent(result="ok")
        assert agent._t0 is None
        assert (await agent.invoke("q").result()).response == "ok"
