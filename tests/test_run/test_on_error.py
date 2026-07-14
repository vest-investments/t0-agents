"""``Run.events(on_error=...)`` terminal-invariant behaviour.

``on_error="raise"`` (default) preserves the historical contract: fatal
drive-time failures re-raise out of the iterator.  ``on_error="event"``
converts them into a single terminal :class:`RunErrorEvent`, so every
stream in that mode ends with exactly one ``DoneEvent | RunErrorEvent``.
Cancellation is never converted in either mode.

Uses :class:`ScriptedAgent` — the gateway-free fake replays events and
fails through the same ``_drive``/buffer machinery real runs use, so the
post-drain branch under test is the production code path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from t0_agents import DoneEvent, RunCancelledError, RunErrorEvent, TextDeltaEvent
from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.stream.events import AgentEvent
from t0_agents.testing import ScriptedAgent


async def _drain(run_events: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    collected: list[AgentEvent] = []
    async for event in run_events:
        collected.append(event)
    return collected


class TestOnErrorRaise:
    async def test_default_reraises_the_original_exception(self) -> None:
        agent = ScriptedAgent(raises=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await _drain(agent.invoke("q").events())

    async def test_events_before_the_failure_still_stream(self) -> None:
        delta = TextDeltaEvent(id="t1", delta="partial")
        agent = ScriptedAgent(events=[delta], raises=ValueError("boom"))
        run = agent.invoke("q")
        seen: list[AgentEvent] = []
        with pytest.raises(ValueError):
            async for event in run.events():
                seen.append(event)
        assert seen == [delta]


class TestOnErrorEvent:
    async def test_failure_becomes_a_single_terminal_run_error(self) -> None:
        agent = ScriptedAgent(raises=ValueError("boom"))
        events = await _drain(agent.invoke("q").events(on_error="event"))
        assert len(events) == 1
        terminal = events[0]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.error_type == "ValueError"
        assert terminal.message == "boom"
        assert terminal.detail == {}

    async def test_mid_stream_abort_keeps_prior_events(self) -> None:
        delta = TextDeltaEvent(id="t1", delta="partial")
        agent = ScriptedAgent(events=[delta], raises=RuntimeError("died"))
        events = await _drain(agent.invoke("q").events(on_error="event"))
        assert events[0] == delta
        assert isinstance(events[-1], RunErrorEvent)
        assert len([e for e in events if isinstance(e, RunErrorEvent)]) == 1

    async def test_original_exception_rides_the_event(self) -> None:
        cause = RuntimeError("died")
        agent = ScriptedAgent(raises=cause)
        events = await _drain(agent.invoke("q").events(on_error="event"))
        terminal = events[-1]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.exception is cause

    async def test_max_turns_contributes_structured_detail(self) -> None:
        agent = ScriptedAgent(raises=MaxTurnsExceededError(20))
        events = await _drain(agent.invoke("q").events(on_error="event"))
        terminal = events[-1]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.error_type == "MaxTurnsExceededError"
        assert terminal.detail == {"max_turns": 20}

    async def test_blank_str_exception_falls_back_to_class_name(self) -> None:
        agent = ScriptedAgent(raises=TimeoutError())
        events = await _drain(agent.invoke("q").events(on_error="event"))
        terminal = events[-1]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.message == "TimeoutError"

    async def test_message_prefers_the_exceptions_text_attribute(self) -> None:
        # TensorZeroError carries the provider's detailed error fragments
        # on ``.text`` while str(e) is often a generic status line.
        class _GatewayError(Exception):
            def __init__(self) -> None:
                super().__init__("HTTP 502")
                self.text = "provider rejected: model overloaded"

        agent = ScriptedAgent(raises=_GatewayError())
        events = await _drain(agent.invoke("q").events(on_error="event"))
        terminal = events[-1]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.message == "provider rejected: model overloaded"

    async def test_non_str_text_attribute_is_ignored(self) -> None:
        class _OddError(Exception):
            text = None

        agent = ScriptedAgent(raises=_OddError("odd"))
        events = await _drain(agent.invoke("q").events(on_error="event"))
        terminal = events[-1]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.message == "odd"

    async def test_successful_run_emits_done_and_no_run_error(self) -> None:
        agent = ScriptedAgent(result="fine")
        events = await _drain(agent.invoke("q").events(on_error="event"))
        assert isinstance(events[-1], DoneEvent)
        assert not any(isinstance(e, RunErrorEvent) for e in events)

    async def test_replay_of_a_failed_run_ends_with_one_terminal_again(self) -> None:
        agent = ScriptedAgent(raises=ValueError("boom"))
        run = agent.invoke("q")
        first = await _drain(run.events(on_error="event"))
        second = await _drain(run.events(replay=True, on_error="event"))
        for drained in (first, second):
            assert len([e for e in drained if isinstance(e, RunErrorEvent)]) == 1
            assert isinstance(drained[-1], RunErrorEvent)

    async def test_cancellation_is_never_converted(self) -> None:
        agent = ScriptedAgent(result="never")
        run = agent.invoke("q")
        await run.cancel()
        with pytest.raises(RunCancelledError):
            await _drain(run.events(on_error="event"))

    async def test_result_still_raises_after_event_mode_drain(self) -> None:
        agent = ScriptedAgent(raises=ValueError("boom"))
        run = agent.invoke("q")
        await _drain(run.events(on_error="event"))
        with pytest.raises(ValueError, match="boom"):
            await run.result()


class TestScriptedReplayOrder:
    async def test_events_replay_before_raises_fires(self) -> None:
        script = [
            TextDeltaEvent(id="t1", delta="a"),
            TextDeltaEvent(id="t1", delta="b"),
        ]
        agent = ScriptedAgent(events=script, raises=ValueError("boom"))
        run = agent.invoke("q")
        seen: list[AgentEvent] = []
        with pytest.raises(ValueError):
            async for event in run.events():
                seen.append(event)
        assert seen == script
        assert run.state == "failed"

    async def test_cancelled_scripted_run_state(self) -> None:
        agent = ScriptedAgent(events=[TextDeltaEvent(id="t1", delta="a")], result="x")
        run = agent.invoke("q")
        await run.cancel()
        assert run.state == "cancelled"
        await asyncio.sleep(0)
