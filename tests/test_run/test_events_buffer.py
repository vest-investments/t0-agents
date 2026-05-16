"""Tests for the bounded :class:`EventBuffer` that backs ``Run.events()``.

Drives the buffer directly with synthetic events so the assertions are
not entangled with the streaming runner's behaviour.  The buffer's
public contract is small: ``put`` appends, ``close`` ends the stream,
``iter_from(start_cursor=...)`` yields from a logical cursor and
respects ``truncated`` after overflow.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from t0_agents.core._runtime import EventBuffer
from t0_agents.stream.events import (
    ErrorEvent,
    TextDeltaEvent,
    TurnEndEvent,
)


def _delta(i: int) -> TextDeltaEvent:
    return TextDeltaEvent(id="t1", delta=f"d{i}")


class TestLiveCursor:
    async def test_live_starts_at_current_total_pushed(self) -> None:
        buf = EventBuffer()
        await buf.put(_delta(0))
        await buf.put(_delta(1))
        # Subscribe at total_pushed=2 — only future events.
        cursor = buf.total_pushed
        events: list[str] = []

        async def consume() -> None:
            async for ev in buf.iter_from(start_cursor=cursor):
                assert isinstance(ev, TextDeltaEvent)
                events.append(ev.delta)

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await buf.put(_delta(2))
        await buf.put(_delta(3))
        await buf.close()
        await consumer
        assert events == ["d2", "d3"]

    async def test_live_sees_no_buffered_events_when_subscribed_at_end(
        self,
    ) -> None:
        buf = EventBuffer()
        await buf.put(_delta(0))
        cursor = buf.total_pushed
        await buf.close()
        events = [ev async for ev in buf.iter_from(start_cursor=cursor)]
        assert events == []


class TestReplay:
    async def test_replay_from_zero_yields_full_history(self) -> None:
        buf = EventBuffer()
        await buf.put(_delta(0))
        await buf.put(_delta(1))
        await buf.put(_delta(2))
        await buf.close()
        events = [ev async for ev in buf.iter_from(start_cursor=0)]
        assert [e.delta for e in events] == ["d0", "d1", "d2"]


class TestDoubleSubscribe:
    async def test_two_iterators_see_the_same_events_concurrently(
        self,
    ) -> None:
        buf = EventBuffer()
        first: list[str] = []
        second: list[str] = []

        async def consume(out: list[str]) -> None:
            async for ev in buf.iter_from(start_cursor=0):
                assert isinstance(ev, TextDeltaEvent)
                out.append(ev.delta)

        t1 = asyncio.create_task(consume(first))
        t2 = asyncio.create_task(consume(second))
        await asyncio.sleep(0)
        await buf.put(_delta(0))
        await buf.put(_delta(1))
        await asyncio.sleep(0)
        await buf.close()
        await asyncio.gather(t1, t2)
        assert first == ["d0", "d1"]
        assert second == ["d0", "d1"]

    async def test_second_subscription_after_first_completes_resumes_live(
        self,
    ) -> None:
        # First consumer takes events 0..1 then breaks (terminates).
        # A second consumer subscribed at live cursor sees only the
        # NEXT events the producer emits — events 0..1 are not re-seen.
        buf = EventBuffer()
        await buf.put(_delta(0))
        await buf.put(_delta(1))
        await asyncio.sleep(0)
        live_cursor = buf.total_pushed  # post-first-batch live cursor
        await buf.put(_delta(2))
        await buf.put(_delta(3))
        await buf.close()
        events = [
            ev async for ev in buf.iter_from(start_cursor=live_cursor)
        ]
        assert [e.delta for e in events] == ["d2", "d3"]


class TestOverflow:
    """Verify the 10 000-event cap, truncated flag, and WARN log."""

    async def test_truncated_flag_set_after_overflow(self) -> None:
        buf = EventBuffer()
        # Push MAX_EVENTS + 1 to force one drop.
        for i in range(buf.MAX_EVENTS + 1):
            await buf.put(_delta(i))
        assert buf.truncated is True

    async def test_replay_after_overflow_starts_at_first_available(
        self,
    ) -> None:
        # Make MAX_EVENTS small for tractable assertions via subclassing
        # — EventBuffer reads from ``self.MAX_EVENTS``.
        class _SmallBuffer(EventBuffer):
            MAX_EVENTS = 3

        small = _SmallBuffer()
        for i in range(5):
            await small.put(_delta(i))
        # The oldest 2 events were dropped; only d2, d3, d4 remain.
        await small.close()
        events = [ev async for ev in small.iter_from(start_cursor=0)]
        deltas = [e.delta for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas == ["d2", "d3", "d4"]
        assert small.truncated is True

    async def test_overflow_emits_one_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="t0_agents.core._runtime")

        class _Tiny(EventBuffer):
            MAX_EVENTS = 2

        tiny = _Tiny()
        for i in range(6):
            await tiny.put(_delta(i))
        warnings = [
            r for r in caplog.records
            if r.name == "t0_agents.core._runtime"
            and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1, (
            "Overflow warning must fire exactly once per buffer instance, "
            f"not once per drop.  Got {len(warnings)} warnings."
        )
        assert "EventBuffer overflow" in warnings[0].getMessage()


class TestPutAfterCloseFails:
    async def test_put_after_close_is_programmer_error(self) -> None:
        buf = EventBuffer()
        await buf.close()
        with pytest.raises(RuntimeError, match="programmer error"):
            await buf.put(_delta(0))


class TestMixedEventTypes:
    async def test_buffer_is_event_type_agnostic(self) -> None:
        buf = EventBuffer()
        await buf.put(_delta(0))
        await buf.put(TurnEndEvent(turn=1))
        await buf.put(ErrorEvent(message="oops"))
        await buf.close()
        events = [ev async for ev in buf.iter_from(start_cursor=0)]
        assert len(events) == 3
        assert isinstance(events[0], TextDeltaEvent)
        assert isinstance(events[1], TurnEndEvent)
        assert isinstance(events[2], ErrorEvent)
