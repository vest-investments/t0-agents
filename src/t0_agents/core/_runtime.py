"""Internal runtime primitives for :class:`t0_agents.core.run.Run`.

Houses the :class:`EventBuffer` that backs ``Run.events()`` and
``Run.stream_text()``.  Lives in a private module (leading underscore)
because the buffer's surface is an implementation detail of ``Run``;
public consumers go through ``Run`` rather than allocating buffers
directly.

The buffer is bounded (default :data:`EventBuffer.MAX_EVENTS` = 10 000)
to keep long-running agents from leaking memory.  When the cap is
reached the oldest event is dropped, :attr:`EventBuffer.truncated`
latches ``True``, and a single WARN log fires per buffer instance so
operators see the overflow without log spam.  Each consumer iterator
tracks its own logical cursor, so concurrent consumers do not affect
each other — and ``replay=True`` consumers see every event that is
still in the buffer (which is the full history when no truncation
has occurred).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from t0_agents.stream.events import AgentEvent

logger = logging.getLogger(__name__)


class EventBuffer:
    """Bounded, replayable event buffer for one :class:`Run`.

    Producers call :meth:`put` once per event and :meth:`close` once at
    end-of-stream.  Consumers call :meth:`iter_from` to read events
    starting at a logical cursor (``self.total_pushed`` for "live",
    ``0`` for "replay").  The buffer wakes every waiting consumer via
    an :class:`asyncio.Condition` whenever a new event arrives or the
    buffer is closed.

    Concurrent consumers each maintain their own cursor; the buffer
    holds the underlying storage and the synchronisation primitive.
    """

    MAX_EVENTS: int = 10_000

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []
        self._drop_count: int = 0
        self._total_pushed: int = 0
        self._truncated: bool = False
        self._warned_truncation: bool = False
        self._closed = asyncio.Event()
        self._cond = asyncio.Condition()

    @property
    def total_pushed(self) -> int:
        """Logical index that the next ``put`` will assign."""
        return self._total_pushed

    @property
    def truncated(self) -> bool:
        """``True`` if any oldest events were dropped to honour the cap."""
        return self._truncated

    @property
    def closed(self) -> bool:
        """``True`` after :meth:`close` has been called."""
        return self._closed.is_set()

    async def put(self, event: AgentEvent) -> None:
        """Append ``event`` and wake every waiting consumer.

        Drops the oldest event when the buffer is already at
        :attr:`MAX_EVENTS`.  Sets :attr:`truncated` on the first drop
        and emits a single WARN log so operators see the overflow.
        """
        async with self._cond:
            if self._closed.is_set():
                raise RuntimeError(
                    "EventBuffer.put() called after close() — programmer error"
                )
            self._events.append(event)
            self._total_pushed += 1
            if len(self._events) > self.MAX_EVENTS:
                self._events.pop(0)
                self._drop_count += 1
                if not self._truncated:
                    self._truncated = True
                if not self._warned_truncation:
                    logger.warning(
                        "EventBuffer overflow: more than %d events buffered; "
                        "oldest events are being dropped.  Run.replay() / "
                        "events(replay=True) will surface only the most "
                        "recent %d events.",
                        self.MAX_EVENTS,
                        self.MAX_EVENTS,
                    )
                    self._warned_truncation = True
            self._cond.notify_all()

    async def close(self) -> None:
        """Mark the buffer closed and wake every waiting consumer."""
        async with self._cond:
            self._closed.set()
            self._cond.notify_all()

    async def iter_from(self, *, start_cursor: int) -> AsyncIterator[AgentEvent]:
        """Yield events starting at the given logical cursor.

        The cursor is a logical index into the stream of pushed events
        (not a physical index into the underlying list).  When the
        cursor is below the earliest still-buffered event because of
        truncation, the iterator skips forward to the first available
        event.  When the cursor reaches the current head, the iterator
        waits on the :class:`asyncio.Condition` until either a new
        event arrives or the buffer is closed.
        """
        cursor = max(0, start_cursor)
        while True:
            await self._cond.acquire()
            try:
                while cursor >= self._total_pushed and not self._closed.is_set():
                    await self._cond.wait()
                if cursor >= self._total_pushed and self._closed.is_set():
                    return
                if cursor < self._drop_count:
                    cursor = self._drop_count
                    if cursor >= self._total_pushed:
                        # The buffer was closed between catch-up and read;
                        # loop back to the closed-check at the top.
                        continue
                event = self._events[cursor - self._drop_count]
                cursor += 1
            finally:
                self._cond.release()
            yield event
