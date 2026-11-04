"""Status-hint synthesis over an :data:`AgentEvent` stream.

The status policy a production consumer ran in front of its mobile SSE
wire, as pure library functions.  The translator watches events flow
past and interleaves opaque, user-facing :class:`Status` hints:

* ``"Calling {name}…"`` for every :class:`ToolCallEvent` — never
  deduplicated (two calls are two hints).  ``server``/``tool`` carry the
  decomposed identity so clients can localise; ``text`` keeps the raw
  name as an English fallback.
* ``"Thinking…"`` when a reasoning block opens — deduplicated within a
  contiguous thought block, and suppressed entirely once a tool call has
  fired (clients render only the latest hint; trailing reasoning would
  clobber the more specific per-tool hint).  A terminal
  ``DoneEvent | RunErrorEvent`` re-arms both rules for the next turn.

Composition belongs at the endpoint/consumer boundary — never inside a
worker or on a bus — so the translator is a free function over any event
iterator.  Glow wires it as ``FluentRun.statuses()``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    RunErrorEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
)

__all__ = ["Status", "split_tool_identity", "synthesize_statuses"]


@dataclass(slots=True)
class Status:
    """One opaque status hint — deliberately NOT part of the event union.

    ``text`` is the ready-to-render English fallback; ``server`` and
    ``tool`` are the decomposed tool identity (``None`` for hints that
    have no tool, and for tools whose name matched no known server
    prefix — attributing those is consumer policy).
    """

    text: str
    server: str | None = None
    tool: str | None = None


def split_tool_identity(
    name: str, known_servers: Sequence[str]
) -> tuple[str | None, str]:
    """Split a namespaced tool name into ``(server, bare_tool)``.

    Longest known prefix wins, so a server whose name prefixes another
    (``fred`` vs ``fred_api``) never steals the longer match.  A name
    matching no known server returns ``(None, name)`` unchanged.
    """
    for key in sorted(known_servers, key=len, reverse=True):
        if name.startswith(f"{key}_"):
            return key, name[len(key) + 1 :]
    return None, name


async def synthesize_statuses(
    events: AsyncIterator[AgentEvent],
    *,
    known_servers: Sequence[str] = (),
) -> AsyncIterator[AgentEvent | Status]:
    """Re-emit ``events`` verbatim, interleaving :class:`Status` hints.

    Every input event is yielded first, then any hint it triggers —
    ordering a consumer can rely on.  See the module docstring for the
    full policy.
    """
    in_thought_block = False
    saw_tool_call = False
    async for event in events:
        yield event
        match event:
            case ToolCallEvent(name=name):
                in_thought_block = False
                saw_tool_call = True
                server, tool = split_tool_identity(name, known_servers)
                yield Status(text=f"Calling {name}…", server=server, tool=tool)
            case ThoughtEvent() | ThoughtDeltaEvent():
                if not in_thought_block and not saw_tool_call:
                    yield Status(text="Thinking…")
                in_thought_block = True
            case DoneEvent() | RunErrorEvent():
                in_thought_block = False
                saw_tool_call = False
            case _:
                # Internal turn boundaries and tool results keep the
                # per-tool hint standing; only a terminal re-arms it.
                in_thought_block = False
