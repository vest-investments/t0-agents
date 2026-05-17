"""Unit tests for :mod:`t0_agents.glow.selectors` and the
``FluentRun.events`` selector dispatch (M2).

Seven cases over a fake event stream — no TZ gateway, no real
:class:`Run`.  Validates:

* Each module-level selector picks events of its own class only.
* :meth:`Selector.where` AND-chains predicates across repeated calls.
* :meth:`Selector.map` replaces the transform and is applied to the
  yielded value.
* Multiple selectors passed to :meth:`FluentRun.events` union with
  first-match-wins semantics (mirrors :func:`isinstance` dispatch).
* Empty selectors preserve the M1 behaviour (yield every event).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow import (
    Done,
    Error,
    FluentRun,
    Text,
    Thought,
    ToolCall,
    ToolResult,
    TurnEnd,
)
from t0_agents.glow.selectors import Selector
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextEndEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)

# ---------------------------------------------------------------------------
# Stream fakes
# ---------------------------------------------------------------------------


class _StreamFakeRun:
    """Duck-typed :class:`Run` that yields a pre-set event list.

    Constructor takes an event list; :meth:`events` is an async
    generator that yields them in order.  :meth:`result` returns a
    minimal :class:`AgentResult` so the ``await`` path in
    :class:`FluentRun` stays well-defined (tests that drive
    ``events()`` don't ``await`` the FluentRun, but constructing one
    requires the contract to be intact).
    """

    def __init__(self, events: list[AgentEvent]) -> None:
        self._events = list(events)

    async def events(self) -> AsyncIterator[AgentEvent]:
        for event in self._events:
            yield event

    async def result(self) -> AgentResult[Any]:
        return AgentResult(value="ignored", episode_id=None, inference_ids=[])


def _make_fluent_run(events: list[AgentEvent]) -> FluentRun[Any]:
    """Build a :class:`FluentRun` whose underlying ``Run`` yields ``events``."""
    fake = _StreamFakeRun(events)

    def build(_mode: OutputMode[Any]) -> Run[Any]:
        return cast(Run[Any], fake)

    return FluentRun(build, ChatOutput())


# Sample events used across multiple tests.
_TEXT_END = TextEndEvent(id="t1")
_TOOL_CALL = ToolCallEvent(id="c1", name="add", arguments={"a": 1, "b": 2})
_TOOL_RESULT = ToolResultEvent(id="c1", name="add", result="3")
_THOUGHT_FOO = ThoughtEvent(id="th1", text="foo plan")
_THOUGHT_BAR = ThoughtEvent(id="th2", text="bar plan")
_TURN_END = TurnEndEvent(turn=0)
_DONE = DoneEvent(
    result=AgentResult(value="done", episode_id=None, inference_ids=[]),
)
_ERROR = ErrorEvent(message="boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_thought_selector_picks_thought_events_only() -> None:
    fr = _make_fluent_run([_TEXT_END, _THOUGHT_FOO, _TOOL_CALL])
    out = [e async for e in fr.events(Thought)]
    assert out == [_THOUGHT_FOO]


async def test_tool_call_selector_picks_tool_call_events_only() -> None:
    fr = _make_fluent_run([_TEXT_END, _TOOL_CALL, _THOUGHT_FOO])
    out = [e async for e in fr.events(ToolCall)]
    assert out == [_TOOL_CALL]


async def test_text_selector_picks_text_end_events_only() -> None:
    fr = _make_fluent_run([_TEXT_END, _THOUGHT_FOO, _TOOL_CALL])
    out = [e async for e in fr.events(Text)]
    assert out == [_TEXT_END]


async def test_where_chains_predicate_with_and() -> None:
    """``Thought.where(pred)`` keeps only events whose ``text`` matches."""
    fr = _make_fluent_run([_THOUGHT_FOO, _THOUGHT_BAR, _TEXT_END])
    selector = Thought.where(lambda t: "foo" in t.text)
    out = [e async for e in fr.events(selector)]
    assert out == [_THOUGHT_FOO]


async def test_where_chains_multiple_predicates_with_and() -> None:
    """Composed ``.where().where()`` AND-chains both predicates."""
    fr = _make_fluent_run([_THOUGHT_FOO, _THOUGHT_BAR])
    selector = Thought.where(lambda t: "foo" in t.text).where(lambda t: t.id == "th1")
    out = [e async for e in fr.events(selector)]
    assert out == [_THOUGHT_FOO]


async def test_map_replaces_transform_and_is_applied() -> None:
    """``ToolCall.map(fn)`` yields ``fn(tc)`` instead of the raw event."""
    fr = _make_fluent_run([_TEXT_END, _TOOL_CALL])
    selector = ToolCall.map(lambda tc: f"calling {tc.name}")
    out = [e async for e in fr.events(selector)]
    assert out == ["calling add"]


async def test_multiple_selectors_union_first_match_wins() -> None:
    """Two selectors → events matching either are yielded; first match transforms."""
    fr = _make_fluent_run([_THOUGHT_FOO, _TOOL_CALL, _TEXT_END, _THOUGHT_BAR])
    out = [
        e async for e in fr.events(
            Thought.map(lambda t: f"thought:{t.text}"),
            ToolCall.map(lambda tc: f"call:{tc.name}"),
        )
    ]
    assert out == ["thought:foo plan", "call:add", "thought:bar plan"]


async def test_no_selectors_yields_every_event() -> None:
    """Empty ``selectors`` preserves M1 behaviour: stream is unchanged."""
    fr = _make_fluent_run([_TEXT_END, _THOUGHT_FOO, _TOOL_CALL, _TURN_END])
    out = [e async for e in fr.events()]
    assert out == [_TEXT_END, _THOUGHT_FOO, _TOOL_CALL, _TURN_END]


async def test_remaining_selectors_match_their_event_types() -> None:
    """Smoke: ``ToolResult``, ``TurnEnd``, ``Done``, ``Error`` each pick their class."""
    fr = _make_fluent_run([_TEXT_END, _TOOL_RESULT, _TURN_END, _DONE, _ERROR])
    assert [e async for e in fr.events(ToolResult)] == [_TOOL_RESULT]

    fr = _make_fluent_run([_TEXT_END, _TOOL_RESULT, _TURN_END, _DONE, _ERROR])
    assert [e async for e in fr.events(TurnEnd)] == [_TURN_END]

    fr = _make_fluent_run([_TEXT_END, _TOOL_RESULT, _TURN_END, _DONE, _ERROR])
    assert [e async for e in fr.events(Done)] == [_DONE]

    fr = _make_fluent_run([_TEXT_END, _TOOL_RESULT, _TURN_END, _DONE, _ERROR])
    assert [e async for e in fr.events(Error)] == [_ERROR]


def test_selector_constructor_defaults_passthrough() -> None:
    """Bare ``Selector(EventClass)`` matches every event of that type."""
    selector: Selector[ThoughtEvent] = Selector(ThoughtEvent)
    assert selector.matches(_THOUGHT_FOO) is True
    assert selector.matches(_THOUGHT_BAR) is True
    assert selector.matches(_TEXT_END) is False
    assert selector.transform(_THOUGHT_FOO) is _THOUGHT_FOO
