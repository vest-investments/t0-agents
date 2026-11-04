"""Vectors for :mod:`t0_agents.stream.status` — the ported status policy.

Ported from the production suite that pinned this policy at its origin
(a consumer's StatusTranslator): originals re-emit first; ``Calling …``
never dedupes; ``Thinking…`` dedupes within a contiguous thought block
and is suppressed after a tool call until a terminal event re-arms it;
tool identity splits on the longest known server prefix.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from t0_agents import (
    AgentResult,
    DoneEvent,
    RunErrorEvent,
    Status,
    TextDeltaEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
    TurnEndEvent,
    split_tool_identity,
    synthesize_statuses,
)
from t0_agents.stream.events import AgentEvent

KNOWN_SERVERS = ("customers", "fred", "edgar", "massive", "mywallst")


async def _stream(events: Sequence[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def _translate(
    events: Sequence[AgentEvent],
    *,
    known_servers: Sequence[str] = KNOWN_SERVERS,
) -> list[AgentEvent | Status]:
    items: list[AgentEvent | Status] = []
    async for item in synthesize_statuses(_stream(events), known_servers=known_servers):
        items.append(item)
    return items


def _statuses(items: list[AgentEvent | Status]) -> list[Status]:
    return [item for item in items if isinstance(item, Status)]


def _tool_call(name: str, call_id: str = "c1") -> ToolCallEvent:
    return ToolCallEvent(id=call_id, name=name, arguments={})


def _done() -> DoneEvent:
    return DoneEvent(result=AgentResult(value="ok", turns=1))


def _run_error() -> RunErrorEvent:
    return RunErrorEvent(error_type="ValueError", message="boom")


class TestCallingHint:
    async def test_inserts_calling_after_tool_call(self) -> None:
        call = _tool_call("get_weather")
        items = await _translate([call])
        assert items == [
            call,
            Status(text="Calling get_weather…", server=None, tool="get_weather"),
        ]

    async def test_calling_carries_prefixed_tool_identity(self) -> None:
        items = await _translate([_tool_call("mywallst_get_portfolio")])
        assert _statuses(items) == [
            Status(
                text="Calling mywallst_get_portfolio…",
                server="mywallst",
                tool="get_portfolio",
            )
        ]

    async def test_does_not_dedupe_repeated_tool_calls(self) -> None:
        items = await _translate([_tool_call("alpha", "c1"), _tool_call("bravo", "c2")])
        assert len(items) == 4
        assert [s.text for s in _statuses(items)] == ["Calling alpha…", "Calling bravo…"]


class TestThinkingHint:
    async def test_inserts_thinking_after_thought(self) -> None:
        items = await _translate([ThoughtEvent(id="th1", text="hmm")])
        assert _statuses(items) == [Status(text="Thinking…")]

    async def test_inserts_thinking_after_thought_delta(self) -> None:
        items = await _translate([ThoughtDeltaEvent(id="th1", delta="hm")])
        assert _statuses(items) == [Status(text="Thinking…")]

    async def test_dedupes_thinking_within_contiguous_thought_block(self) -> None:
        events: list[AgentEvent] = [
            ThoughtDeltaEvent(id="th1", delta="a"),
            ThoughtDeltaEvent(id="th1", delta="b"),
            ThoughtDeltaEvent(id="th1", delta="c"),
            ThoughtEvent(id="th1", text="abc"),
        ]
        items = await _translate(events)
        assert _statuses(items) == [Status(text="Thinking…")]

    async def test_thinking_resurfaces_after_text_breaks_the_block(self) -> None:
        events: list[AgentEvent] = [
            ThoughtDeltaEvent(id="th1", delta="a"),
            TextDeltaEvent(id="t1", delta="hello"),
            ThoughtDeltaEvent(id="th2", delta="b"),
        ]
        items = await _translate(events)
        assert [s.text for s in _statuses(items)] == ["Thinking…", "Thinking…"]

    async def test_thinking_suppressed_after_tool_call_within_turn(self) -> None:
        events: list[AgentEvent] = [
            ThoughtDeltaEvent(id="th1", delta="a"),
            _tool_call("get_weather"),
            ThoughtDeltaEvent(id="th2", delta="b"),
        ]
        items = await _translate(events)
        assert [s.text for s in _statuses(items)] == ["Thinking…", "Calling get_weather…"]

    async def test_multiple_thoughts_after_tool_all_suppressed(self) -> None:
        events: list[AgentEvent] = [
            _tool_call("get_weather"),
            ThoughtDeltaEvent(id="th1", delta="a"),
            ThoughtEvent(id="th1", text="a"),
            ThoughtDeltaEvent(id="th2", delta="b"),
        ]
        items = await _translate(events)
        assert [s.text for s in _statuses(items)] == ["Calling get_weather…"]

    @pytest.mark.parametrize("terminal", [_done(), _run_error()], ids=["done", "run_error"])
    async def test_terminal_re_arms_thinking(self, terminal: AgentEvent) -> None:
        events: list[AgentEvent] = [
            ThoughtDeltaEvent(id="th1", delta="a"),
            _tool_call("get_weather"),
            ThoughtDeltaEvent(id="th2", delta="b"),
            terminal,
            ThoughtDeltaEvent(id="th3", delta="c"),
        ]
        items = await _translate(events)
        assert [s.text for s in _statuses(items)] == [
            "Thinking…",
            "Calling get_weather…",
            "Thinking…",
        ]

    async def test_turn_end_does_not_re_arm_the_tool_suppression(self) -> None:
        events: list[AgentEvent] = [
            _tool_call("get_weather"),
            TurnEndEvent(turn=1),
            ThoughtDeltaEvent(id="th1", delta="a"),
        ]
        items = await _translate(events)
        assert [s.text for s in _statuses(items)] == ["Calling get_weather…"]


class TestOrderingAndPassthrough:
    async def test_originals_re_emit_first_and_verbatim(self) -> None:
        events: list[AgentEvent] = [
            ThoughtDeltaEvent(id="th1", delta="a"),
            _tool_call("get_weather"),
            _done(),
        ]
        items = await _translate(events)
        passthrough = [item for item in items if not isinstance(item, Status)]
        assert passthrough == events
        assert isinstance(items[0], ThoughtDeltaEvent)
        assert isinstance(items[1], Status)


class TestSplitToolIdentity:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("customers_place_trade", ("customers", "place_trade")),
            ("fred_get_macro_snapshot", ("fred", "get_macro_snapshot")),
            ("edgar_get_filing", ("edgar", "get_filing")),
            ("massive_get_quote", ("massive", "get_quote")),
            ("mywallst_get_portfolio", ("mywallst", "get_portfolio")),
            ("place_trade", (None, "place_trade")),
            ("retrieve_user_profile", (None, "retrieve_user_profile")),
        ],
    )
    def test_split_vectors(self, name: str, expected: tuple[str | None, str]) -> None:
        assert split_tool_identity(name, KNOWN_SERVERS) == expected

    @pytest.mark.parametrize(
        "servers", [("fred", "fred_api"), ("fred_api", "fred")], ids=["short-first", "long-first"]
    )
    def test_prefers_longest_prefix(self, servers: tuple[str, ...]) -> None:
        assert split_tool_identity("fred_api_get_x", servers) == ("fred_api", "get_x")

    def test_unknown_server_stays_none(self) -> None:
        assert split_tool_identity("bare_tool", ()) == (None, "bare_tool")
