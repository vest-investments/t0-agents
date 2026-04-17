"""Unit tests for :class:`t0_agents.cli.tui.TUIRenderer`."""

from __future__ import annotations

from collections.abc import AsyncIterator

from rich.console import Console

from t0_agents.cli.tui import TUIRenderer
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)


def _make_console() -> Console:
    """Force plain-text output with a fixed width for snapshot stability."""
    return Console(
        file=None,
        force_terminal=False,
        color_system=None,
        width=80,
        legacy_windows=False,
        highlight=False,
    )


async def _stream(events: list[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def test_text_deltas_accumulate() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        TextDeltaEvent(id="t-1", delta="Hel"),
        TextDeltaEvent(id="t-1", delta="lo "),
        TextDeltaEvent(id="t-1", delta="world"),
        TextEndEvent(id="t-1"),
        DoneEvent(result=AgentResult(response="Hello world")),
    ]
    await renderer.consume(_stream(events))
    assert "Hello world" in renderer.render_text()


async def test_tool_call_status_rendered() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        ToolCallEvent(id="tc-1", name="add", arguments={"a": 1, "b": 2}),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    rendered = renderer.render_text()
    assert "add" in rendered
    assert "1" in rendered and "2" in rendered
    assert "🔧" in rendered


async def test_tool_result_success_marker() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        ToolCallEvent(id="tc-1", name="add", arguments={"a": 1, "b": 2}),
        ToolResultEvent(id="tc-1", name="add", result="3", is_error=False),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    assert "✓" in renderer.render_text()


async def test_tool_result_error_marker() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        ToolCallEvent(id="tc-1", name="bomb", arguments={}),
        ToolResultEvent(id="tc-1", name="bomb", result="boom", is_error=True),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    assert "❌" in renderer.render_text()


async def test_done_event_captures_episode_id() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    result = AgentResult(response="done", episode_id="episode-xyz")
    events: list[AgentEvent] = [DoneEvent(result=result)]
    captured = await renderer.consume(_stream(events))
    assert captured is not None
    assert captured.episode_id == "episode-xyz"
    assert renderer.last_episode_id == "episode-xyz"
    assert renderer.last_result is result


async def test_error_event_renders_red_marker() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        ErrorEvent(message="invalid chunk"),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    assert "! invalid chunk" in renderer.render_text()


async def test_thought_hidden_by_default() -> None:
    renderer = TUIRenderer("assistant", console=_make_console(), verbose=False)
    events: list[AgentEvent] = [
        ThoughtEvent(id="th-1", text="thinking hard"),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    assert "thinking hard" not in renderer.render_text()


async def test_thought_visible_when_verbose() -> None:
    renderer = TUIRenderer("assistant", console=_make_console(), verbose=True)
    events: list[AgentEvent] = [
        ThoughtDeltaEvent(id="th-1", delta="let me"),
        ThoughtDeltaEvent(id="th-1", delta=" think"),
        ThoughtEvent(id="th-1", text="let me think"),
        DoneEvent(result=AgentResult(response="")),
    ]
    await renderer.consume(_stream(events))
    assert "let me think" in renderer.render_text()


async def test_reset_clears_between_turns() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    first_events: list[AgentEvent] = [
        TextDeltaEvent(id="t-1", delta="first turn"),
        DoneEvent(result=AgentResult(response="first turn", episode_id="ep-1")),
    ]
    await renderer.consume(_stream(first_events))
    assert "first turn" in renderer.render_text()

    second_events: list[AgentEvent] = [
        TextDeltaEvent(id="t-2", delta="second turn"),
        DoneEvent(result=AgentResult(response="second turn", episode_id="ep-1")),
    ]
    await renderer.consume(_stream(second_events))
    rendered = renderer.render_text()
    assert "second turn" in rendered
    assert "first turn" not in rendered


async def test_turn_end_is_accepted_no_op() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [
        TurnEndEvent(turn=1),
        TextDeltaEvent(id="t-1", delta="hi"),
        TurnEndEvent(turn=2),
        DoneEvent(result=AgentResult(response="hi")),
    ]
    await renderer.consume(_stream(events))
    assert "hi" in renderer.render_text()


async def test_empty_stream_renders_placeholder() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    events: list[AgentEvent] = [DoneEvent(result=AgentResult(response=""))]
    await renderer.consume(_stream(events))
    assert "…" in renderer.render_text()


def test_pause_and_resume_with_no_live_is_noop() -> None:
    renderer = TUIRenderer("assistant", console=_make_console())
    renderer.pause()
    renderer.resume()
