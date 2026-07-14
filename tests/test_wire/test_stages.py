"""Per-stage behaviour of the wire projection stages.

``project`` is the stateless event→frame translation (including the
frozen terminal-error mapping); the stateful stages get their own test
groups as they land. The end-to-end public projection is pinned
separately by the ported vest-ai golden in ``test_public_golden.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

import pytest

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    RunErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.testing import ScriptedAgent
from t0_agents.wire import (
    DoneFrame,
    ErrorFrame,
    TextDeltaFrame,
    TextEndFrame,
    ThoughtDeltaFrame,
    ThoughtFrame,
    ToolCallDeltaFrame,
    ToolCallFrame,
    ToolProgressFrame,
    ToolResultFrame,
    TurnEndFrame,
    WireFrame,
)
from t0_agents.wire.stages import MAX_ARG_BYTES, as_terminal_safe, project


async def _stream(events: Iterable[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def _project(events: Iterable[AgentEvent]) -> list[WireFrame]:
    return [frame async for frame in project(_stream(events))]


class TestProjectMapping:
    async def test_text_and_thought_and_turn_arms(self) -> None:
        frames = await _project(
            [
                TextDeltaEvent(id="b1", delta="hi"),
                TextEndEvent(id="b1"),
                ThoughtDeltaEvent(id="th1", delta="hmm"),
                ThoughtEvent(id="th1", text="hmm, considered"),
                TurnEndEvent(turn=1),
            ]
        )
        assert frames == [
            TextDeltaFrame(block_id="b1", delta="hi"),
            TextEndFrame(block_id="b1"),
            ThoughtDeltaFrame(id="th1", delta="hmm"),
            ThoughtFrame(id="th1", text="hmm, considered"),
            TurnEndFrame(turn=1),
        ]

    async def test_tool_call_serialises_and_keeps_unicode(self) -> None:
        [frame] = await _project(
            [ToolCallEvent(id="t1", name="get_weather", arguments={"city": "Añasco"})]
        )
        assert isinstance(frame, ToolCallFrame)
        assert frame.arguments_json == '{"city": "Añasco"}'
        assert frame.truncated_from is None
        assert '"type": "tool_call"' in frame.raw_event_json

    async def test_tool_call_delta_passthrough(self) -> None:
        [frame] = await _project(
            [ToolCallDeltaEvent(id="t1", raw_name_delta="get_", raw_arguments_delta="{")]
        )
        assert frame == ToolCallDeltaFrame(
            id="t1", raw_name_delta="get_", raw_arguments_delta="{"
        )

    async def test_tool_result_truncates_on_byte_length(self) -> None:
        long_result = "x" * (MAX_ARG_BYTES + 100)
        [frame] = await _project(
            [ToolResultEvent(id="t1", name="search", result=long_result)]
        )
        assert isinstance(frame, ToolResultFrame)
        assert frame.result.endswith("…")
        assert len(frame.result.encode("utf-8")) <= MAX_ARG_BYTES + len("…".encode())
        assert frame.truncated_from == len(long_result)

    async def test_value_at_exactly_the_cap_is_left_alone(self) -> None:
        exact = "y" * MAX_ARG_BYTES
        [frame] = await _project([ToolResultEvent(id="t1", name="search", result=exact)])
        assert isinstance(frame, ToolResultFrame)
        assert frame.result == exact
        assert frame.truncated_from is None

    async def test_tool_progress_maps_field_for_field(self) -> None:
        [frame] = await _project(
            [ToolProgressEvent(id="t1", progress=3.0, total=10.0, message="fetching")]
        )
        assert frame == ToolProgressFrame(
            id="t1", progress=3.0, total=10.0, message="fetching"
        )

    async def test_done_unwraps_the_result(self) -> None:
        result: AgentResult[str] = AgentResult(
            value="The answer is 42.",
            inference_ids=["inf-1", "inf-2"],
            turns=2,
            response_source="thought",
        )
        [frame] = await _project([DoneEvent(result=result)])
        assert frame == DoneFrame(
            turns=2,
            response_source="thought",
            response_text="The answer is 42.",
            inference_ids=("inf-1", "inf-2"),
        )

    async def test_recoverable_error_passes_through(self) -> None:
        [frame] = await _project([ErrorEvent(message="boom", detail={"k": "v"})])
        assert frame == ErrorFrame(message="boom", detail={"k": "v"})


class TestProjectRunError:
    async def test_generic_run_error_carries_exc_class(self) -> None:
        [frame] = await _project(
            [RunErrorEvent(error_type="TimeoutError", message="TimeoutError")]
        )
        assert frame == ErrorFrame(
            message="TimeoutError", detail={"exc_class": "TimeoutError"}
        )

    async def test_max_turns_maps_to_the_frozen_shape(self) -> None:
        [frame] = await _project(
            [
                RunErrorEvent(
                    error_type="MaxTurnsExceededError",
                    message="Agent loop exceeded 20 turns",
                    detail={"max_turns": 20},
                )
            ]
        )
        assert frame == ErrorFrame(
            message="max_turns_exceeded",
            detail={"max_turns": 20, "exc_class": "MaxTurnsExceededError"},
        )

    async def test_the_exception_object_never_reaches_the_wire(self) -> None:
        event = RunErrorEvent(
            error_type="ValueError",
            message="bad",
            exception=ValueError("bad"),
        )
        [frame] = await _project([event])
        assert frame == ErrorFrame(message="bad", detail={"exc_class": "ValueError"})


async def _raising_frames(
    frames: Iterable[WireFrame], exc: BaseException
) -> AsyncIterator[WireFrame]:
    for frame in frames:
        yield frame
    raise exc


class TestAsTerminalSafe:
    async def test_a_clean_stream_passes_through_unchanged(self) -> None:
        source = [TextDeltaFrame(block_id="b1", delta="hi"), TurnEndFrame(turn=1)]
        out = [f async for f in as_terminal_safe(_stream_frames(source))]
        assert out == source

    async def test_max_turns_replays_frames_then_ends_with_the_frozen_error(self) -> None:
        prior = TextDeltaFrame(block_id="b1", delta="partial")
        out = [
            f
            async for f in as_terminal_safe(
                _raising_frames([prior], MaxTurnsExceededError(20))
            )
        ]
        assert out == [
            prior,
            ErrorFrame(
                message="max_turns_exceeded",
                detail={"max_turns": 20, "exc_class": "MaxTurnsExceededError"},
            ),
        ]

    async def test_a_bare_exception_falls_back_to_the_class_name(self) -> None:
        out = [f async for f in as_terminal_safe(_raising_frames([], TimeoutError()))]
        assert out == [
            ErrorFrame(message="TimeoutError", detail={"exc_class": "TimeoutError"})
        ]

    async def test_the_text_attribute_wins_over_str(self) -> None:
        class FakeGatewayError(Exception):
            text = "provider said: rate limited"

        out = [
            f
            async for f in as_terminal_safe(
                _raising_frames([], FakeGatewayError("generic status"))
            )
        ]
        assert out == [
            ErrorFrame(
                message="provider said: rate limited",
                detail={"exc_class": "FakeGatewayError"},
            )
        ]

    async def test_cancellation_propagates(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            async for _ in as_terminal_safe(
                _raising_frames([], asyncio.CancelledError())
            ):
                pass

    async def test_both_failure_paths_produce_identical_wire_output(self) -> None:
        """A raised exception and its RunErrorEvent must be indistinguishable."""

        def scripted() -> ScriptedAgent:
            return ScriptedAgent(
                events=[TextDeltaEvent(id="b1", delta="partial")],
                raises=MaxTurnsExceededError(20),
            )

        raising_run = scripted().invoke("q")
        via_raise = [
            f
            async for f in as_terminal_safe(project(raising_run.events()))
        ]
        event_run = scripted().invoke("q")
        via_event = [
            f async for f in project(event_run.events(on_error="event"))
        ]
        assert via_raise == via_event


async def _stream_frames(frames: Iterable[WireFrame]) -> AsyncIterator[WireFrame]:
    for frame in frames:
        yield frame
