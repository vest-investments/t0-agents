"""Per-stage behaviour of the wire projection stages.

``project`` is the stateless event→frame translation (including the
frozen terminal-error mapping); the stateful stages get their own test
groups as they land. The end-to-end public projection is pinned
separately by the ported vest-ai golden in ``test_public_golden.py``.
"""

from __future__ import annotations

import asyncio
import json
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
    EMPTY_TURN_RETRY_TEXT,
    FALLBACK_BLOCK_ID,
    DoneFrame,
    ElicitationRequestFrame,
    ElicitationResolvedFrame,
    ErrorFrame,
    InitFrame,
    PublicEvent,
    StatusFrame,
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
from t0_agents.wire.stages import (
    MAX_ARG_BYTES,
    as_terminal_safe,
    project,
    public_allowlist,
    with_status,
    with_text_fallback,
)


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


def _tool_call(name: str) -> ToolCallFrame:
    return ToolCallFrame(
        id="t1", name=name, arguments_json="{}", truncated_from=None, raw_event_json="{}"
    )


def _done(
    *, text: str = "", ids: tuple[str, ...] = ("inf-1",), source: str = "text"
) -> DoneFrame:
    assert source in ("text", "thought")
    return DoneFrame(
        turns=1,
        response_source="thought" if source == "thought" else "text",
        response_text=text,
        inference_ids=ids,
    )


async def _statuses(
    frames: Iterable[WireFrame], *, known_servers: tuple[str, ...] = ()
) -> list[WireFrame]:
    return [
        f
        async for f in with_status(_stream_frames(frames), known_servers=known_servers)
    ]


class TestWithStatus:
    async def test_tool_call_hint_carries_the_split_identity(self) -> None:
        out = await _statuses([_tool_call("fred_get_macro")], known_servers=("fred",))
        assert out == [
            _tool_call("fred_get_macro"),
            StatusFrame(text="Calling fred_get_macro…", server="fred", tool="get_macro"),
        ]

    async def test_unprefixed_tool_name_has_no_server(self) -> None:
        [_, status] = await _statuses([_tool_call("get_weather")])
        assert status == StatusFrame(
            text="Calling get_weather…", server=None, tool="get_weather"
        )

    async def test_every_tool_call_fires_its_own_hint(self) -> None:
        out = await _statuses([_tool_call("a"), _tool_call("b")])
        assert [f.text for f in out if isinstance(f, StatusFrame)] == [
            "Calling a…",
            "Calling b…",
        ]

    async def test_thinking_fires_once_per_contiguous_thought_block(self) -> None:
        out = await _statuses(
            [
                ThoughtDeltaFrame(id="th1", delta="a"),
                ThoughtDeltaFrame(id="th1", delta="b"),
            ]
        )
        assert [f for f in out if isinstance(f, StatusFrame)] == [
            StatusFrame(text="Thinking…")
        ]

    async def test_a_text_delta_ends_the_thought_block(self) -> None:
        out = await _statuses(
            [
                ThoughtDeltaFrame(id="th1", delta="a"),
                TextDeltaFrame(block_id="b1", delta="hi"),
                ThoughtDeltaFrame(id="th2", delta="b"),
            ]
        )
        assert [f for f in out if isinstance(f, StatusFrame)] == [
            StatusFrame(text="Thinking…"),
            StatusFrame(text="Thinking…"),
        ]

    async def test_a_tool_call_suppresses_thinking_for_the_rest_of_the_run(self) -> None:
        out = await _statuses(
            [_tool_call("get_weather"), ThoughtDeltaFrame(id="th1", delta="a")]
        )
        statuses = [f for f in out if isinstance(f, StatusFrame)]
        assert statuses == [
            StatusFrame(text="Calling get_weather…", server=None, tool="get_weather")
        ]

    async def test_the_run_terminator_rearms_both_flags(self) -> None:
        out = await _statuses(
            [
                _tool_call("get_weather"),
                _done(text="hi"),
                ThoughtDeltaFrame(id="th1", delta="next run"),
            ]
        )
        assert [f.text for f in out if isinstance(f, StatusFrame)] == [
            "Calling get_weather…",
            "Thinking…",
        ]

    async def test_an_error_frame_also_rearms(self) -> None:
        out = await _statuses(
            [
                _tool_call("get_weather"),
                ErrorFrame(message="boom", detail={}),
                ThoughtDeltaFrame(id="th1", delta="next run"),
            ]
        )
        assert [f.text for f in out if isinstance(f, StatusFrame)] == [
            "Calling get_weather…",
            "Thinking…",
        ]

    async def test_originals_flow_through_first_and_unchanged(self) -> None:
        source = [
            ThoughtDeltaFrame(id="th1", delta="a"),
            TextDeltaFrame(block_id="b1", delta="hi"),
            _done(text="hi"),
        ]
        out = await _statuses(source)
        assert [f for f in out if not isinstance(f, StatusFrame)] == source


async def _fallback(frames: Iterable[WireFrame]) -> list[WireFrame]:
    return [f async for f in with_text_fallback(_stream_frames(frames))]


class TestWithTextFallback:
    async def test_a_texty_run_passes_through_untouched(self) -> None:
        source = [
            TextDeltaFrame(block_id="b1", delta="hi"),
            TextEndFrame(block_id="b1"),
            _done(text="hi there"),
        ]
        assert await _fallback(source) == source

    async def test_reasoning_fallback_synthesises_the_answer(self) -> None:
        out = await _fallback(
            [
                ThoughtDeltaFrame(id="th1", delta="hmm"),
                _done(text="The answer.", ids=("inf-1",), source="thought"),
            ]
        )
        assert out == [
            ThoughtDeltaFrame(id="th1", delta="hmm"),
            TextDeltaFrame(block_id="inf-1", delta="The answer."),
            TextEndFrame(block_id="inf-1"),
            _done(text="The answer.", ids=("inf-1",), source="thought"),
        ]

    async def test_the_belt_covers_a_blank_response(self) -> None:
        out = await _fallback([_done(text="   \n\t  ", ids=("inf-9",))])
        assert out[:2] == [
            TextDeltaFrame(block_id="inf-9", delta=EMPTY_TURN_RETRY_TEXT),
            TextEndFrame(block_id="inf-9"),
        ]

    async def test_no_inference_ids_uses_the_fallback_block_id(self) -> None:
        out = await _fallback([_done(text="hi", ids=())])
        assert out[0] == TextDeltaFrame(block_id=FALLBACK_BLOCK_ID, delta="hi")

    async def test_blank_only_deltas_do_not_count_as_text(self) -> None:
        out = await _fallback(
            [
                TextDeltaFrame(block_id="b1", delta=""),
                TextEndFrame(block_id="b1"),
                _done(text="answer", ids=("inf-1",)),
            ]
        )
        assert TextDeltaFrame(block_id="inf-1", delta="answer") in out

    async def test_saw_text_rearms_per_run_on_a_long_lived_stream(self) -> None:
        """Deliberate deviation from vest-ai's projector, which never
        re-arms ``_saw_text`` and so suppresses the reasoning fallback
        for every run after the first texty one on the same connection.
        The library re-arms per run, consistent with the status
        translator's terminator semantics."""
        out = await _fallback(
            [
                TextDeltaFrame(block_id="b1", delta="run one"),
                _done(text="run one"),
                ThoughtDeltaFrame(id="th1", delta="run two thinks"),
                _done(text="run two answer", ids=("inf-2",), source="thought"),
            ]
        )
        assert TextDeltaFrame(block_id="inf-2", delta="run two answer") in out

    async def test_an_elicitation_prompt_does_not_suppress_the_fallback(self) -> None:
        out = await _fallback(
            [
                ElicitationRequestFrame(
                    id="e1", message="Proceed?", response_type=None, schema_json="{}"
                ),
                _done(text="done after prompt", ids=("inf-3",)),
            ]
        )
        assert TextDeltaFrame(block_id="inf-3", delta="done after prompt") in out

    async def test_a_terminal_error_rearms_for_the_next_run(self) -> None:
        terminal = ErrorFrame(
            message="boom", detail={"exc_class": "TimeoutError"}
        )
        out = await _fallback(
            [
                TextDeltaFrame(block_id="b1", delta="run one text"),
                terminal,
                ThoughtDeltaFrame(id="th1", delta="run two thinks"),
                _done(text="run two answer", ids=("inf-2",), source="thought"),
            ]
        )
        assert TextDeltaFrame(block_id="inf-2", delta="run two answer") in out

    async def test_a_recoverable_error_does_not_reset_mid_run(self) -> None:
        """A recoverable ErrorEvent passthrough (no exc_class) must not
        re-arm saw_text: the run continues, and a reset would make the
        fallback duplicate the text that already streamed."""
        out = await _fallback(
            [
                TextDeltaFrame(block_id="b1", delta="the answer"),
                ErrorFrame(message="transient", detail={}),
                _done(text="the answer", ids=("inf-1",)),
            ]
        )
        assert out == [
            TextDeltaFrame(block_id="b1", delta="the answer"),
            ErrorFrame(message="transient", detail={}),
            _done(text="the answer", ids=("inf-1",)),
        ]

    async def test_an_init_frame_rearms_for_the_new_session(self) -> None:
        out = await _fallback(
            [
                TextDeltaFrame(block_id="b1", delta="old session text"),
                InitFrame(conversation_id="conv-2"),
                _done(text="new session answer", ids=("inf-5",), source="thought"),
            ]
        )
        assert TextDeltaFrame(block_id="inf-5", delta="new session answer") in out


async def _public(frames: Iterable[WireFrame]) -> list[PublicEvent]:
    return [event async for event in public_allowlist(_stream_frames(frames))]


class TestPublicAllowlist:
    async def test_init_and_error_and_status_payloads_carry_no_type_field(self) -> None:
        out = await _public(
            [
                InitFrame(conversation_id="conv-1"),
                ErrorFrame(message="boom", detail={"k": "v"}),
                StatusFrame(text="Thinking…"),
            ]
        )
        assert out == [
            PublicEvent("init", {"conversation_id": "conv-1"}),
            PublicEvent("error", {"message": "boom", "detail": {"k": "v"}}),
            PublicEvent("status", {"text": "Thinking…", "server": None, "tool": None}),
        ]

    async def test_text_frames_project_as_chunk_and_end(self) -> None:
        out = await _public(
            [
                TextDeltaFrame(block_id="b1", delta="Hola — ñandú"),
                TextEndFrame(block_id="b1"),
            ]
        )
        assert out == [
            PublicEvent(
                "chunk", {"type": "chunk", "content": "Hola — ñandú", "message_id": "b1"}
            ),
            PublicEvent("end", {"type": "end", "id": "b1"}),
        ]

    async def test_done_id_is_the_last_inference_id(self) -> None:
        [event] = await _public([_done(text="hi", ids=("inf-1", "inf-2"))])
        assert event == PublicEvent(
            "done",
            {"type": "done", "id": "inf-2", "turns": 1, "response_source": "text"},
        )

    async def test_done_without_inference_ids_uses_the_fallback_id(self) -> None:
        [event] = await _public([_done(text="hi", ids=())])
        assert event.data["id"] == FALLBACK_BLOCK_ID

    async def test_elicitation_request_reparses_the_schema(self) -> None:
        schema = {"type": "object", "properties": {"value": {"type": "boolean"}}}
        [event] = await _public(
            [
                ElicitationRequestFrame(
                    id="e1",
                    message="Confirm?",
                    response_type="bool",
                    schema_json=json.dumps(schema),
                )
            ]
        )
        assert event == PublicEvent(
            "elicitation_request",
            {
                "type": "elicitation_request",
                "id": "e1",
                "message": "Confirm?",
                "response_type": "bool",
                "schema": schema,
            },
        )

    async def test_a_malformed_schema_degrades_to_an_empty_object(self) -> None:
        [event] = await _public(
            [
                ElicitationRequestFrame(
                    id="e1", message="?", response_type=None, schema_json="{not json"
                )
            ]
        )
        assert event.data["schema"] == {}

    async def test_elicitation_resolved_payload(self) -> None:
        [event] = await _public([ElicitationResolvedFrame(id="e1", action="accept")])
        assert event == PublicEvent(
            "elicitation_resolved",
            {"type": "elicitation_resolved", "id": "e1", "action": "accept"},
        )

    async def test_the_privacy_guard_drops_tool_thought_and_turn_frames(self) -> None:
        out = await _public(
            [
                _tool_call("get_weather"),
                ToolCallDeltaFrame(id="t1", raw_name_delta="x", raw_arguments_delta=""),
                ToolResultFrame(
                    id="t1",
                    name="get_weather",
                    result="72F",
                    is_error=False,
                    truncated_from=None,
                    raw_event_json="{}",
                ),
                ToolProgressFrame(id="t1", progress=1.0, total=None, message=None),
                ThoughtFrame(id="th1", text="secret reasoning"),
                ThoughtDeltaFrame(id="th1", delta="secret"),
                TurnEndFrame(turn=1),
            ]
        )
        assert out == []
