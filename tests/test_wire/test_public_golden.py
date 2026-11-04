"""Public-wire golden — a production byte-compat gate, ported at dict level.

For every event variant and every stateful fallback sequence, the
public projection ``project → with_text_fallback → public_allowlist``
must reproduce the production wire shape exactly: the frozen event
names ``init, chunk, end, status, error, done, elicitation_request,
elicitation_resolved`` and their frozen payload dicts. The upstream
gate freezes UTF-8 SSE bytes; byte serialisation is the consumer's
transport (``json.dumps(..., ensure_ascii=False)`` on these dicts), so
the dict level is the exact library-owned slice of that contract.

``_GoldenMapper`` is a **frozen snapshot** ported verbatim from the
production gate. If this test fails, fix the stages — NEVER the
snapshot. The one sanctioned exception is a deliberate revision of the
contract itself, recorded here so the snapshot's history stays honest:
a recoverable ``ErrorEvent`` is internal telemetry and produces **no**
public event — the public ``error`` is terminal — so the mapper drops
it and the former ``error`` fixture is ``dropped_recoverable_error``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.testing import ScriptedAgent
from t0_agents.wire import (
    ElicitationRequestFrame,
    ElicitationResolvedFrame,
    ErrorFrame,
    PublicEvent,
    as_terminal_safe,
    project,
    public_allowlist,
    with_text_fallback,
)

_EMPTY_TURN_RETRY_TEXT = "I couldn't complete that — please try again."
"""Frozen copy of the production retry copy — kept as an independent
literal so a drift in ``wire.EMPTY_TURN_RETRY_TEXT`` fails this gate
rather than passing silently."""

_FALLBACK_BLOCK_ID = "thought-fallback"
"""Frozen copy of the production fallback block id — independent
literal for the same reason."""


class _GoldenMapper:
    """Frozen snapshot of the canonical public wire shape.

    Ported verbatim (dict level) from a consumer's byte-compat gate:
    tracks whether a non-blank ``TextDeltaEvent`` was emitted during
    the turn so it can decide, on ``DoneEvent``, what to synthesise
    before ``done`` — the reasoning fallback ``chunk`` + ``end`` from
    the final response, or the never-blank belt's retry pair.
    """

    def __init__(self) -> None:
        self._saw_text: bool = False

    def map(self, event: AgentEvent) -> list[PublicEvent]:
        if isinstance(event, TextDeltaEvent):
            if event.delta.strip():
                self._saw_text = True
            return [
                PublicEvent(
                    "chunk",
                    {
                        "type": "chunk",
                        "content": event.delta,
                        "message_id": event.id,
                    },
                )
            ]
        if isinstance(event, TextEndEvent):
            return [PublicEvent("end", {"type": "end", "id": event.id})]
        if isinstance(event, ErrorEvent):
            return []
        if isinstance(event, DoneEvent):
            return self._done_frames(event)
        return []

    def _done_frames(self, event: DoneEvent) -> list[PublicEvent]:
        frames = self._maybe_fallback_frames(event)
        done_id = self._fallback_message_id(event)
        frames.append(
            PublicEvent(
                "done",
                {
                    "type": "done",
                    "id": done_id,
                    "turns": event.result.turns,
                    "response_source": event.result.response_source,
                },
            )
        )
        return frames

    def _maybe_fallback_frames(self, event: DoneEvent) -> list[PublicEvent]:
        if self._saw_text:
            return []
        text = event.result.response
        body = text if text.strip() else _EMPTY_TURN_RETRY_TEXT
        message_id = self._fallback_message_id(event)
        return [
            PublicEvent(
                "chunk",
                {"type": "chunk", "content": body, "message_id": message_id},
            ),
            PublicEvent("end", {"type": "end", "id": message_id}),
        ]

    @staticmethod
    def _fallback_message_id(event: DoneEvent) -> str:
        if event.result.inference_ids:
            return event.result.inference_ids[-1]
        return _FALLBACK_BLOCK_ID


def _collect_golden(events: Sequence[AgentEvent]) -> list[PublicEvent]:
    golden = _GoldenMapper()
    out: list[PublicEvent] = []
    for event in events:
        out.extend(golden.map(event))
    return out


async def _event_stream(events: Sequence[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def _collect_public(events: Sequence[AgentEvent]) -> list[PublicEvent]:
    stream = public_allowlist(with_text_fallback(project(_event_stream(events))))
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Fixtures — ported verbatim from the production gate.
# ---------------------------------------------------------------------------


_PER_EVENT_FIXTURES: list[tuple[str, list[AgentEvent]]] = [
    ("text_delta_normal", [TextDeltaEvent(id="b1", delta="Hola — ñandú")]),
    ("text_delta_empty", [TextDeltaEvent(id="b1", delta="")]),
    ("text_end", [TextEndEvent(id="b1")]),
    (
        "done_text_not_seen_with_response",
        [
            DoneEvent(
                result=AgentResult(
                    response="The answer is 42.",
                    turns=2,
                    inference_ids=["inf-99"],
                    response_source="thought",
                )
            )
        ],
    ),
    (
        "done_text_not_seen_empty_response",
        [
            DoneEvent(
                result=AgentResult(
                    response="",
                    turns=1,
                    inference_ids=["inf-99"],
                    response_source="text",
                )
            )
        ],
    ),
    (
        "done_text_not_seen_whitespace_response",
        [
            DoneEvent(
                result=AgentResult(
                    response="   \n\t  ",
                    turns=1,
                    inference_ids=["inf"],
                    response_source="thought",
                )
            )
        ],
    ),
    (
        "done_no_inference_ids_uses_fallback_id",
        [
            DoneEvent(
                result=AgentResult(
                    response="hi",
                    turns=1,
                    inference_ids=[],
                    response_source="thought",
                )
            )
        ],
    ),
    (
        "dropped_tool_call",
        [ToolCallEvent(id="t", name="get_weather", arguments={"city": "NYC"})],
    ),
    (
        "dropped_tool_call_delta",
        [ToolCallDeltaEvent(id="t", raw_arguments_delta="x")],
    ),
    ("dropped_tool_result", [ToolResultEvent(id="t", name="t", result="ok")]),
    ("dropped_thought", [ThoughtEvent(id="th", text="x")]),
    ("dropped_thought_delta", [ThoughtDeltaEvent(id="th", delta="x")]),
    ("dropped_turn_end", [TurnEndEvent(turn=1)]),
    (
        "dropped_recoverable_error",
        [ErrorEvent(message="boom", detail={"k": "v"})],
    ),
]


_PER_SEQUENCE_FIXTURES: list[tuple[str, list[AgentEvent]]] = [
    (
        "seq_text_then_done_no_fallback",
        [
            TextDeltaEvent(id="b1", delta="hi"),
            TextEndEvent(id="b1"),
            DoneEvent(
                result=AgentResult(
                    response="hi there",
                    turns=1,
                    inference_ids=["inf-99"],
                    response_source="text",
                )
            ),
        ],
    ),
    (
        "seq_thought_then_done_fallback",
        [
            ThoughtDeltaEvent(id="th", delta="reasoning"),
            ThoughtEvent(id="th", text="reasoning..."),
            DoneEvent(
                result=AgentResult(
                    response="The answer.",
                    turns=2,
                    inference_ids=["inf-1"],
                    response_source="thought",
                )
            ),
        ],
    ),
    (
        "seq_empty_textdelta_then_done_fallback",
        [
            TextDeltaEvent(id="b1", delta=""),
            TextEndEvent(id="b1"),
            DoneEvent(
                result=AgentResult(
                    response="answer",
                    turns=1,
                    inference_ids=["inf"],
                    response_source="thought",
                )
            ),
        ],
    ),
    (
        "seq_empty_textdelta_then_empty_done_retry_belt",
        [
            TextDeltaEvent(id="b1", delta=""),
            TextEndEvent(id="b1"),
            DoneEvent(
                result=AgentResult(
                    response="",
                    turns=1,
                    inference_ids=["inf"],
                    response_source="thought",
                )
            ),
        ],
    ),
    (
        "seq_whitespace_textdelta_then_empty_done_retry_belt",
        [
            TextDeltaEvent(id="b1", delta="  \t "),
            TextEndEvent(id="b1"),
            DoneEvent(
                result=AgentResult(
                    response="",
                    turns=1,
                    inference_ids=["inf"],
                    response_source="thought",
                )
            ),
        ],
    ),
    (
        "seq_text_recoverable_error_text_done",
        [
            TextDeltaEvent(id="b1", delta="Let me check"),
            TextEndEvent(id="b1"),
            ErrorEvent(
                message="invalid tool-call JSON arguments: Extra data",
                detail={"tool_call_id": "t1", "name": "search"},
            ),
            TextDeltaEvent(id="b2", delta="Here is the answer."),
            TextEndEvent(id="b2"),
            DoneEvent(
                result=AgentResult(
                    response="Here is the answer.",
                    turns=2,
                    inference_ids=["inf-1", "inf-2"],
                    response_source="text",
                )
            ),
        ],
    ),
]


_ALL_FIXTURES: list[tuple[str, list[AgentEvent]]] = (
    _PER_EVENT_FIXTURES + _PER_SEQUENCE_FIXTURES
)


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "events",
    [pytest.param(events, id=fixture_id) for fixture_id, events in _ALL_FIXTURES],
)
async def test_public_projection_matches_the_golden_wire_shape(
    events: list[AgentEvent],
) -> None:
    """Every fixture produces identical (name, payload) sequences from
    the live stages and the frozen golden snapshot. If this fails, fix
    the stages — never the snapshot."""
    assert await _collect_public(events) == _collect_golden(events)


async def test_every_payload_is_json_serialisable_without_ascii_escapes() -> None:
    """The consumer's byte layer is ``json.dumps(..., ensure_ascii=False)``
    — every payload the projection emits must survive it."""
    for _, events in _ALL_FIXTURES:
        for name, data in await _collect_public(events):
            wire = json.dumps(data, ensure_ascii=False)
            assert isinstance(name, str)
            assert json.loads(wire) == data


async def test_dropped_events_produce_zero_output_in_both_pipelines() -> None:
    """Sanity: the dropped fixtures actually produce zero output — not
    just equal-but-non-empty output — so the privacy contract is
    visibly satisfied."""
    for fixture_id, events in _PER_EVENT_FIXTURES:
        if not fixture_id.startswith("dropped_"):
            continue
        assert await _collect_public(events) == [], fixture_id
        assert _collect_golden(events) == [], fixture_id


async def test_a_recoverable_error_mid_run_yields_no_public_error() -> None:
    """The sequence that motivated the contract revision, asserted
    directly rather than merely equal to the snapshot: the anomaly is
    internal, the answer streams on, and ``done`` lands."""
    events = dict(_ALL_FIXTURES)["seq_text_recoverable_error_text_done"]
    names = [name for name, _ in await _collect_public(events)]
    assert names == ["chunk", "end", "chunk", "end", "done"]


# ---------------------------------------------------------------------------
# MaxTurnsExceededError — in production the exception is raised outside
# the event fixture matrix; the frozen wire is
# error{message: "max_turns_exceeded", detail: {max_turns, exc_class}}.
# ---------------------------------------------------------------------------

_MAX_TURNS_GOLDEN = PublicEvent(
    "error",
    {
        "message": "max_turns_exceeded",
        "detail": {"max_turns": 20, "exc_class": "MaxTurnsExceededError"},
    },
)


async def test_max_turns_public_wire_from_the_frame() -> None:
    detail = {"max_turns": 20, "exc_class": "MaxTurnsExceededError"}
    frames = ErrorFrame(message="max_turns_exceeded", detail=detail)

    async def _one() -> AsyncIterator[ErrorFrame]:
        yield frames

    out = [event async for event in public_allowlist(_one())]
    assert out == [_MAX_TURNS_GOLDEN]


async def test_max_turns_public_wire_end_to_end_from_a_raising_run() -> None:
    """The full composition over a run that trips the loop guard ends
    with exactly the frozen error payload."""
    agent = ScriptedAgent(raises=MaxTurnsExceededError(20))
    run = agent.invoke("q")
    stream = public_allowlist(
        with_text_fallback(as_terminal_safe(project(run.events())))
    )
    out = [event async for event in stream]
    assert out == [_MAX_TURNS_GOLDEN]


# ---------------------------------------------------------------------------
# Elicitations — published out-of-band by the consumer, not produced by
# project(); the projection of the constructed frames is the contract.
# ---------------------------------------------------------------------------


async def _project_frames(
    frames: Sequence[ElicitationRequestFrame | ElicitationResolvedFrame],
) -> list[PublicEvent]:
    async def _stream() -> AsyncIterator[
        ElicitationRequestFrame | ElicitationResolvedFrame
    ]:
        for frame in frames:
            yield frame

    return [event async for event in public_allowlist(_stream())]


async def test_elicitation_request_public_wire() -> None:
    """Binary (empty schema) and bool scalar (object schema) requests;
    ``schema_json`` is re-parsed into an object on the wire."""
    binary = await _project_frames(
        [
            ElicitationRequestFrame(
                id="e1", message="Confirm?", response_type=None, schema_json="{}"
            )
        ]
    )
    assert binary == [
        PublicEvent(
            "elicitation_request",
            {
                "type": "elicitation_request",
                "id": "e1",
                "message": "Confirm?",
                "response_type": None,
                "schema": {},
            },
        )
    ]

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "boolean"}},
    }
    scalar = await _project_frames(
        [
            ElicitationRequestFrame(
                id="e2",
                message="Place order?",
                response_type="bool",
                schema_json=json.dumps(schema),
            )
        ]
    )
    assert scalar == [
        PublicEvent(
            "elicitation_request",
            {
                "type": "elicitation_request",
                "id": "e2",
                "message": "Place order?",
                "response_type": "bool",
                "schema": schema,
            },
        )
    ]


async def test_elicitation_resolved_public_wire() -> None:
    out = await _project_frames([ElicitationResolvedFrame(id="e1", action="accept")])
    assert out == [
        PublicEvent(
            "elicitation_resolved",
            {"type": "elicitation_resolved", "id": "e1", "action": "accept"},
        )
    ]
