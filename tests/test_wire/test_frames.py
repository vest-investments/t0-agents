"""Round-trip golden for the wire-frame codec.

Every :data:`WireFrame` variant must survive ``frame_to_dict`` →
``json.dumps`` / ``json.loads`` → ``frame_from_dict`` unchanged — buses
and recovery replays depend on it. Unknown kinds, missing
discriminators, and version mismatches must raise, never degrade.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from t0_agents.wire import (
    WIRE_VERSION,
    DoneFrame,
    ElicitationRequestFrame,
    ElicitationResolvedFrame,
    ErrorFrame,
    InitFrame,
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
    frame_from_dict,
    frame_to_dict,
)

ALL_VARIANTS: list[WireFrame] = [
    InitFrame(conversation_id="conv-1"),
    TextDeltaFrame(block_id="b1", delta="Hola — ñandú"),
    TextEndFrame(block_id="b1"),
    ToolCallDeltaFrame(id="t1", raw_name_delta="get_", raw_arguments_delta='{"ci'),
    ToolCallFrame(
        id="t1",
        name="get_weather",
        arguments_json='{"city": "NYC"}',
        truncated_from=None,
        raw_event_json='{"type": "tool_call"}',
    ),
    ToolResultFrame(
        id="t1",
        name="get_weather",
        result="72F",
        is_error=False,
        truncated_from=9000,
        raw_event_json='{"type": "tool_result"}',
    ),
    ToolProgressFrame(id="t1", progress=3.0, total=10.0, message="fetching"),
    ThoughtDeltaFrame(id="th1", delta="hmm"),
    ThoughtFrame(id="th1", text="hmm, considered"),
    TurnEndFrame(turn=2),
    DoneFrame(
        turns=2,
        response_source="thought",
        response_text="The answer is 42.",
        inference_ids=("inf-1", "inf-2"),
    ),
    ErrorFrame(message="boom", detail={"k": "v", "nested": {"n": 1}}),
    StatusFrame(text="Calling get_weather…", server="fred", tool="get_macro"),
    StatusFrame(text="Thinking…"),
    ElicitationRequestFrame(
        id="e1",
        message="Proceed?",
        response_type=None,
        schema_json="{}",
    ),
    ElicitationRequestFrame(
        id="e2",
        message="Confirm the trade",
        response_type="bool",
        schema_json='{"type": "object", "properties": {"value": {"type": "boolean"}}}',
    ),
    ElicitationResolvedFrame(id="e1", action="accept"),
]


@pytest.mark.parametrize("frame", ALL_VARIANTS, ids=lambda f: type(f).__name__)
def test_round_trip_through_json(frame: WireFrame) -> None:
    encoded = frame_to_dict(frame)
    assert encoded["v"] == WIRE_VERSION
    assert encoded["kind"] == frame.kind
    over_the_wire = json.loads(json.dumps(encoded, ensure_ascii=False))
    assert frame_from_dict(over_the_wire) == frame


def test_every_union_variant_is_covered() -> None:
    covered = {type(frame) for frame in ALL_VARIANTS}
    union_members = set(WireFrame.__args__)  # type: ignore[attr-defined]
    assert covered == union_members


def test_inference_ids_decode_as_tuple() -> None:
    frame = DoneFrame(
        turns=1, response_source="text", response_text="hi", inference_ids=("inf-1",)
    )
    decoded = frame_from_dict(frame_to_dict(frame))
    assert isinstance(decoded, DoneFrame)
    assert decoded.inference_ids == ("inf-1",)
    assert isinstance(decoded.inference_ids, tuple)


def test_encoded_detail_never_aliases_the_frame() -> None:
    frame = ErrorFrame(message="boom", detail={"nested": {"n": 1}})
    encoded = frame_to_dict(frame)
    encoded["detail"]["nested"]["n"] = 999
    assert frame.detail["nested"]["n"] == 1


def test_version_mismatch_raises() -> None:
    data = frame_to_dict(TurnEndFrame(turn=1))
    data["v"] = WIRE_VERSION + 1
    with pytest.raises(ValueError, match="unsupported wire version"):
        frame_from_dict(data)


def test_missing_version_assumes_current() -> None:
    data = frame_to_dict(TurnEndFrame(turn=1))
    del data["v"]
    assert frame_from_dict(data) == TurnEndFrame(turn=1)


def test_unknown_fields_are_ignored_for_forward_compat() -> None:
    """An older reader must tolerate payloads from a newer additive
    writer — extra fields decode away, they never TypeError."""
    data = frame_to_dict(TurnEndFrame(turn=1))
    data["field_from_the_future"] = "ignored"
    assert frame_from_dict(data) == TurnEndFrame(turn=1)

    done = frame_to_dict(
        DoneFrame(
            turns=1, response_source="text", response_text="hi", inference_ids=("i1",)
        )
    )
    done["field_from_the_future"] = {"nested": True}
    decoded = frame_from_dict(done)
    assert isinstance(decoded, DoneFrame)
    assert decoded.inference_ids == ("i1",)


def test_missing_kind_raises() -> None:
    with pytest.raises(ValueError, match="no 'kind' discriminator"):
        frame_from_dict({"v": WIRE_VERSION, "turn": 1})


def test_unknown_kind_raises() -> None:
    data: dict[str, Any] = {"v": WIRE_VERSION, "kind": "hologram", "x": 1}
    with pytest.raises(ValueError, match="unknown WireFrame kind"):
        frame_from_dict(data)
