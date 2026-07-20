"""Round-trip golden for :mod:`t0_agents.stream.codec`.

Every variant of the union must survive ``event_to_dict`` → ``json`` →
``event_from_dict`` unchanged (dataclass equality), with the two
documented exceptions: :attr:`RunErrorEvent.exception` decodes as
``None`` (and is ``compare=False``), and non-primitive
:attr:`AgentResult.value` payloads degrade to plain data.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from pydantic import BaseModel

from t0_agents import (
    CODEC_VERSION,
    AgentResult,
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
    event_from_dict,
    event_to_dict,
)
from t0_agents.core.output import OutputDiagnostics
from t0_agents.core.types import GuardrailLogEntry
from t0_agents.stream.events import AgentEvent

_RICH_RESULT = AgentResult(
    value="It's 22°C.",
    episode_id="ep-1",
    inference_ids=["inf-1", "inf-2"],
    turns=3,
    response_source="thought",
    guardrail_log=[
        GuardrailLogEntry(
            guardrail_name="no_pii",
            hook="after_run",
            kind="pass",
            tool_name=None,
            detail="",
        )
    ],
    diagnostics=OutputDiagnostics(
        parse_strategy="permissive",
        attempts=1,
        raw_text='{"answer": 22}',
        failure_detail=None,
    ),
)

_ALL_VARIANTS: list[AgentEvent] = [
    TextDeltaEvent(id="t1", delta="Hel"),
    TextEndEvent(id="t1"),
    ToolCallDeltaEvent(id="c1", raw_name_delta="get_", raw_arguments_delta='{"ci'),
    ToolCallEvent(id="c1", name="get_weather", arguments={"city": "Bogotá", "days": 3}),
    ToolResultEvent(
        id="c1",
        name="get_weather",
        result='{"temp": 22}',
        is_error=False,
        structured={"temp": 22},
    ),
    ToolProgressEvent(id="c1", progress=2.0, total=5.0, message="fetching page 2"),
    ThoughtDeltaEvent(id="th1", delta="hmm"),
    ThoughtEvent(id="th1", text="hmm, weather"),
    TurnEndEvent(turn=2),
    DoneEvent(result=_RICH_RESULT),
    ErrorEvent(message="invalid tool JSON", detail={"tool": "get_weather"}),
    RunErrorEvent(
        error_type="MaxTurnsExceededError",
        message="Agent loop exceeded 20 turns",
        detail={"max_turns": 20},
        exception=None,
    ),
]


def _union_member_count() -> int:
    import typing

    from t0_agents.stream import events

    return len(typing.get_args(events.AgentEvent))


class TestGoldenRoundTrip:
    def test_covers_every_union_variant(self) -> None:
        covered = {type(event) for event in _ALL_VARIANTS}
        assert len(covered) == len(_ALL_VARIANTS) == _union_member_count()

    @pytest.mark.parametrize("event", _ALL_VARIANTS, ids=lambda e: e.type)
    def test_round_trips_through_json(self, event: AgentEvent) -> None:
        encoded = event_to_dict(event)
        assert encoded["v"] == CODEC_VERSION
        assert encoded["type"] == event.type
        wire = json.dumps(encoded, ensure_ascii=False)
        decoded = event_from_dict(json.loads(wire))
        assert decoded == event

    def test_container_fields_are_copies(self) -> None:
        event = ToolCallEvent(id="c1", name="t", arguments={"nested": {"a": 1}})
        encoded = event_to_dict(event)
        encoded["arguments"]["nested"]["a"] = 999
        assert event.arguments["nested"]["a"] == 1

    def test_run_error_exception_never_encodes(self) -> None:
        event = RunErrorEvent(
            error_type="ValueError",
            message="boom",
            exception=ValueError("boom"),
        )
        encoded = event_to_dict(event)
        assert "exception" not in encoded
        decoded = event_from_dict(encoded)
        assert isinstance(decoded, RunErrorEvent)
        assert decoded.exception is None
        assert decoded == event  # exception is compare=False

    def test_tool_result_structured_is_encoded_and_round_trips(self) -> None:
        populated = ToolResultEvent(
            id="c1", name="get_weather", result='{"temp": 22}', structured={"temp": 22}
        )
        encoded = event_to_dict(populated)
        assert encoded["structured"] == {"temp": 22}
        assert event_from_dict(encoded) == populated

        default = ToolResultEvent(id="c2", name="t", result="ok")
        assert event_to_dict(default)["structured"] is None
        assert event_from_dict(event_to_dict(default)) == default


class TestValueDegradation:
    def test_pydantic_value_encodes_via_model_dump(self) -> None:
        class Weather(BaseModel):
            temp: int
            city: str

        done = DoneEvent(result=AgentResult(value=Weather(temp=22, city="Bogotá"), turns=1))
        decoded = event_from_dict(json.loads(json.dumps(event_to_dict(done))))
        assert isinstance(decoded, DoneEvent)
        assert decoded.result.value == {"temp": 22, "city": "Bogotá"}

    def test_dataclass_value_encodes_via_asdict(self) -> None:
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        done = DoneEvent(result=AgentResult(value=Point(1, 2), turns=1))
        decoded = event_from_dict(event_to_dict(done))
        assert isinstance(decoded, DoneEvent)
        assert decoded.result.value == {"x": 1, "y": 2}

    def test_collections_of_models_degrade_recursively(self) -> None:
        class Item(BaseModel):
            name: str

        @dataclasses.dataclass
        class Point:
            x: int

        value = {"items": [Item(name="a"), Item(name="b")], "origin": (Point(1),)}
        done = DoneEvent(result=AgentResult(value=value, turns=1))
        decoded = event_from_dict(json.loads(json.dumps(event_to_dict(done))))
        assert isinstance(decoded, DoneEvent)
        assert decoded.result.value == {
            "items": [{"name": "a"}, {"name": "b"}],
            "origin": [{"x": 1}],  # tuples decode as lists — JSON has no tuple
        }


class TestDecodeErrors:
    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown AgentEvent type: 'status'"):
            event_from_dict({"v": CODEC_VERSION, "type": "status"})

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="no 'type' discriminator"):
            event_from_dict({"v": CODEC_VERSION, "id": "t1"})

    def test_version_mismatch_raises(self) -> None:
        data: dict[str, Any] = {"v": CODEC_VERSION + 1, "type": "text_end", "id": "t1"}
        with pytest.raises(ValueError, match="unsupported event codec version"):
            event_from_dict(data)

    def test_missing_version_defaults_to_current(self) -> None:
        decoded = event_from_dict({"type": "text_end", "id": "t1"})
        assert decoded == TextEndEvent(id="t1")
