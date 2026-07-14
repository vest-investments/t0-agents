"""Versioned dict codec for the :data:`AgentEvent` union.

``event_to_dict`` / ``event_from_dict`` turn any event into a plain,
JSON-serialisable dict and back — the primitive every bus, SSE relay,
and recovery replay needs, so consumers stop hand-rolling mirror unions.
The dict shape is ``{"v": CODEC_VERSION, "type": <discriminator>,
**fields}`` where ``type`` is the event's existing ``Literal``
discriminator.

Fidelity contract:

* Every field of every variant round-trips **except**
  :attr:`RunErrorEvent.exception` (in-process only — decodes as
  ``None``) and :attr:`AgentResult.value` for non-primitive payloads:
  a pydantic model encodes via its ``model_dump()``, a dataclass via
  :func:`dataclasses.asdict`, anything else passes through and must be
  JSON-serialisable.  The decoded ``value`` is therefore plain data,
  not a re-hydrated instance — the codec cannot know the class.
* Unknown ``type`` values and codec-version mismatches raise
  :class:`ValueError` — an outdated payload must fail loudly, never
  degrade silently.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from t0_agents.core.output import OutputDiagnostics
from t0_agents.core.types import AgentResult, GuardrailLogEntry
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

__all__ = ["CODEC_VERSION", "event_from_dict", "event_to_dict"]

CODEC_VERSION = 1
"""Bump on any non-additive change to the encoded shape."""

_EVENT_TYPES: dict[str, Callable[..., AgentEvent]] = {
    "text_delta": TextDeltaEvent,
    "text_end": TextEndEvent,
    "tool_call_delta": ToolCallDeltaEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "tool_progress": ToolProgressEvent,
    "thought_delta": ThoughtDeltaEvent,
    "thought": ThoughtEvent,
    "turn_end": TurnEndEvent,
    "done": DoneEvent,
    "error": ErrorEvent,
    "run_error": RunErrorEvent,
}


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Encode ``event`` as a plain JSON-serialisable dict.

    Container fields (``arguments``, ``detail``) are deep-copied, so
    mutating the returned dict never aliases the live event.
    """
    if isinstance(event, DoneEvent):
        return {
            "v": CODEC_VERSION,
            "type": event.type,
            "result": _result_to_dict(event.result),
        }
    if isinstance(event, RunErrorEvent):
        # Built by hand: dataclasses.asdict would deep-copy the
        # non-serialised ``exception`` field (traceback and all).
        return {
            "v": CODEC_VERSION,
            "type": event.type,
            "error_type": event.error_type,
            "message": event.message,
            "detail": copy.deepcopy(event.detail),
        }
    data: dict[str, Any] = {"v": CODEC_VERSION}
    data.update(dataclasses.asdict(event))
    return data


def event_from_dict(data: Mapping[str, Any]) -> AgentEvent:
    """Decode a dict produced by :func:`event_to_dict`.

    Raises:
        ValueError: on a missing/unknown ``type`` discriminator or a
            codec-version mismatch.
    """
    version = data.get("v", CODEC_VERSION)
    if version != CODEC_VERSION:
        raise ValueError(
            f"unsupported event codec version: {version!r} (supported: {CODEC_VERSION})"
        )
    kind = data.get("type")
    if kind is None:
        raise ValueError("event dict has no 'type' discriminator")
    event_cls = _EVENT_TYPES.get(kind)
    if event_cls is None:
        raise ValueError(f"unknown AgentEvent type: {kind!r}")
    payload = {key: value for key, value in data.items() if key not in ("v", "type")}
    if event_cls is DoneEvent:
        return DoneEvent(result=_result_from_dict(payload["result"]))
    return event_cls(**payload)


def _value_to_data(value: Any) -> Any:
    """Degrade ``value`` to plain data (see module fidelity contract)."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _result_to_dict(result: AgentResult[Any]) -> dict[str, Any]:
    return {
        "value": _value_to_data(result.value),
        "episode_id": result.episode_id,
        "inference_ids": list(result.inference_ids),
        "turns": result.turns,
        "response_source": result.response_source,
        "guardrail_log": [dataclasses.asdict(entry) for entry in result.guardrail_log],
        "diagnostics": (
            dataclasses.asdict(result.diagnostics) if result.diagnostics is not None else None
        ),
    }


def _result_from_dict(data: Mapping[str, Any]) -> AgentResult[Any]:
    diagnostics = data.get("diagnostics")
    return AgentResult(
        value=data["value"],
        episode_id=data.get("episode_id"),
        inference_ids=list(data.get("inference_ids") or []),
        turns=int(data.get("turns", 0)),
        response_source=cast(
            Literal["text", "thought"], data.get("response_source", "text")
        ),
        guardrail_log=[
            GuardrailLogEntry(**entry) for entry in (data.get("guardrail_log") or [])
        ],
        diagnostics=OutputDiagnostics(**diagnostics) if diagnostics is not None else None,
    )
