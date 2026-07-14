"""Pure composable projection stages over the :data:`WireFrame` union.

Each stage is an async generator ``AsyncIterator → AsyncIterator`` with
no transport imports — consumers compose their own pipeline and
serialise the output however their transport needs. The canonical
composition (the shape a public SSE endpoint wants) is::

    project → as_terminal_safe → with_status → with_text_fallback
            → public_allowlist

Every ``match`` over the union ends in ``assert_never``, so adding a
frame variant refuses to type-check until each stage makes an explicit
decision about it — for :func:`public_allowlist` that decision is
render-vs-drop, and render-by-default would be a privacy regression.

The wire layer should consume **unfiltered** ``run.events()``: with
``on_error="event"`` the terminal :class:`RunErrorEvent` is subject to
selector filtering like any other event, and a filtered stream would
lose its terminal frame.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Final, assert_never

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
from t0_agents.wire.frames import (
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

MAX_ARG_BYTES: Final[int] = 8192
"""Truncation cap on ``json.dumps(arguments)`` for tool-call frames."""

MAX_RESULT_BYTES: Final[int] = 8192
"""Truncation cap on :class:`ToolResultEvent.result`."""

MAX_RAW_BYTES: Final[int] = 16384
"""Truncation cap on the JSON-serialised raw tool event carried for
debug surfaces. Larger than :data:`MAX_ARG_BYTES` /
:data:`MAX_RESULT_BYTES` because the raw dict contains those fields
plus the ``type`` discriminator and any future ones."""

_TRUNC_INDICATOR: Final[str] = "…"


def _truncate(value: str, max_bytes: int) -> tuple[str, int | None]:
    """Truncate ``value`` to ``max_bytes`` UTF-8 bytes.

    Returns the possibly-truncated string and the original
    ``len(value)`` when truncation happened, or ``None`` when the value
    was already small enough. Truncates on byte length so the wire
    payload stays bounded regardless of multi-byte code points; the
    indicator character ``…`` (3 UTF-8 bytes) is appended after
    truncation, so a value at exactly ``max_bytes`` is left alone.
    """
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, None
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + _TRUNC_INDICATOR, len(value)


def _error_frame(error_type: str, message: str, detail: Mapping[str, Any]) -> ErrorFrame:
    """Build the frozen terminal-error frame shape.

    ``detail["exc_class"]`` always carries the exception class name as
    the machine-readable discriminator. ``MaxTurnsExceededError`` is
    special-cased to the frozen public message ``"max_turns_exceeded"``
    so consumers key on a stable token instead of the human string.
    """
    if error_type == "MaxTurnsExceededError":
        message = "max_turns_exceeded"
    return ErrorFrame(message=message, detail={**detail, "exc_class": error_type})


async def project(events: AsyncIterator[AgentEvent]) -> AsyncIterator[WireFrame]:
    """Map each :data:`AgentEvent` to its wire frame — stateless.

    Cross-event synthesis belongs to the stateful stages
    (:func:`with_status`, :func:`with_text_fallback`); this stage is a
    pure per-event translation. Tool-call arguments, tool results, and
    raw-event JSON are truncated here so every downstream projection
    consumes the same bounded strings.

    A terminal :class:`RunErrorEvent` (``on_error="event"``) maps to the
    same frozen :class:`ErrorFrame` shape that :func:`as_terminal_safe`
    builds from a raised exception — the two failure paths are
    indistinguishable on the wire.
    """
    async for event in events:
        match event:
            case TextDeltaEvent():
                yield TextDeltaFrame(block_id=event.id, delta=event.delta)
            case TextEndEvent():
                yield TextEndFrame(block_id=event.id)
            case ToolCallDeltaEvent():
                yield ToolCallDeltaFrame(
                    id=event.id,
                    raw_name_delta=event.raw_name_delta,
                    raw_arguments_delta=event.raw_arguments_delta,
                )
            case ToolCallEvent():
                args_json = json.dumps(event.arguments, ensure_ascii=False)
                truncated_args, truncated_from = _truncate(args_json, MAX_ARG_BYTES)
                raw_event_json, _ = _truncate(
                    json.dumps(dataclasses.asdict(event), ensure_ascii=False),
                    MAX_RAW_BYTES,
                )
                yield ToolCallFrame(
                    id=event.id,
                    name=event.name,
                    arguments_json=truncated_args,
                    truncated_from=truncated_from,
                    raw_event_json=raw_event_json,
                )
            case ToolResultEvent():
                truncated_result, truncated_from = _truncate(
                    event.result, MAX_RESULT_BYTES
                )
                raw_event_json, _ = _truncate(
                    json.dumps(dataclasses.asdict(event), ensure_ascii=False),
                    MAX_RAW_BYTES,
                )
                yield ToolResultFrame(
                    id=event.id,
                    name=event.name,
                    result=truncated_result,
                    is_error=event.is_error,
                    truncated_from=truncated_from,
                    raw_event_json=raw_event_json,
                )
            case ToolProgressEvent():
                yield ToolProgressFrame(
                    id=event.id,
                    progress=event.progress,
                    total=event.total,
                    message=event.message,
                )
            case ThoughtDeltaEvent():
                yield ThoughtDeltaFrame(id=event.id, delta=event.delta)
            case ThoughtEvent():
                yield ThoughtFrame(id=event.id, text=event.text)
            case TurnEndEvent():
                yield TurnEndFrame(turn=event.turn)
            case DoneEvent():
                yield DoneFrame(
                    turns=event.result.turns,
                    response_source=event.result.response_source,
                    response_text=event.result.response,
                    inference_ids=tuple(event.result.inference_ids),
                )
            case ErrorEvent():
                yield ErrorFrame(message=event.message, detail=event.detail)
            case RunErrorEvent():
                yield _error_frame(event.error_type, event.message, event.detail)
            case _:
                assert_never(event)


__all__ = [
    "MAX_ARG_BYTES",
    "MAX_RAW_BYTES",
    "MAX_RESULT_BYTES",
    "project",
]
