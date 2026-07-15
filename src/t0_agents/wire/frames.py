"""Frozen wire-frame union — the versioned consumer contract.

:data:`WireFrame` is the shape t0-agents promises to downstream
consumers (SSE relays, buses, recovery replays). It mirrors the
:data:`t0_agents.stream.events.AgentEvent` union one frame per event,
plus three consumer-facing additions the event union deliberately does
not carry:

* :class:`InitFrame` — a consumer-synthesised greeting (e.g. once per
  SSE subscription).
* :class:`StatusFrame` — opaque progress hints synthesised by
  :func:`t0_agents.wire.stages.with_status`.
* :class:`ElicitationRequestFrame` / :class:`ElicitationResolvedFrame`
  — human-in-the-loop prompts, published out-of-band from the event
  stream by the consumer's elicitation wiring.

Evolution policy — **additive only**:

* A new frame is a new class + a new ``kind`` literal; existing frames
  never lose or rename fields, and existing ``kind`` strings never
  change meaning. Consumers with exhaustive ``match`` statements add
  one arm per new frame and are otherwise untouched.
* :data:`WIRE_VERSION` is bumped **only** on a non-additive change to
  the encoded shape, which is a breaking release of the contract.
  :func:`frame_from_dict` rejects mismatched versions loudly.

Each frame is ``@dataclass(frozen=True, slots=True)`` with a
``kind: ClassVar[str]`` discriminator matching the upstream event
``type`` literal. ``ClassVar`` stays out of ``__slots__`` and
``dataclasses.fields(...)``, so the discriminator does not bloat
instances and does not appear in the codec's field iteration.

This module is **not a broker**: it defines stable frames and a dict
codec, and :mod:`t0_agents.wire.stages` defines pure projections over
them. Transports (Redis buses, SSE serialisers, WebSockets) belong to
the consumer.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

WIRE_VERSION = 1
"""Bump only on a non-additive change to the encoded shape."""


@dataclass(frozen=True, slots=True)
class InitFrame:
    """Consumer-synthesised greeting carrying the ``conversation_id``.

    Not produced by :func:`t0_agents.wire.stages.project` — the
    consumer emits it once per subscription before relaying the run's
    frames, so a client learns its conversation id on the same channel
    it will stream from.
    """

    kind: ClassVar[str] = "init"
    conversation_id: str


@dataclass(frozen=True, slots=True)
class TextDeltaFrame:
    """One delta of assistant text for the content block ``block_id``."""

    kind: ClassVar[str] = "text_delta"
    block_id: str
    delta: str


@dataclass(frozen=True, slots=True)
class TextEndFrame:
    """Final marker for a text block — pairs with the prior
    :class:`TextDeltaFrame` s sharing ``block_id``."""

    kind: ClassVar[str] = "text_end"
    block_id: str


@dataclass(frozen=True, slots=True)
class ToolCallDeltaFrame:
    """Partial tool-call name or arguments as the gateway streams them.

    Matches :class:`t0_agents.stream.events.ToolCallDeltaEvent` 1:1 —
    consumer-specific accumulation is a projection concern.
    """

    kind: ClassVar[str] = "tool_call_delta"
    id: str
    raw_name_delta: str
    raw_arguments_delta: str


@dataclass(frozen=True, slots=True)
class ToolCallFrame:
    """Assembled tool call.

    ``arguments_json`` is the already-serialised + truncated JSON
    string of the parsed ``arguments`` dict; ``raw_event_json`` is the
    truncated JSON of the full event for debug surfaces. Both bounds
    are applied by :func:`t0_agents.wire.stages.project` so every
    downstream projection consumes the same bounded strings.
    """

    kind: ClassVar[str] = "tool_call"
    id: str
    name: str
    arguments_json: str
    truncated_from: int | None
    raw_event_json: str


@dataclass(frozen=True, slots=True)
class ToolResultFrame:
    """Result of executing the matching :class:`ToolCallFrame`.

    ``result`` is the truncated string a projection renders;
    ``truncated_from`` carries the original length when truncation
    happened (``None`` otherwise).
    """

    kind: ClassVar[str] = "tool_result"
    id: str
    name: str
    result: str
    is_error: bool
    truncated_from: int | None
    raw_event_json: str


@dataclass(frozen=True, slots=True)
class ToolProgressFrame:
    """Progress report from a running MCP tool.

    Mirrors :class:`t0_agents.stream.events.ToolProgressEvent`; ``id``
    correlates with the in-flight :class:`ToolCallFrame`.
    """

    kind: ClassVar[str] = "tool_progress"
    id: str
    progress: float
    total: float | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ThoughtDeltaFrame:
    """One delta of reasoning text for the thought block ``id``."""

    kind: ClassVar[str] = "thought_delta"
    id: str
    delta: str


@dataclass(frozen=True, slots=True)
class ThoughtFrame:
    """Assembled reasoning text after the final delta for ``id``."""

    kind: ClassVar[str] = "thought"
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class TurnEndFrame:
    """Emitted once per agent turn, after every block for that turn
    finished assembling."""

    kind: ClassVar[str] = "turn_end"
    turn: int


@dataclass(frozen=True, slots=True)
class DoneFrame:
    """Terminal success frame: carries everything any projection needs.

    ``response_text`` and ``inference_ids`` feed the reasoning-model
    fallback in :func:`t0_agents.wire.stages.with_text_fallback`
    (synthesise text frames from the final response when no text
    streamed during the run); ``turns`` and ``response_source`` feed
    the public ``done`` payload.
    """

    kind: ClassVar[str] = "done"
    turns: int
    response_source: Literal["text", "thought"]
    response_text: str
    inference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ErrorFrame:
    """Anomaly on the stream — recoverable or terminal.

    A recoverable :class:`t0_agents.stream.events.ErrorEvent` maps here
    verbatim; terminal failures (a raised exception or a
    :class:`t0_agents.stream.events.RunErrorEvent`) map here with
    ``detail["exc_class"]`` carrying the exception class name as a
    machine-readable discriminator. The wire ``kind`` stays ``error``
    for both — whether the stream continues is answered by whether more
    frames follow.
    """

    kind: ClassVar[str] = "error"
    message: str
    detail: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StatusFrame:
    """Opaque progress hint synthesised by
    :func:`t0_agents.wire.stages.with_status`.

    ``text`` is human-readable English (e.g. ``"Calling get_weather…"``,
    ``"Thinking…"``) — consumers render it verbatim and must not parse
    it for structured detail. ``server`` / ``tool`` carry the
    structured tool identity behind the hint so a client can map it to
    a custom, localized message; both are ``None`` for non-tool hints.
    """

    kind: ClassVar[str] = "status"
    text: str
    server: str | None = None
    tool: str | None = None


@dataclass(frozen=True, slots=True)
class ElicitationRequestFrame:
    """Human-in-the-loop prompt fired by an MCP tool's ``ctx.elicit(...)``.

    Published by the consumer out-of-band from the event stream (an MCP
    tool's elicit call goes through the ``fastmcp`` elicitation handler,
    which the consumer wires to publish this frame). ``id`` is the
    correlation key the resolve path uses to look up the pending
    request.

    ``response_type`` mirrors MCP's wire format: ``None`` for binary
    approval (accept / decline), or the requested scalar type name as a
    string (``"bool"``, ``"str"``, ``"int"``, …) when the tool asked
    for structured input. ``schema_json`` carries the raw MCP
    requested-schema JSON so clients render by schema introspection —
    never by Python class name, which is not a stable contract.
    """

    kind: ClassVar[str] = "elicitation_request"
    id: str
    message: str
    response_type: str | None
    schema_json: str


@dataclass(frozen=True, slots=True)
class ElicitationResolvedFrame:
    """Confirms a prior :class:`ElicitationRequestFrame` was answered.

    Published by the consumer's resolve path so every subscriber
    (current view, reconnect replay, multi-pane) sees the prompt
    closed.
    """

    kind: ClassVar[str] = "elicitation_resolved"
    id: str
    action: Literal["accept", "decline", "cancel"]


WireFrame = (
    InitFrame
    | TextDeltaFrame
    | TextEndFrame
    | ToolCallDeltaFrame
    | ToolCallFrame
    | ToolResultFrame
    | ToolProgressFrame
    | ThoughtDeltaFrame
    | ThoughtFrame
    | TurnEndFrame
    | DoneFrame
    | ErrorFrame
    | StatusFrame
    | ElicitationRequestFrame
    | ElicitationResolvedFrame
)


_FRAME_KINDS: dict[str, type[WireFrame]] = {
    "init": InitFrame,
    "text_delta": TextDeltaFrame,
    "text_end": TextEndFrame,
    "tool_call_delta": ToolCallDeltaFrame,
    "tool_call": ToolCallFrame,
    "tool_result": ToolResultFrame,
    "tool_progress": ToolProgressFrame,
    "thought_delta": ThoughtDeltaFrame,
    "thought": ThoughtFrame,
    "turn_end": TurnEndFrame,
    "done": DoneFrame,
    "error": ErrorFrame,
    "status": StatusFrame,
    "elicitation_request": ElicitationRequestFrame,
    "elicitation_resolved": ElicitationResolvedFrame,
}


def frame_to_dict(frame: WireFrame) -> dict[str, Any]:
    """Encode ``frame`` as a plain JSON-serialisable dict.

    The shape is ``{"v": WIRE_VERSION, "kind": <discriminator>,
    **fields}``. Tuples encode as lists (JSON has no tuple) and dict
    fields are deep-copied, so mutating the returned dict never aliases
    the live frame.
    """
    data: dict[str, Any] = {"v": WIRE_VERSION, "kind": frame.kind}
    for field in dataclasses.fields(frame):
        value = getattr(frame, field.name)
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = copy.deepcopy(value)
        data[field.name] = value
    return data


def frame_from_dict(data: Mapping[str, Any]) -> WireFrame:
    """Decode a dict produced by :func:`frame_to_dict`.

    Unknown *fields* are ignored: under additive-only evolution a newer
    writer may add fields to an existing frame without a version bump,
    and an older reader must tolerate them. Unknown *kinds* and version
    mismatches still fail loudly — those mark payloads this reader
    cannot represent at all.

    Raises:
        ValueError: on a missing/unknown ``kind`` discriminator or a
            wire-version mismatch.
    """
    version = data.get("v", WIRE_VERSION)
    if version != WIRE_VERSION:
        raise ValueError(
            f"unsupported wire version: {version!r} (supported: {WIRE_VERSION})"
        )
    kind = data.get("kind")
    if kind is None:
        raise ValueError("wire dict has no 'kind' discriminator")
    frame_cls = _FRAME_KINDS.get(kind)
    if frame_cls is None:
        raise ValueError(f"unknown WireFrame kind: {kind!r}")
    allowed_fields = {field.name for field in dataclasses.fields(frame_cls)}
    payload = {key: value for key, value in data.items() if key in allowed_fields}
    if frame_cls is DoneFrame and "inference_ids" in payload:
        payload["inference_ids"] = tuple(payload["inference_ids"])
    elif frame_cls is ErrorFrame and "detail" in payload:
        payload["detail"] = copy.deepcopy(payload["detail"])
    return frame_cls(**payload)


__all__ = [
    "WIRE_VERSION",
    "DoneFrame",
    "ElicitationRequestFrame",
    "ElicitationResolvedFrame",
    "ErrorFrame",
    "InitFrame",
    "StatusFrame",
    "TextDeltaFrame",
    "TextEndFrame",
    "ThoughtDeltaFrame",
    "ThoughtFrame",
    "ToolCallDeltaFrame",
    "ToolCallFrame",
    "ToolProgressFrame",
    "ToolResultFrame",
    "TurnEndFrame",
    "WireFrame",
    "frame_from_dict",
    "frame_to_dict",
]
