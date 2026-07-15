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
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Final, NamedTuple, assert_never

from t0_agents.core.exceptions import MaxTurnsExceededError
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
from t0_agents.stream.status import split_tool_identity
from t0_agents.wire.frames import (
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

EMPTY_TURN_RETRY_TEXT: Final[str] = "I couldn't complete that — please try again."
"""Belt text :func:`with_text_fallback` synthesises when a run produced
neither streamed text nor a final response — the wire never ends a
successful run blank. Part of the frozen public contract."""

FALLBACK_BLOCK_ID: Final[str] = "thought-fallback"
"""Block id for synthesised fallback text (and the public ``done``
payload id) when :class:`DoneFrame.inference_ids` is empty. Part of the
frozen public contract."""


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


async def as_terminal_safe(
    frames: AsyncIterator[WireFrame],
) -> AsyncIterator[WireFrame]:
    """Guarantee the stream ends in a frame, never an exception.

    An exception raised while draining ``frames`` is converted into one
    terminal :class:`ErrorFrame` — the same frozen shape
    :func:`project` builds from a :class:`RunErrorEvent`, so a consumer
    sees identical wire output whether the upstream ran with
    ``on_error="event"`` or the default raising mode. The ``message``
    policy mirrors ``Run.events(on_error="event")``: prefer the
    exception's ``text`` attribute (TensorZero errors carry the
    provider detail there), then ``str(exc)``, then the class name
    (bare exceptions like ``TimeoutError()`` stringify empty).

    ``asyncio.CancelledError`` derives from :class:`BaseException` and
    propagates — cancellation is not a wire event.
    """
    try:
        async for frame in frames:
            yield frame
    except Exception as exc:
        detail: dict[str, Any] = {}
        if isinstance(exc, MaxTurnsExceededError):
            detail["max_turns"] = exc.max_turns
        text = getattr(exc, "text", None)
        message = (text if isinstance(text, str) else None) or str(exc) or type(exc).__name__
        yield _error_frame(type(exc).__name__, message, detail)


async def with_status(
    frames: AsyncIterator[WireFrame],
    *,
    known_servers: Sequence[str] = (),
) -> AsyncIterator[WireFrame]:
    """Intercalate :class:`StatusFrame` hints after their triggers.

    Every input frame is re-emitted first, then any synthesised status
    — downstream stages see the same underlying sequence they would
    have seen without this stage. The policy is the production one:

    * ``"Calling {name}…"`` after every :class:`ToolCallFrame` (no
      dedup — each call is news), carrying the ``(server, tool)``
      identity split via
      :func:`t0_agents.stream.status.split_tool_identity` against
      ``known_servers``.
    * ``"Thinking…"`` once per contiguous reasoning block, suppressed
      for the rest of the run once any tool call fired — the per-tool
      hint stands until the next tool call or the run terminator.
    * A terminal :class:`DoneFrame` / :class:`ErrorFrame` re-arms both
      flags, so the next run on a long-lived stream surfaces its own
      initial ``"Thinking…"``.
    """
    in_thought_block = False
    saw_tool_call = False
    async for frame in frames:
        yield frame
        match frame:
            case ToolCallFrame(name=name):
                in_thought_block = False
                saw_tool_call = True
                server, tool = split_tool_identity(name, known_servers)
                yield StatusFrame(text=f"Calling {name}…", server=server, tool=tool)
            case ThoughtFrame() | ThoughtDeltaFrame():
                if not in_thought_block and not saw_tool_call:
                    in_thought_block = True
                    yield StatusFrame(text="Thinking…")
            case DoneFrame() | ErrorFrame():
                in_thought_block = False
                saw_tool_call = False
            case (
                InitFrame()
                | TextDeltaFrame()
                | TextEndFrame()
                | ToolCallDeltaFrame()
                | ToolResultFrame()
                | ToolProgressFrame()
                | TurnEndFrame()
                | StatusFrame()
                | ElicitationRequestFrame()
                | ElicitationResolvedFrame()
            ):
                # Any non-thought, non-tool frame ends the contiguous
                # thought block; saw_tool_call deliberately stands —
                # only the run terminator re-arms it.
                in_thought_block = False
            case _:
                assert_never(frame)


async def with_text_fallback(
    frames: AsyncIterator[WireFrame],
) -> AsyncIterator[WireFrame]:
    """Guarantee every successful run ends with visible text.

    Reasoning models (OpenAI o-series, ``gpt-oss`` via Harmony) route
    the final answer through thought summaries and stream no text
    deltas; the thought frames are dropped by
    :func:`public_allowlist`, so without this stage the client would
    see a blank run. When a :class:`DoneFrame` arrives and no non-blank
    :class:`TextDeltaFrame` streamed this run, a
    ``TextDeltaFrame`` + :class:`TextEndFrame` pair is synthesised
    before it — from :attr:`DoneFrame.response_text` when non-blank
    (reasoning fallback), else from :data:`EMPTY_TURN_RETRY_TEXT`
    (never-blank belt). The synthesised block id is the last inference
    id, or :data:`FALLBACK_BLOCK_ID` when there is none — matching the
    public ``done`` payload id.

    ``saw_text`` re-arms at every run boundary so each run on a
    long-lived stream gets its own fallback decision: after a
    :class:`DoneFrame`, after a *terminal* :class:`ErrorFrame`, and on
    an :class:`InitFrame` (a new session). A terminal error frame is
    recognised by ``detail["exc_class"]`` — the discriminator both
    terminal paths always set — while a recoverable
    :class:`~t0_agents.stream.events.ErrorEvent` passthrough carries no
    ``exc_class`` and must NOT re-arm: the run continues, and resetting
    mid-run would make the fallback duplicate text that already
    streamed. Elicitation frames never touch ``saw_text`` — a prompt is
    not assistant text and must not suppress the fallback.
    """
    saw_text = False
    async for frame in frames:
        match frame:
            case TextDeltaFrame(delta=delta):
                if delta.strip():
                    saw_text = True
                yield frame
            case DoneFrame(response_text=text, inference_ids=ids):
                if not saw_text:
                    msg_id = ids[-1] if ids else FALLBACK_BLOCK_ID
                    body = text if text.strip() else EMPTY_TURN_RETRY_TEXT
                    yield TextDeltaFrame(block_id=msg_id, delta=body)
                    yield TextEndFrame(block_id=msg_id)
                saw_text = False
                yield frame
            case InitFrame():
                saw_text = False
                yield frame
            case ErrorFrame(detail=detail):
                if "exc_class" in detail:
                    saw_text = False
                yield frame
            case (
                TextEndFrame()
                | ToolCallDeltaFrame()
                | ToolCallFrame()
                | ToolResultFrame()
                | ToolProgressFrame()
                | ThoughtDeltaFrame()
                | ThoughtFrame()
                | TurnEndFrame()
                | StatusFrame()
                | ElicitationRequestFrame()
                | ElicitationResolvedFrame()
            ):
                yield frame
            case _:
                assert_never(frame)


class PublicEvent(NamedTuple):
    """One public wire event: a frozen ``name`` plus its payload dict.

    The terminal output of :func:`public_allowlist` — the consumer
    serialises it onto its transport (an SSE endpoint writes
    ``event: {name}\\ndata: {json.dumps(data)}\\n\\n``); the names and
    payload shapes are the frozen contract, the bytes are the
    consumer's.
    """

    name: str
    data: dict[str, Any]


async def public_allowlist(
    frames: AsyncIterator[WireFrame],
) -> AsyncIterator[PublicEvent]:
    """Project frames onto the frozen public wire — the privacy guard.

    Exactly eight event names exist: ``init, chunk, end, status, error,
    done, elicitation_request, elicitation_resolved``. Tool activity
    (:class:`ToolCallFrame` / :class:`ToolCallDeltaFrame` /
    :class:`ToolResultFrame` / :class:`ToolProgressFrame`) and
    reasoning (:class:`ThoughtFrame` / :class:`ThoughtDeltaFrame`), plus
    :class:`TurnEndFrame`, are dropped: arguments, results, and thought
    content must never reach a public client, and the ``assert_never``
    arm forces an explicit render-vs-drop decision for every future
    frame — render-by-default would be a privacy regression.

    Payload shapes are frozen exactly as production emits them,
    including which payloads carry a ``"type"`` field and which do not.
    Compose after :func:`with_text_fallback` so synthesised fallback
    text is projected like streamed text.
    """
    async for frame in frames:
        match frame:
            case InitFrame(conversation_id=conversation_id):
                yield PublicEvent("init", {"conversation_id": conversation_id})
            case TextDeltaFrame(block_id=block_id, delta=delta):
                yield PublicEvent(
                    "chunk",
                    {"type": "chunk", "content": delta, "message_id": block_id},
                )
            case TextEndFrame(block_id=block_id):
                yield PublicEvent("end", {"type": "end", "id": block_id})
            case ErrorFrame(message=message, detail=detail):
                yield PublicEvent("error", {"message": message, "detail": detail})
            case StatusFrame(text=text, server=server, tool=tool):
                yield PublicEvent(
                    "status", {"text": text, "server": server, "tool": tool}
                )
            case DoneFrame(
                turns=turns, response_source=response_source, inference_ids=ids
            ):
                msg_id = ids[-1] if ids else FALLBACK_BLOCK_ID
                yield PublicEvent(
                    "done",
                    {
                        "type": "done",
                        "id": msg_id,
                        "turns": turns,
                        "response_source": response_source,
                    },
                )
            case ElicitationRequestFrame(
                id=elicitation_id,
                message=message,
                response_type=response_type,
                schema_json=schema_json,
            ):
                # schema_json is re-parsed into an object so the client
                # decodes the payload once instead of double-decoding an
                # embedded string; malformed degrades to {} rather than
                # raising, which would tear down the stream
                # mid-conversation.
                try:
                    schema: Any = json.loads(schema_json) if schema_json else {}
                except (json.JSONDecodeError, TypeError):
                    schema = {}
                yield PublicEvent(
                    "elicitation_request",
                    {
                        "type": "elicitation_request",
                        "id": elicitation_id,
                        "message": message,
                        "response_type": response_type,
                        "schema": schema,
                    },
                )
            case ElicitationResolvedFrame(id=elicitation_id, action=action):
                yield PublicEvent(
                    "elicitation_resolved",
                    {
                        "type": "elicitation_resolved",
                        "id": elicitation_id,
                        "action": action,
                    },
                )
            case (
                ToolCallFrame()
                | ToolCallDeltaFrame()
                | ToolResultFrame()
                | ToolProgressFrame()
                | ThoughtFrame()
                | ThoughtDeltaFrame()
                | TurnEndFrame()
            ):
                continue
            case _:
                assert_never(frame)


__all__ = [
    "EMPTY_TURN_RETRY_TEXT",
    "FALLBACK_BLOCK_ID",
    "MAX_ARG_BYTES",
    "MAX_RAW_BYTES",
    "MAX_RESULT_BYTES",
    "PublicEvent",
    "as_terminal_safe",
    "project",
    "public_allowlist",
    "with_status",
    "with_text_fallback",
]
