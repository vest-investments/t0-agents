"""SSE + WebSocket routes for :meth:`Agent.stream`.

Two endpoints:

* **``GET /chat/stream``** (SSE) — push-only delta stream.  Query-string
  driven so the browser can open it with ``new EventSource(...)``.  No
  bidirectional channel, so any tool that calls ``ctx.elicit()`` ends up
  hitting the forwarder's auto-decline default (no ``on_request`` is
  wired) and the tool surfaces an elicitation error in a
  :class:`~t0_agents.stream.events.ToolResultEvent`.
* **``WS /chat/ws``** (WebSocket) — bidirectional JSON frames.  Elicitation
  is piped through a per-connection
  :class:`~t0_agents.core.elicitation_forwarder.ElicitationForwarder` so
  the browser can resolve prompts in-flight.

Concurrency model for WS (single-conn, single-stream-at-a-time):

* A **receiver task** (the route body) loops on ``ws.receive_json``.
* A **stream task** is spawned on every ``user_message`` frame.  When a
  new ``user_message`` arrives while an earlier stream is still active,
  the earlier stream is cancelled — one stream per connection.
* ``elicitation_response`` frames resolve the forwarder's Future, which
  unblocks the stream task inside its ``ctx.elicit()`` await.

On :class:`fastapi.WebSocketDisconnect` the receiver exits, the stream
task is cancelled, and ``forwarder.cancel_all()`` drains outstanding
elicitations with ``action="decline"``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    RequestEnvelope,
)
from t0_agents.core.exceptions import (
    GuardrailBlockedError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.serve.session import SessionFactory
from t0_agents.serve.ws_protocol import (
    WSElicitationResponse,
    WSErrorFrame,
    WSUserMessage,
)
from t0_agents.stream.events import AgentEvent, ErrorEvent

logger = logging.getLogger(__name__)


def build_streaming_router(*, session_factory: SessionFactory) -> APIRouter:
    """Assemble the streaming routes against a :class:`SessionFactory`.

    Mounted alongside the non-streaming router — callers that set
    ``streaming=True`` on :func:`create_app` get both route groups.
    """
    router = APIRouter()

    # -- SSE ---------------------------------------------------------------

    @router.get("/chat/stream")
    async def chat_stream(
        message: str,
        episode_id: str | None = None,
        tags: str | None = None,
    ) -> EventSourceResponse:
        """Stream :class:`AgentEvent` frames as SSE events.

        The response sets ``Content-Type: text/event-stream`` via
        :class:`sse_starlette.sse.EventSourceResponse`.  Each frame uses
        the standard SSE envelope::

            event: text_delta
            data: {"id": "...", "delta": "...", "type": "text_delta"}

        ``tags`` is an optional JSON-encoded ``dict[str, str]`` query
        parameter (e.g. ``?tags=%7B%22env%22%3A%22staging%22%7D``) that
        shallow-merges onto the agent's configured tags for parity with
        ``POST /chat``'s request body.  Invalid JSON or a non-object
        decoded value surfaces as the first SSE frame — an
        :class:`~t0_agents.stream.events.ErrorEvent` with
        ``detail.kind == "invalid_tags"`` — and the stream terminates.

        Clients decode with ``EventSource`` in the browser or any
        SSE-aware HTTP library.
        """
        return EventSourceResponse(
            _sse_event_stream(session_factory, message, episode_id, tags)
        )

    # -- WebSocket ---------------------------------------------------------

    @router.websocket("/chat/ws")
    async def chat_ws(ws: WebSocket) -> None:
        await ws.accept()

        async def send_envelope(envelope: RequestEnvelope) -> None:
            await ws.send_json(envelope)

        forwarder = ElicitationForwarder(on_request=send_envelope)

        async with session_factory.session(
            elicitation_handler=forwarder.handler,
        ) as agent:
            active_stream: asyncio.Task[None] | None = None
            try:
                while True:
                    raw = await ws.receive_json()
                    try:
                        envelope = _validate_ws_incoming(raw)
                    except ValidationError as exc:
                        await ws.send_json(
                            WSErrorFrame(
                                message="invalid WS frame",
                                detail={"errors": exc.errors()},
                            ).model_dump(),
                        )
                        continue

                    if isinstance(envelope, WSUserMessage):
                        active_stream = await _start_stream_task(
                            ws=ws,
                            agent=agent,
                            frame=envelope,
                            previous=active_stream,
                        )
                    elif isinstance(envelope, WSElicitationResponse):
                        await forwarder.resolve(
                            envelope.elicitation_id, envelope.payload,
                        )
            except WebSocketDisconnect:
                logger.debug("WS /chat/ws disconnect; tearing down session")
            finally:
                if active_stream is not None and not active_stream.done():
                    active_stream.cancel()
                    try:
                        await active_stream
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception("Stream task raised during teardown")
                await forwarder.cancel_all(action="decline")

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _sse_event_stream(
    session_factory: SessionFactory,
    message: str,
    episode_id: str | None,
    raw_tags: str | None,
) -> AsyncIterator[dict[str, str]]:
    """Yield ``EventSourceResponse``-shaped dicts for the SSE endpoint."""
    try:
        parsed_tags = _parse_tags_query(raw_tags)
    except ValueError as exc:
        yield _to_sse_frame(
            ErrorEvent(message=str(exc), detail={"kind": "invalid_tags"}),
        )
        return
    try:
        async with session_factory.session() as agent:
            # v1.0: invoke().events() bypasses the agent.stream
            # DeprecationWarning from this internal call-site.
            async for event in agent.invoke(
                message, episode_id=episode_id, tags=parsed_tags,
            ).events():
                yield _to_sse_frame(event)
    except MaxTurnsExceededError as exc:
        yield _to_sse_frame(
            ErrorEvent(message=str(exc), detail={"kind": "max_turns"}),
        )
    except GuardrailBlockedError as exc:
        yield _to_sse_frame(
            ErrorEvent(message=str(exc), detail={"kind": "guardrail_blocked"}),
        )
    except T0AgentsError as exc:
        yield _to_sse_frame(
            ErrorEvent(message=str(exc), detail={"kind": type(exc).__name__}),
        )


def _parse_tags_query(raw: str | None) -> dict[str, str] | None:
    """Decode the ``?tags=<json>`` SSE query parameter.

    ``None`` and empty-string decode to ``None`` (no per-request override).
    Anything else must decode to a JSON object whose keys and values are
    all strings; other shapes raise :class:`ValueError` so the caller can
    surface an ``ErrorEvent`` with ``detail.kind == "invalid_tags"``.
    """
    if raw is None or raw == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tags query param is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"tags query param must decode to a JSON object, "
            f"got {type(parsed).__name__}",
        )
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                "tags query param must have only string keys and string values",
            )
    return cast(dict[str, str], parsed)


def _to_sse_frame(event: AgentEvent) -> dict[str, str]:
    return {
        "event": event.type,
        "data": json.dumps(_event_to_json(event)),
    }


def _event_to_json(event: AgentEvent) -> dict[str, Any]:
    """Convert an event dataclass to a JSON-serialisable dict.

    DoneEvent gets a custom shape: ``result`` is projected through the
    v0.12 read-side keys (``response``, ``parsed``, ``parse_strategy``,
    ``json_parse_attempts``, ``function_kind``) so downstream
    HTTP / WS clients keep working byte-identical across the v1.0
    refactor (chat byte-identity invariant §2.3).
    """
    payload = dataclasses.asdict(event)
    if event.type == "done" and "result" in payload:
        result_obj = event.result
        payload["result"] = {
            "response": result_obj.response,
            "episode_id": result_obj.episode_id,
            "inference_ids": list(result_obj.inference_ids),
            "turns": result_obj.turns,
            "response_source": result_obj.response_source,
            "guardrail_log": [
                dataclasses.asdict(entry) for entry in result_obj.guardrail_log
            ],
            "function_kind": result_obj.function_kind,
            "parsed": result_obj.parsed,
            "parse_strategy": result_obj.parse_strategy,
            "json_parse_attempts": result_obj.json_parse_attempts,
        }
    return payload


def _validate_ws_incoming(raw: Any) -> WSUserMessage | WSElicitationResponse:
    """Parse a raw JSON frame into the tagged incoming union.

    Using ``RootModel`` would work too, but for a 2-variant discriminated
    union the explicit isinstance branches keep the route logic readable.
    """
    if not isinstance(raw, dict):
        raise ValidationError.from_exception_data(
            "WSIncoming",
            [{"type": "dict_type", "loc": (), "input": raw}],
        )
    kind = raw.get("type")
    if kind == "user_message":
        return WSUserMessage.model_validate(raw)
    if kind == "elicitation_response":
        return WSElicitationResponse.model_validate(raw)
    raise ValidationError.from_exception_data(
        "WSIncoming",
        [
            {
                "type": "literal_error",
                "loc": ("type",),
                "input": kind,
                "ctx": {"expected": "'user_message' or 'elicitation_response'"},
            }
        ],
    )


async def _start_stream_task(
    *,
    ws: WebSocket,
    agent: Any,
    frame: WSUserMessage,
    previous: asyncio.Task[None] | None,
) -> asyncio.Task[None]:
    """Cancel a previous stream (if any) then launch a new one."""
    if previous is not None and not previous.done():
        previous.cancel()
        try:
            await previous
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Previous stream task raised on cancellation")
    return asyncio.create_task(_drive_stream(ws, agent, frame))


async def _drive_stream(
    ws: WebSocket, agent: Any, frame: WSUserMessage,
) -> None:
    """Stream events for one ``user_message`` frame back to the WS client."""
    try:
        # v1.0: invoke().events() — no DeprecationWarning from this
        # internal call-site.
        async for event in agent.invoke(
            frame.message,
            episode_id=frame.episode_id,
            tags=frame.tags,
        ).events():
            await ws.send_json(_event_to_json(event))
    except asyncio.CancelledError:
        raise
    except MaxTurnsExceededError as exc:
        await _safe_send(
            ws,
            WSErrorFrame(
                message=str(exc), detail={"kind": "max_turns"}
            ).model_dump(),
        )
    except GuardrailBlockedError as exc:
        await _safe_send(
            ws,
            WSErrorFrame(
                message=str(exc), detail={"kind": "guardrail_blocked"}
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Stream task crashed")
        await _safe_send(
            ws,
            WSErrorFrame(
                message=str(exc), detail={"kind": type(exc).__name__}
            ).model_dump(),
        )


async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Send a frame, swallow WS disconnect errors (client already gone)."""
    try:
        await ws.send_json(payload)
    except Exception:
        logger.debug("WS send failed (client likely disconnected)", exc_info=True)
