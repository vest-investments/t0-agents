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
from typing import Any

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
    """Assemble the Phase E streaming routes against a :class:`SessionFactory`.

    Mounted under the same app as the Phase D router — callers that set
    ``streaming=True`` on :func:`create_app` get both route groups.
    """
    router = APIRouter()

    # -- SSE ---------------------------------------------------------------

    @router.get("/chat/stream")
    async def chat_stream(
        message: str,
        episode_id: str | None = None,
    ) -> EventSourceResponse:
        """Stream :class:`AgentEvent` frames as SSE events.

        The response sets ``Content-Type: text/event-stream`` via
        :class:`sse_starlette.sse.EventSourceResponse`.  Each frame uses
        the standard SSE envelope::

            event: text_delta
            data: {"id": "...", "delta": "...", "type": "text_delta"}

        Clients decode with ``EventSource`` in the browser or any
        SSE-aware HTTP library.
        """
        return EventSourceResponse(
            _sse_event_stream(session_factory, message, episode_id)
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
) -> AsyncIterator[dict[str, str]]:
    """Yield ``EventSourceResponse``-shaped dicts for the SSE endpoint."""
    try:
        async with session_factory.session() as agent:
            async for event in agent.stream(message, episode_id=episode_id):
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


def _to_sse_frame(event: AgentEvent) -> dict[str, str]:
    return {
        "event": event.type,
        "data": json.dumps(_event_to_json(event)),
    }


def _event_to_json(event: AgentEvent) -> dict[str, Any]:
    """Convert an event dataclass to a JSON-serialisable dict."""
    return dataclasses.asdict(event)


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
        async for event in agent.stream(
            frame.message,
            episode_id=frame.episode_id,
        ):
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
