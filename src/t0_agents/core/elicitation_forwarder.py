"""Future-based elicitation bridge between an in-process FastMCP handler
and an out-of-process resolver (typically a WebSocket in Phase E).

Phase D shipped the skeleton with a default ``handler`` that always
declined — non-streaming HTTP has no return channel, so elicitation is
surfaced as an explicit tool error.  Phase E completes the surface:

* ``__init__(*, on_request=None, default_timeout=300.0)`` — the
  ``on_request`` callback is how the forwarder hands a pending request off
  to the remote channel.  Keeping the argument optional preserves the
  Phase D contract (``ElicitationForwarder()`` still declines by default).
* ``handler(message, response_type, params, context)`` — the FastMCP-side
  entry point.  Builds an :class:`ElicitationRequest`, calls
  ``on_request(envelope)``, awaits the Future with a timeout, and returns
  the corresponding :class:`ElicitResult`.
* ``resolve(elicitation_id, payload)`` — drain one Future when the remote
  side replies.
* ``cancel_all(action="decline")`` — drain every pending Future, used on
  WS disconnect.

Each WS connection owns its own forwarder instance (one elicitation handler
per FastMCP ``Client``, per FastMCP Q3 resolution in
``02-open-questions.md``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from fastmcp.client.elicitation import ElicitResult

logger = logging.getLogger(__name__)

RequestEnvelope = dict[str, Any]
"""Shape handed to ``on_request``:

``{"type": "elicitation_request", "elicitation_id": ..., "message": ...,
"schema": ..., "response_type": ...}``.

The serving layer JSON-encodes this envelope and sends it over the WS.
"""

OnRequest = Callable[[RequestEnvelope], Awaitable[None]]


@dataclass(slots=True)
class ElicitationRequest:
    """A single pending elicitation awaiting an out-of-process answer.

    ``elicitation_id`` is the correlation key between the in-process
    handler and the remote resolver.  ``response_type`` mirrors the
    ``type[T] | None`` slot of the FastMCP
    :data:`~fastmcp.client.elicitation.ElicitationHandler` alias — it is
    ``None`` for binary (approval) elicitations and a concrete class for
    structured-data elicitations.  ``future`` is created inside
    :meth:`ElicitationForwarder.handler`, which ensures an event loop is
    running.
    """

    elicitation_id: str
    message: str
    response_type: type[Any] | None
    params: Any
    future: asyncio.Future[ElicitResult[Any]] | None = None


class ElicitationForwarder:
    """Bridges FastMCP's inline elicitation handler to a remote channel.

    Default (Phase D) behaviour — constructed without ``on_request`` —
    declines every elicitation so non-streaming HTTP servers surface
    elicitation attempts as tool errors rather than hanging.

    Streaming (Phase E) behaviour — constructed with ``on_request`` wired
    to ``ws.send_json`` — pushes a request envelope to the WS client,
    blocks on a Future, and resolves the Future from incoming WS frames
    via :meth:`resolve`.
    """

    def __init__(
        self,
        *,
        on_request: OnRequest | None = None,
        default_timeout: float = 300.0,
    ) -> None:
        if default_timeout <= 0:
            raise ValueError(
                f"default_timeout must be positive, got {default_timeout}"
            )
        self._pending: dict[str, ElicitationRequest] = {}
        self._on_request = on_request
        self._default_timeout = default_timeout

    @property
    def pending(self) -> dict[str, ElicitationRequest]:
        """Read-only view of currently-pending elicitations."""
        return self._pending

    async def handler(
        self,
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        """FastMCP-side entry point.

        Builds a pending request, invokes :attr:`on_request`, awaits the
        Future.  Auto-declines on:

        * Phase D default — ``on_request is None``.
        * timeout after :attr:`default_timeout` seconds.
        * task cancellation (e.g. WS disconnect cancels the surrounding
          streaming task, which cancels this coroutine).
        """
        if self._on_request is None:
            return ElicitResult(action="decline")

        elicitation_id = str(uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ElicitResult[Any]] = loop.create_future()
        request = ElicitationRequest(
            elicitation_id=elicitation_id,
            message=message,
            response_type=response_type,
            params=params,
            future=fut,
        )
        self._pending[elicitation_id] = request

        envelope: RequestEnvelope = {
            "type": "elicitation_request",
            "elicitation_id": elicitation_id,
            "message": message,
            "schema": getattr(params, "requestedSchema", None),
            "response_type": (
                response_type.__name__ if response_type is not None else None
            ),
        }

        try:
            await self._on_request(envelope)
        except Exception:
            logger.exception(
                "on_request failed for elicitation %s; auto-declining",
                elicitation_id,
            )
            self._pending.pop(elicitation_id, None)
            return ElicitResult(action="decline")

        try:
            return await asyncio.wait_for(fut, timeout=self._default_timeout)
        except TimeoutError:
            logger.warning(
                "Elicitation %s timed out after %.1fs; declining",
                elicitation_id,
                self._default_timeout,
            )
            return ElicitResult(action="decline")
        except asyncio.CancelledError:
            logger.info(
                "Elicitation %s cancelled (likely WS disconnect); declining",
                elicitation_id,
            )
            return ElicitResult(action="decline")
        finally:
            self._pending.pop(elicitation_id, None)

    async def resolve(
        self, elicitation_id: str, payload: dict[str, Any]
    ) -> None:
        """Resolve one pending elicitation from an incoming remote frame.

        ``payload`` is the remote side's JSON response.  Accepted keys:

        * ``action`` (``"accept" | "decline" | "cancel"``, default
          ``"accept"``).
        * ``content`` (dict; ignored when ``action`` is not ``"accept"``).

        Unknown ``elicitation_id`` is logged and ignored — the remote may
        have raced the timeout or an earlier :meth:`cancel_all`.
        """
        request = self._pending.get(elicitation_id)
        if request is None or request.future is None:
            logger.warning(
                "resolve() called for unknown elicitation_id=%s", elicitation_id
            )
            return
        if request.future.done():
            logger.debug(
                "resolve() called for already-settled elicitation_id=%s",
                elicitation_id,
            )
            return
        action = payload.get("action", "accept")
        if action not in ("accept", "decline", "cancel"):
            logger.warning(
                "resolve() got invalid action=%r for elicitation_id=%s; "
                "coercing to decline",
                action,
                elicitation_id,
            )
            action = "decline"
        content = payload.get("content") if action == "accept" else None
        request.future.set_result(
            ElicitResult(action=action, content=content)
        )

    async def cancel_all(
        self,
        *,
        action: Literal["accept", "decline", "cancel"] = "decline",
    ) -> None:
        """Drain every pending Future with ``action`` (default ``decline``).

        Called by the WS route on disconnect: any tool still inside
        ``ctx.elicit()`` observes the decline result and completes cleanly
        rather than hanging until the timeout.
        """
        for elicitation_id, request in list(self._pending.items()):
            if request.future is None or request.future.done():
                continue
            request.future.set_result(ElicitResult(action=action, content=None))
            logger.debug(
                "cancel_all() resolved elicitation_id=%s with action=%s",
                elicitation_id,
                action,
            )
