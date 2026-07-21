"""Future-based elicitation bridge between an in-process FastMCP handler
and an out-of-process resolver (typically a WebSocket).

An :class:`ElicitationForwarder` can run in two modes:

* **Auto-decline.** Constructed without ``on_request``, the forwarder's
  :meth:`~ElicitationForwarder.handler` declines every elicitation
  immediately.  This is the safe default for contexts that have no return
  channel for a human response (for example a non-streaming HTTP agent) —
  tool code inside ``ctx.elicit(...)`` observes an explicit decline
  rather than hanging.
* **Remote resolver.** Constructed with an ``on_request`` callback, the
  handler builds an :class:`ElicitationRequest`, invokes
  ``on_request(envelope)`` to hand the request off to the remote side
  (for example by sending a JSON frame over a WebSocket), and awaits an
  :class:`asyncio.Future` with a timeout.  The remote side resolves the
  future by calling :meth:`~ElicitationForwarder.resolve` with the
  incoming response payload.

The public surface is four methods plus an observer seam:

* ``__init__(*, on_request=None, default_timeout=300.0, observer=None)``.
  ``observer`` (rebindable via the ``observer`` attribute) witnesses the
  request/resolve lifecycle with the real ``elicitation_id`` — the seam
  glow uses to emit in-stream elicitation events whose id a remote
  resolver can answer.
* ``handler(message, response_type, params, context)`` — the FastMCP-side
  entry point.
* ``resolve(elicitation_id, payload)`` — drain one Future when the remote
  side replies.
* ``cancel_all(action="decline")`` — drain every pending Future, used on
  disconnect.

Each remote channel (for example each WebSocket connection) owns its own
forwarder instance: ``fastmcp.Client`` binds a single
``elicitation_handler`` per client, so per-session isolation requires a
per-session client — and therefore a per-session forwarder.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from fastmcp.client.elicitation import ElicitResult

logger = logging.getLogger(__name__)


@runtime_checkable
class ElicitationObserver(Protocol):
    """Read-only witness of a forwarder's elicitation lifecycle.

    ``requested`` fires with the exact envelope handed to ``on_request``
    (so the observer sees the real ``elicitation_id`` — the id a remote
    resolver answers with); ``resolved`` fires once per elicitation with
    the settled action (``"accept"`` / ``"decline"`` / ``"cancel"``),
    on every path — remote resolution, timeout, cancellation, and
    ``on_request`` failure alike.  Both are awaited on the elicitation
    path (they typically push onto an event buffer).  Observers must not
    raise; an exception is logged and the elicitation proceeds — a
    broken observability hook must never break a human-in-the-loop
    confirm.
    """

    async def requested(self, envelope: RequestEnvelope) -> None: ...

    async def resolved(self, elicitation_id: str, action: str) -> None: ...

RequestEnvelope = dict[str, Any]
"""Shape handed to ``on_request``:

``{"type": "elicitation_request", "elicitation_id": ..., "message": ...,
"schema": ..., "response_type": ...}``.

The serving layer JSON-encodes this envelope and sends it over the WS.
"""

OnRequest = Callable[[RequestEnvelope], Awaitable[None]]


_SCHEMA_TYPE_NAMES: dict[str, str] = {
    "boolean": "bool",
    "integer": "int",
    "number": "float",
    "string": "str",
}


def _response_type_from_schema(
    schema: Any, response_type: type[Any] | None
) -> str | None:
    """Stable ``response_type`` label for the request envelope.

    ``None`` for a binary (approval) elicitation.  For a scalar
    elicitation FastMCP wraps the value as ``{"value": {"type":
    <json-type>}}`` under an internally-generated class whose
    ``__name__`` (``"ScalarElicitationType"``) is unstable by design;
    the JSON-schema ``type`` the ``schema`` field already carries is the
    stable contract, so read it and map it to the Python name.  Falls
    back to the schema ``title`` (structured models), then the type's
    ``__name__`` (or ``str()`` for an exotic alias with no ``__name__``).
    """
    if response_type is None:
        return None
    if isinstance(schema, Mapping):
        props = schema.get("properties")
        value = props.get("value") if isinstance(props, Mapping) else None
        if isinstance(value, Mapping):
            value_type = value.get("type")
            if isinstance(value_type, str):
                return _SCHEMA_TYPE_NAMES.get(value_type, value_type)
        title = schema.get("title")
        if isinstance(title, str):
            return title
    return getattr(response_type, "__name__", None) or str(response_type)


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

    Default behaviour — constructed without ``on_request`` — declines
    every elicitation so servers with no return channel surface
    elicitation attempts as explicit errors rather than hanging.

    Remote-resolver behaviour — constructed with ``on_request`` wired to
    a send function (for example ``ws.send_json``) — pushes a request
    envelope to the remote side, blocks on a Future, and resolves the
    Future from incoming frames via :meth:`resolve`.
    """

    def __init__(
        self,
        *,
        on_request: OnRequest | None = None,
        default_timeout: float = 300.0,
        observer: ElicitationObserver | None = None,
    ) -> None:
        if default_timeout <= 0:
            raise ValueError(
                f"default_timeout must be positive, got {default_timeout}"
            )
        self._pending: dict[str, ElicitationRequest] = {}
        self._on_request = on_request
        self._default_timeout = default_timeout
        # Public and rebindable: glow binds an observer when a forwarder
        # is attached to a conversation, so in-stream elicitation events
        # carry the SAME id a remote resolver answers with.  One observer
        # per forwarder — a forwarder shared across conversations keeps
        # only the last binding (matching its one-channel contract).
        self.observer: ElicitationObserver | None = observer

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

        * no remote resolver wired — ``on_request is None``.
        * timeout after :attr:`default_timeout` seconds.
        * task cancellation (for example a WebSocket disconnect cancels
          the surrounding streaming task, which cancels this coroutine).
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

        schema = getattr(params, "requestedSchema", None)
        envelope: RequestEnvelope = {
            "type": "elicitation_request",
            "elicitation_id": elicitation_id,
            "message": message,
            "schema": schema,
            "response_type": _response_type_from_schema(schema, response_type),
        }

        await self._notify_requested(envelope)

        try:
            await self._on_request(envelope)
        except Exception:
            logger.exception(
                "on_request failed for elicitation %s; auto-declining",
                elicitation_id,
            )
            self._pending.pop(elicitation_id, None)
            await self._notify_resolved(elicitation_id, "decline")
            return ElicitResult(action="decline")

        action = "decline"
        try:
            result = await asyncio.wait_for(fut, timeout=self._default_timeout)
            action = result.action
            return result
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
            await self._notify_resolved(elicitation_id, action)

    async def _notify_requested(self, envelope: RequestEnvelope) -> None:
        if self.observer is None:
            return
        try:
            await self.observer.requested(envelope)
        except (Exception, asyncio.CancelledError):
            # CancelledError included: ``handler`` swallows cancellation by
            # contract (returns a decline), so a cancelled notification in
            # its ``finally`` must not resurrect the cancellation — the
            # event is best-effort on that path.
            logger.warning(
                "elicitation observer.requested raised; continuing",
                exc_info=True,
            )

    async def _notify_resolved(self, elicitation_id: str, action: str) -> None:
        if self.observer is None:
            return
        try:
            await self.observer.resolved(elicitation_id, action)
        except (Exception, asyncio.CancelledError):
            logger.warning(
                "elicitation observer.resolved raised; continuing",
                exc_info=True,
            )

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
