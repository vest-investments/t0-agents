"""Future-based elicitation bridge between an in-process FastMCP handler
and an out-of-process resolver (browser WebSocket in Phase E).

Phase D ships the skeleton: the forwarder's ``handler`` always returns
``ElicitResult(action="decline")``, matching :func:`auto_decline_handler`
behaviour, so non-streaming HTTP servers surface elicitation attempts as
explicit tool errors rather than hanging.  Phase E will complete the
public surface by adding ``forward()``, ``resolve()`` and ``wait_for()``
methods that bridge an ``asyncio.Future`` keyed on ``elicitation_id`` to
a remote channel (see ``02-open-questions.md`` — FastMCP Q2 and the
resolution in the Phase D log).

The Future dict is pre-allocated so Phase E doesn't break the public
import path; a developer writing a custom forwarder today can already
subclass :class:`ElicitationForwarder` and override ``handler``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastmcp.client.elicitation import ElicitResult


@dataclass(slots=True)
class ElicitationRequest:
    """A single pending elicitation awaiting an out-of-process answer.

    ``elicitation_id`` is the correlation key between the in-process
    handler and the remote resolver.  ``response_type`` mirrors the
    ``type[T] | None`` slot of the FastMCP
    :data:`~fastmcp.client.elicitation.ElicitationHandler` alias — it is
    ``None`` for binary (approval) elicitations and a concrete class for
    structured-data elicitations.  ``future`` is created lazily by
    Phase E's ``forward()`` (an event loop must be running), so Phase D
    leaves it optional.
    """

    elicitation_id: str
    message: str
    response_type: type[Any] | None
    params: Any
    future: asyncio.Future[ElicitResult[Any]] | None = None


class ElicitationForwarder:
    """Phase D skeleton — Phase E completes the remote resolver surface.

    The class exists today so that downstream callers and phase-E work
    can import it from a stable location.  In Phase D the ``handler``
    method unconditionally declines, preserving the "elicitation
    impossible over plain HTTP" contract documented in
    ``docs/serving.md``.

    Subclasses may override ``handler`` to implement a different
    default policy (for example an auto-approve variant for testing).
    """

    def __init__(self) -> None:
        # Keyed by ``elicitation_id``; populated by ``forward()`` (Phase E)
        # and drained by ``resolve()`` (Phase E).  Held on the forwarder so
        # Phase E's additions do not need to widen the constructor.
        self._pending: dict[str, ElicitationRequest] = {}

    async def handler(
        self,
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        """Default Phase D behaviour: decline every elicitation.

        Shape matches :data:`fastmcp.client.elicitation.ElicitationHandler`.
        """
        return ElicitResult(action="decline")
