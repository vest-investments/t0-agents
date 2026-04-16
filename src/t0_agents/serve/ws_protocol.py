"""Pydantic v2 models for the WebSocket wire protocol.

Incoming frames (client → server) are validated here; outgoing frames
(server → client) are serialised from :class:`~t0_agents.stream.events.AgentEvent`
dataclasses via :func:`dataclasses.asdict` at the route boundary.  The only
outbound shape that does **not** derive from an ``AgentEvent`` is
:class:`WSElicitationRequest` — FastMCP's ``ctx.elicit()`` is bridged into
the WS channel through the :class:`~t0_agents.core.elicitation_forwarder.ElicitationForwarder`
and emerges here as a distinct envelope so browser clients can tell tool
chatter apart from HITL prompts.

Discriminator: every frame carries a ``type`` field; incoming frames use a
discriminated union so Pydantic picks the right model automatically.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class WSUserMessage(BaseModel):
    """Client → server: begin a new streaming turn.

    When a stream is already in flight the server cancels it before
    starting this one — Phase E is one-stream-per-connection by design
    (see ``docs/streaming.md`` 'Backpressure').
    """

    type: Literal["user_message"]
    message: str
    episode_id: str | None = None
    tags: dict[str, str] | None = None


class WSElicitationResponse(BaseModel):
    """Client → server: resolve a pending elicitation.

    ``payload`` mirrors the shape the FastMCP
    :class:`~fastmcp.client.elicitation.ElicitResult` constructor accepts:

    * ``action``: one of ``"accept"``, ``"decline"``, ``"cancel"``
      (default ``"accept"``).
    * ``content``: scalar-or-dict payload (required when
      ``action == "accept"``; ignored otherwise).  For scalar response
      types FastMCP expects ``{"value": ...}`` wrapping — see
      ``docs/elicitation.md``.
    """

    type: Literal["elicitation_response"]
    elicitation_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


WSIncoming = Annotated[
    WSUserMessage | WSElicitationResponse,
    Field(discriminator="type"),
]
"""Discriminated union for every frame the server accepts."""


class WSElicitationRequest(BaseModel):
    """Server → client: FastMCP tool is awaiting an elicitation response.

    The browser / TUI renders a form based on ``schema`` and posts back a
    :class:`WSElicitationResponse` using the same ``elicitation_id``.
    """

    type: Literal["elicitation_request"] = "elicitation_request"
    elicitation_id: str
    message: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    response_type: str | None = None

    model_config = {"populate_by_name": True}


class WSErrorFrame(BaseModel):
    """Server → client: protocol or dispatch-level error.

    Agent-run errors (``MaxTurnsExceededError``, ``GuardrailBlockedError``)
    surface through this frame rather than closing the WS so the browser
    can display a human-readable message.  The connection stays open so
    the user can retry.
    """

    type: Literal["ws_error"] = "ws_error"
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
