"""Pydantic v2 request/response models for the HTTP surface.

Kept intentionally close to :class:`~t0_agents.core.types.AgentResult`
and :class:`~t0_agents.core.config.AgentConfig` so there is a one-to-one
mapping between in-process agent state and what a caller sees over the
wire.  ``guardrail_log`` is serialized from
:class:`~t0_agents.core.types.GuardrailLogEntry` dataclasses via
``dataclasses.asdict`` at the route boundary — see
:func:`t0_agents.serve.routes.build_router`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for ``POST /chat``.

    ``tags`` merge on top of the agent's configured tags for observability
    — they do **not** replace ``namespace`` or the deployment label.
    """

    message: str
    episode_id: str | None = None
    tags: dict[str, str] | None = None


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``.

    Mirrors :class:`~t0_agents.core.types.AgentResult` 1:1 with
    ``guardrail_log`` converted from a list of dataclass instances to a
    list of plain dicts.
    """

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = Field(default_factory=list)
    turns: int = 0
    response_source: str = "text"
    guardrail_log: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Body for ``POST /feedback``.

    Fields match the six kwargs of
    :meth:`tensorzero.AsyncTensorZeroGateway.feedback`; the caller is
    responsible for pre-declaring ``metric_name`` in ``tensorzero.toml``
    (see ``docs/guardrails.md`` for the metrics-vs-feedback contract).
    """

    metric_name: str
    value: Any
    inference_id: str | None = None
    episode_id: str | None = None
    tags: dict[str, str] | None = None
    dryrun: bool | None = None


class FeedbackResponse(BaseModel):
    """Response body for ``POST /feedback``."""

    feedback_id: str


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Values are free-form strings so subclasses of the router can report
    richer states (``"degraded"``, ``"starting"``, ...) without a schema
    change.  The default implementation emits ``"ok"``, ``"down"``, or
    ``"per-session"`` — the last is a healthy state meaning "each
    request builds its own client".
    """

    gateway: str
    tool_provider: str


class InfoResponse(BaseModel):
    """Response body for ``GET /info``.

    Captures the non-secret subset of :class:`AgentConfig`.  ``has_memory``
    is a boolean rather than the memory backend class so deployments can
    swap implementations without leaking the class name.  Configured
    guardrails are reported by their ``name`` attribute only.
    """

    name: str
    function_name: str
    namespace: str | None = None
    tags: dict[str, str] | None = None
    has_memory: bool
    guardrails: list[str] = Field(default_factory=list)
