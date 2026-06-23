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

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from t0_agents.core.conversation import ConversationUsage
    from t0_agents.core.types import RunUsage, Usage


class ChatRequest(BaseModel):
    """Body for ``POST /chat``.

    ``tags`` merge on top of the agent's configured tags for observability
    — they do **not** replace ``namespace`` or the deployment label.
    """

    message: str
    episode_id: str | None = None
    tags: dict[str, str] | None = None


class FootprintUsage(BaseModel):
    """The latest non-empty inference's token usage — a *level*, not a sum.

    Mirrors :class:`~t0_agents.core.types.Usage`: the prompt size of the most
    recent turn that actually hit the model (cache-served zero turns are
    skipped), i.e. the number to compare against the model's context window
    when deciding whether to compact.  The two ``provider_cache_*`` fields stay
    ``None`` (lossless) when the provider does not report them (e.g. Groq);
    ``cost`` is ``None`` when the call was unpriced.
    """

    input_tokens: int
    output_tokens: int
    provider_cache_read_input_tokens: int | None = None
    provider_cache_write_input_tokens: int | None = None
    cost: float | None = None


class CumulativeUsage(BaseModel):
    """Cumulative billing totals summed across every inference of a Run.

    Mirrors :class:`~t0_agents.core.types.RunUsage`.  This is the billing /
    budget number — it *sums* each turn's prompt, double-counting the growing
    prefix, so it is deliberately distinct from :class:`FootprintUsage`.
    Includes delegated sub-agent billing (the cost middleware folds children
    into the parent Run).  ``cost_is_priced`` distinguishes a genuine
    ``$0.00`` from "no pricing configured".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    provider_cache_read_input_tokens: int = 0
    provider_cache_write_input_tokens: int = 0
    cost_is_priced: bool = False


class SessionUsage(BaseModel):
    """Per-connection running total across one WebSocket conversation.

    Present only on the streaming ``done`` frame (the WS path is
    multi-invoke).  ``total`` accumulates billing across every turn of the
    connection's episode; ``footprint`` is the latest turn's level (carried
    forward over a footprint-less turn); ``runs_recorded`` counts the folded
    Runs.
    """

    total: CumulativeUsage
    footprint: FootprintUsage | None = None
    runs_recorded: int


class UsageBlock(BaseModel):
    """Additive ``usage`` block surfaced on every serving path.

    Separates the two numbers the live-usage feature exists to distinguish:
    :attr:`footprint` (current prompt size — the compaction signal) and
    :attr:`cumulative` (billing, summed).  :attr:`session` is set only on the
    WebSocket ``done`` frame.  The block is optional and additive — every
    pre-existing response / frame key is unchanged.
    """

    footprint: FootprintUsage | None = None
    cumulative: CumulativeUsage
    session: SessionUsage | None = None


def _footprint_model(usage: Usage) -> FootprintUsage:
    """Project a :class:`~t0_agents.core.types.Usage` onto the wire model."""
    return FootprintUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        provider_cache_read_input_tokens=usage.provider_cache_read_input_tokens,
        provider_cache_write_input_tokens=usage.provider_cache_write_input_tokens,
        cost=usage.cost,
    )


def _cumulative_model(usage: RunUsage) -> CumulativeUsage:
    """Project a :class:`~t0_agents.core.types.RunUsage` onto the wire model."""
    return CumulativeUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=usage.cost,
        provider_cache_read_input_tokens=usage.provider_cache_read_input_tokens,
        provider_cache_write_input_tokens=usage.provider_cache_write_input_tokens,
        cost_is_priced=usage.cost_is_priced,
    )


def build_usage_block(
    *,
    footprint: Usage | None,
    cumulative: RunUsage | None,
    session: ConversationUsage | None = None,
) -> UsageBlock | None:
    """Assemble the wire ``usage`` block from a Run's usage accessors.

    Returns ``None`` when ``cumulative`` is ``None`` (the Run produced no
    inference) so the response simply carries ``"usage": null``.  ``footprint``
    may be ``None`` independently (a whole-response cache hit, or a failure
    before the first inference) and is surfaced as ``null``.  ``session`` — the
    per-connection :class:`~t0_agents.core.conversation.ConversationUsage` — is
    included only on the WebSocket ``done`` frame; its level honours the
    carry-forward semantics of
    :meth:`~t0_agents.core.conversation.ConversationUsage.record`.
    """
    if cumulative is None:
        return None
    session_block: SessionUsage | None = None
    if session is not None:
        session_footprint = session.context_footprint()
        session_block = SessionUsage(
            total=_cumulative_model(session.total),
            footprint=(
                _footprint_model(session_footprint)
                if session_footprint is not None
                else None
            ),
            runs_recorded=session.runs_recorded,
        )
    return UsageBlock(
        footprint=_footprint_model(footprint) if footprint is not None else None,
        cumulative=_cumulative_model(cumulative),
        session=session_block,
    )


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``.

    Mirrors :class:`~t0_agents.core.types.AgentResult` 1:1 with
    ``guardrail_log`` converted from a list of dataclass instances to a
    list of plain dicts.  ``usage`` is an additive, optional
    :class:`UsageBlock` (``null`` when the run produced no inference) — every
    other field predates it and is unchanged.
    """

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = Field(default_factory=list)
    turns: int = 0
    response_source: str = "text"
    guardrail_log: list[dict[str, Any]] = Field(default_factory=list)
    usage: UsageBlock | None = None


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
