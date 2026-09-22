from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

from tensorzero import ExtraBody

from t0_agents.core.output import OutputDiagnostics
from t0_agents.core.structured import ParseStrategy

# The SDK's generated ExtraBody variants are not Mapping subclasses: narrowing this
# to Mapping type-checks here and rejects every SDK-typed patch at the caller.
ExtraBodyPatch: TypeAlias = ExtraBody | Mapping[str, Any]


@runtime_checkable
class ToolProgressHandler(Protocol):
    """Per-call progress relay: ``(progress, total, message)``.

    Structurally identical to fastmcp's ``ProgressHandler`` /
    ``mcp.shared.session.ProgressFnT`` — mirrored here (async ``__call__``
    with the same keyword-passable parameter names) so ``core`` stays
    fastmcp-import-free while ``fastmcp.Client`` still satisfies
    :class:`ToolProvider` in both directions.  A provider that can observe
    server-side progress notifications forwards each one to this handler;
    providers that cannot simply ignore the kwarg.
    """

    async def __call__(
        self, progress: float, total: float | None, message: str | None
    ) -> None: ...


@runtime_checkable
class ToolProvider(Protocol):
    """Structural type for tool discovery and execution.

    Matches the slice of ``fastmcp.Client`` that the agent loop uses:

    - ``list_tools()`` — discover available tools at loop start.
    - ``call_tool(name, arguments, *, timeout=..., progress_handler=...)``
      — execute a single tool, optionally relaying progress notifications.

    Any object implementing these two async methods satisfies the protocol,
    enabling testing with fakes and future non-FastMCP providers.  Wrapper
    providers must forward ``progress_handler`` to their inner provider or
    progress dies at the wrapper boundary.
    """

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        progress_handler: ToolProgressHandler | None = None,
    ) -> Any: ...


@runtime_checkable
class SupportsServerAttribution(Protocol):
    """Optional capability: name the server that owns a public tool name.

    A multi-server :class:`ToolProvider` implements this so per-call
    observability can record which mounted server a tool call hit.
    ``server_for`` is a synchronous name lookup with no I/O — never a
    network round-trip.  Returns ``None`` for a name with no attribution:
    an unprefixed (``prefix=False``) or single local mount has no
    namespace to report.  Providers that cannot attribute a server (a bare
    ``fastmcp.Client``) simply omit the method; callers probe with
    ``isinstance`` and fall back to ``None``.
    """

    def server_for(self, name: str) -> str | None: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Repository-style memory interface.

    The ``Agent`` calls ``load`` before every run and ``save`` after.
    ``search`` is optional for the core loop but required by the protocol
    so that custom backends expose a uniform query surface.

    Implementations return ``MemoryRecord`` instances (defined in
    ``t0_agents.memory.base``).  The protocol uses ``Any`` to avoid a
    circular import between ``core/`` and ``memory/``.
    """

    async def load(self, episode_id: str) -> list[Any]: ...

    async def save(self, record: Any) -> None: ...

    async def search(self, query: str, *, limit: int = 5) -> list[Any]: ...


@runtime_checkable
class Guardrail(Protocol):
    """Hook surface invoked by ``Agent.invoke()`` and ``GuardedToolProvider``.

    Four lifecycle hooks fire around each run and each tool call.  All
    implementations expose a human-readable ``name`` for logging and for
    ``GuardrailLogEntry.guardrail_name``.  The protocol uses ``Any`` for
    context and tool-call arguments to avoid a circular import between
    ``core/`` and ``guardrails/`` — the concrete types live in
    ``t0_agents.guardrails.base.GuardrailContext`` and
    ``tensorzero.ToolCall``.

    Return semantics of ``before_tool``:

    - ``None`` → pass through; the inner ``call_tool`` runs.
    - ``str``  → short-circuit; the string becomes the tool result the
      LLM sees next turn (wrapped in a single ``"ERROR: "`` prefix by the
      existing adapter serializer).
    - Raising ``GuardrailBlockedError`` (or any ``T0AgentsFatalError``) →
      hard-abort the run; the exception propagates past the loop.

    ``after_run`` always fires, even when the run aborted before producing
    an :class:`AgentResult` (for example a ``GuardrailBlockedError`` from
    ``before_run``, an upstream ``TensorZeroError``, or cancellation of a
    streaming turn).  In those cases ``result`` is ``None`` — implementations
    must tolerate the ``None`` branch rather than dereferencing fields
    blindly.  See ``docs/guardrails.md`` for the failure-mode matrix.
    """

    name: str

    async def before_run(self, ctx: Any) -> None: ...

    async def before_tool(self, ctx: Any, tool_call: Any) -> str | None: ...

    async def after_tool(
        self, ctx: Any, tool_call: Any, result_str: str
    ) -> None: ...

    async def after_run(
        self, ctx: Any, result: AgentResult[Any] | None
    ) -> None: ...


@dataclass(slots=True)
class GuardrailLogEntry:
    """One guardrail hook activation recorded during ``Agent.invoke()``.

    Populated by ``GuardedToolProvider`` (for ``before_tool`` / ``after_tool``)
    and by ``Agent.invoke`` itself (for ``before_run`` / ``after_run``).  Attached
    to :class:`AgentResult.guardrail_log` after every run (including fatal
    aborts) so callers can emit per-outcome ``t0.feedback(...)`` manually —
    TensorZero requires metrics to be pre-declared in ``tensorzero.toml``,
    so the framework does not auto-emit.
    """

    guardrail_name: str
    hook: Literal["before_run", "before_tool", "after_tool", "after_run"]
    kind: Literal["pass", "rejection", "block", "modification"]
    tool_name: str | None = None
    detail: str = ""


T = TypeVar("T")

# Sentinel — distinguishes "not passed" from "explicitly passed as None".
# Falls back into runtime branching in :meth:`AgentResult.__init__`; mypy
# narrows via the overload pair so callers never observe the sentinel.
_UNSET: Any = object()


@dataclass(init=False)
class AgentResult(Generic[T]):
    """Typed result of an agent run.

    v1.0 made :class:`AgentResult` generic in ``T`` so single-shot
    output modes (:class:`~t0_agents.core.outputs.JsonOutput`,
    :class:`~t0_agents.core.outputs.StructuredOutput`,
    :class:`~t0_agents.core.outputs.ToolCallOutput`) can express the
    statically-typed shape of their materialised result.  ``T = str``
    for chat (the boundary assistant text), ``T = dict[str, Any]`` for
    JSON, ``T = M`` for ``StructuredOutput[M]`` (Pydantic), and
    ``T = list[tensorzero.ToolCall]`` for the router pattern.

    The five v0.12 read-side fields (``response``, ``parsed``,
    ``parse_strategy``, ``json_parse_attempts``, ``function_kind``) are
    preserved as :func:`property` aliases — Tier 0 byte-identity per
    integration contract §2.4.  Write-side, ``value`` is canonical;
    ``response=`` remains accepted as a constructor alias because
    :mod:`mcp.loop` (invariant §2.1) constructs the chat result with
    ``AgentResult(response=text, ...)``.

    Diagnostics (``diagnostics: OutputDiagnostics | None``) carries the
    parse strategy, retry attempts, and raw text that the JSON path
    needed in v0.12.  Chat results leave ``diagnostics=None`` and the
    properties fall back to the v0.12 defaults.
    """

    value: T
    episode_id: str | None
    inference_ids: list[str]
    turns: int
    response_source: Literal["text", "thought"]
    guardrail_log: list[GuardrailLogEntry]
    diagnostics: OutputDiagnostics | None

    @overload
    def __init__(
        self,
        value: T,
        *,
        episode_id: str | None = ...,
        inference_ids: list[str] | None = ...,
        turns: int = ...,
        response_source: Literal["text", "thought"] = ...,
        guardrail_log: list[GuardrailLogEntry] | None = ...,
        diagnostics: OutputDiagnostics | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        response: str,
        episode_id: str | None = ...,
        inference_ids: list[str] | None = ...,
        turns: int = ...,
        response_source: Literal["text", "thought"] = ...,
        guardrail_log: list[GuardrailLogEntry] | None = ...,
        diagnostics: OutputDiagnostics | None = ...,
    ) -> None: ...

    def __init__(
        self,
        value: Any = _UNSET,
        *,
        response: Any = _UNSET,
        episode_id: str | None = None,
        inference_ids: list[str] | None = None,
        turns: int = 0,
        response_source: Literal["text", "thought"] = "text",
        guardrail_log: list[GuardrailLogEntry] | None = None,
        diagnostics: OutputDiagnostics | None = None,
    ) -> None:
        if value is _UNSET and response is _UNSET:
            raise TypeError(
                "AgentResult requires `value=` (or v0.12-compat `response=`)"
            )
        if value is not _UNSET and response is not _UNSET:
            raise TypeError(
                "AgentResult: pass either `value=` or v0.12-compat `response=`, "
                "not both"
            )
        self.value = cast(T, value if value is not _UNSET else response)
        self.episode_id = episode_id
        self.inference_ids = (
            list(inference_ids) if inference_ids is not None else []
        )
        self.turns = turns
        self.response_source = response_source
        self.guardrail_log = (
            list(guardrail_log) if guardrail_log is not None else []
        )
        self.diagnostics = diagnostics

    # ------------------------------------------------------------------
    # v0.12 read-side compat properties (contract §2.4 Tier 0)
    # ------------------------------------------------------------------

    @property
    def response(self) -> str:
        """The model's textual response (chat path) or raw JSON (JSON path).

        Chat mode: ``self.value`` is the boundary assistant text.
        JSON / structured / tool-call mode: ``self.diagnostics.raw_text``
        carries the model's raw output; falls back to empty string when
        ``diagnostics`` is missing.
        """
        if isinstance(self.value, str):
            return self.value
        if self.diagnostics is not None and self.diagnostics.raw_text is not None:
            return self.diagnostics.raw_text
        return ""

    @property
    def parsed(self) -> Any | None:
        """The parsed JSON payload (JSON / structured / tool-call mode).

        Chat mode: ``None`` — there is no parsed payload, only text.
        Other modes: returns ``self.value`` (the dict / Pydantic
        instance / tool-call list).
        """
        if isinstance(self.value, str):
            return None
        return self.value

    @property
    def parse_strategy(self) -> ParseStrategy:
        """Which recovery layer produced :attr:`parsed`.

        Defaults to ``"strict"`` for results whose ``diagnostics`` is
        ``None`` (chat path or no-diagnostics single-shot path) so v0.12
        consumers that read this field on chat results see the v0.12
        default.
        """
        if self.diagnostics is None:
            return "strict"
        return self.diagnostics.parse_strategy

    @property
    def json_parse_attempts(self) -> int:
        """Number of corrective retry attempts during the JSON path.

        ``0`` when no retry was needed or for chat results.
        """
        if self.diagnostics is None:
            return 0
        return self.diagnostics.attempts

    @property
    def function_kind(self) -> Literal["chat", "json"]:
        """Whether this result came from a chat-style or single-shot run.

        Heuristic for v0.12 compat: chat results carry ``value: str``;
        every other shape (dict, Pydantic instance, ``list[ToolCall]``)
        reports ``"json"``.  Custom :class:`OutputMode` implementations
        that materialise to ``str`` should set
        ``diagnostics.parse_strategy`` appropriately if they want to be
        observable via the JSON-path properties.
        """
        if isinstance(self.value, str):
            return "chat"
        return "json"


@dataclass(frozen=True, slots=True)
class Usage:
    """One inference's token usage — a lossless mirror of ``tensorzero.Usage``.

    Where :class:`RunUsage` is the *cumulative* aggregate over a whole
    :class:`~t0_agents.core.run.Run`, ``Usage`` is the *per-inference*
    record captured by
    :class:`~t0_agents.core.middleware.CostTrackingMiddleware` on each
    ``on_inference`` callback.  It is the element type of
    :meth:`~t0_agents.core.run.Run.usages` and the return type of
    :meth:`~t0_agents.core.run.Run.context_footprint`.

    "Lossless" means the three optional fields keep their ``None`` rather
    than collapsing to ``0`` the way :class:`RunUsage` does for summing:
    a ``None`` ``cost`` means "the provider did not price this call",
    distinct from a real ``0.0``.

    The fields mirror ``tensorzero.types.Usage`` exactly; the type is
    re-derived here (rather than re-exporting the SDK dataclass) so the
    public surface does not depend on the tensorzero type.  The name
    deliberately parallels ``tensorzero.Usage`` — ``core/structured.py``
    imports the SDK type under the same name within its own module, with
    no collision.

    Fields:
        input_tokens: Prompt tokens for this inference — for a chat turn,
            roughly the full conversation re-sent.  ``0`` on a
            cache-served turn (a whole-response cache hit), which is why
            :meth:`~t0_agents.core.run.Run.context_footprint` skips zeros.
        output_tokens: Completion tokens generated.
        provider_cache_read_input_tokens: Prompt tokens served from the
            provider's prompt cache (a subset of ``input_tokens``), or
            ``None`` when the provider does not report it (e.g. Groq).
        provider_cache_write_input_tokens: Prompt tokens written to the
            provider's prompt cache, or ``None``.
        cost: Provider-reported cost (USD) for this inference, or ``None``
            when unpriced (short-hand models / no ``cost`` config).
    """

    input_tokens: int
    output_tokens: int
    provider_cache_read_input_tokens: int | None = None
    provider_cache_write_input_tokens: int | None = None
    cost: float | None = None

    @classmethod
    def from_tz(cls, usage: Any) -> Usage:
        """Build a :class:`Usage` from a ``tensorzero.Usage`` (duck-typed).

        Reads fields via ``getattr`` so any object exposing the same
        attribute names works: the ``on_inference`` callback receives a
        ``ChatInferenceResponse`` / ``JsonInferenceResponse`` whose
        ``.usage`` is the SDK dataclass, and tests pass a stand-in.  The
        two required token counts coerce ``None`` → ``0``; the three
        optional fields are copied verbatim to stay lossless.
        """
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            provider_cache_read_input_tokens=getattr(
                usage, "provider_cache_read_input_tokens", None
            ),
            provider_cache_write_input_tokens=getattr(
                usage, "provider_cache_write_input_tokens", None
            ),
            cost=getattr(usage, "cost", None),
        )

    @property
    def total_tokens(self) -> int:
        """``input_tokens + output_tokens``.

        The two ``provider_cache_*`` fields are a *breakdown* of
        ``input_tokens`` (cached vs fresh prompt tokens), not additional
        tokens, so they are not part of the total.
        """
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Cumulative usage totals across every inference in a :class:`Run`.

    Populated by :class:`~t0_agents.core.middleware.CostTrackingMiddleware`
    once per inference and read by :meth:`Run.usage`.  Aggregates the
    fields tensorzero exposes on every inference response so callers
    can budget, attribute cost, or surface usage to observability
    pipelines without manually walking ``run.transcript()`` or scanning
    ``GuardrailContext``.

    Fields:
        input_tokens: Sum of ``response.usage.input_tokens`` over every
            inference (chat turns + JSON retries).
        output_tokens: Sum of ``response.usage.output_tokens``.
        cost: Sum of ``response.usage.cost`` (treats ``None`` as ``0.0``
            — local providers may omit pricing).
        provider_cache_read_input_tokens: Sum of the per-inference
            ``provider_cache_read_input_tokens`` (``None`` treated as
            ``0``) — the portion of the prompt served from the provider's
            cache.
        provider_cache_write_input_tokens: Sum of the per-inference
            ``provider_cache_write_input_tokens`` (``None`` treated as
            ``0``).
        cost_is_priced: ``True`` when at least one inference reported a
            non-``None`` ``cost`` (a real price, possibly ``0.0``);
            ``False`` when every inference was unpriced.  Distinguishes a
            genuine ``$0.00`` from "no pricing configured" — a bare
            ``cost == 0.0`` cannot, since the summer treats ``None`` as
            ``0.0``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    provider_cache_read_input_tokens: int = 0
    provider_cache_write_input_tokens: int = 0
    cost_is_priced: bool = False


@dataclass(frozen=True, slots=True)
class ToolCallUsage:
    """One tool call as the loop dispatched it — identity and outcome, never payload.

    ``inference_id`` is the gateway id of the inference that emitted the
    call, or ``None`` when the streaming turn ended without a usage chunk.
    ``tool_call_id`` is the model's block id; ``name`` is the name the
    model wrote, kept as written when the agent did not recognise it.
    ``server`` is the mounted server the provider attributed, ``None``
    when it could not.  ``duration_ms`` is ``None`` on the pre-dispatch
    error branches (hallucinated name, invalid arguments) that never
    reach the tool.  Arguments and results are deliberately absent.
    """

    inference_id: str | None
    tool_call_id: str
    name: str
    server: str | None
    duration_ms: float | None
    is_error: bool


#: Key under which a delegated sub-agent's billing usage rides home on the
#: ``meta`` of its :class:`fastmcp.client.client.CallToolResult`.  Namespaced
#: so it never collides with other ``meta`` content.  Written by
#: :class:`~t0_agents.delegation.provider.SubAgentToolProvider` and read back by
#: :class:`~t0_agents.core.middleware.CostTrackingMiddleware`, which folds the
#: child's *billing* into the orchestrator's parent :class:`~t0_agents.core.run.Run`
#: (the parent **footprint** stays parent-only — only billing rolls up).
USAGE_META_KEY = "t0_agents/usage"


def encode_usage_meta(usage: RunUsage) -> dict[str, Any]:
    """Serialise a child Run's :class:`RunUsage` for ``CallToolResult.meta``.

    Returns a single-key dict ``{USAGE_META_KEY: {<RunUsage fields>}}`` of plain
    primitives (JSON-safe, so the payload survives observability sinks).  The
    inverse is :func:`decode_usage_meta`.  Delegation is in-process, so the
    ``CallToolResult`` is handed back as a live object and the round-trip never
    actually serialises — keeping it JSON-safe is forward-looking robustness.
    """
    return {USAGE_META_KEY: asdict(usage)}


def decode_usage_meta(meta: dict[str, Any] | None) -> RunUsage | None:
    """Recover a :class:`RunUsage` from a ``CallToolResult.meta``, or ``None``.

    Returns ``None`` when ``meta`` is ``None`` or not a mapping, lacks
    :data:`USAGE_META_KEY`, or carries a payload that is not a mapping — so a
    tool result without (or with foreign / malformed) ``meta`` folds nothing
    rather than raising.  Unknown payload keys are ignored and missing ones fall
    back to the :class:`RunUsage` defaults, keeping decode tolerant of version
    skew.
    """
    if not isinstance(meta, dict):
        return None
    payload = meta.get(USAGE_META_KEY)
    if not isinstance(payload, dict):
        return None
    known = {f.name for f in fields(RunUsage)}
    return RunUsage(**{k: v for k, v in payload.items() if k in known})
