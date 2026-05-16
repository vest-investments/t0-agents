from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

from t0_agents.core.output import OutputDiagnostics
from t0_agents.core.structured import ParseStrategy


@runtime_checkable
class ToolProvider(Protocol):
    """Structural type for tool discovery and execution.

    Matches the slice of ``fastmcp.Client`` that the agent loop uses:

    - ``list_tools()`` — discover available tools at loop start.
    - ``call_tool(name, arguments, *, timeout=...)`` — execute a single tool.

    Any object implementing these two async methods satisfies the protocol,
    enabling testing with fakes and future non-FastMCP providers.
    """

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any: ...


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
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
