from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

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
    """Hook surface invoked by ``Agent.run()`` and ``GuardedToolProvider``.

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

    async def after_run(self, ctx: Any, result: AgentResult | None) -> None: ...


@dataclass(slots=True)
class GuardrailLogEntry:
    """One guardrail hook activation recorded during ``Agent.run()``.

    Populated by ``GuardedToolProvider`` (for ``before_tool`` / ``after_tool``)
    and by ``Agent.run`` itself (for ``before_run`` / ``after_run``).  Attached
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


@dataclass
class AgentResult:
    """Result of an agent run.

    ``function_kind`` and ``parsed`` carry the structured-outputs
    payload; ``parse_strategy`` and ``json_parse_attempts`` expose the
    JSON path's permissive-parsing and bounded-retry observability.  All
    four fields default to byte-identical chat behaviour: a chat agent's
    result reads the same as v0.9.0 (``function_kind="chat"``,
    ``parsed=None``, ``parse_strategy="strict"`` — semantically
    meaningless for chat but chosen to match the JSON happy-path default
    — and ``json_parse_attempts=0``).
    """

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
    guardrail_log: list[GuardrailLogEntry] = field(default_factory=list)
    function_kind: Literal["chat", "json"] = "chat"
    parsed: Any | None = None
    parse_strategy: ParseStrategy = "strict"
    json_parse_attempts: int = 0
