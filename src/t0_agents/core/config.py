"""Declarative configuration for an Agent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tensorzero import ToolChoice

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import Guardrail, MemoryProvider

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent
    from t0_agents.core.middleware.base import Middleware

_VALID_MEMORY_STRATEGIES = frozenset(
    {"auto", "system_template", "prepended_message", "messages"}
)


@dataclass(slots=True)
class AgentConfig:
    """Declarative configuration for an Agent.

    Captures every parameter that ``run_agent`` accepts so that callers
    configure once and call ``Agent.invoke(message)`` with no
    boilerplate.

    Attributes:
        name: Human-readable agent name; used for logging and tags.
        function_name: TensorZero function name (must exist in ``tensorzero.toml``).
        max_turns: Maximum inference turns before ``MaxTurnsExceededError``.
        tool_timeout: Per-tool-call timeout in seconds, or ``None`` to use the
            framework default (30.0 s).
        parallel_tools: Dispatch tool calls via ``asyncio.gather`` when ``True``.
        dryrun: Forward ``dryrun=True`` to every ``t0.inference()`` call.
        tool_choice: Tool-choice directive (``"auto"``, ``"required"``, ``"off"``,
            or ``{"specific": "<name>"}``).
        tags: Free-form tags forwarded to ``t0.inference()`` for observability.
        namespace: TensorZero namespace for experimentation routing.
        memory: Optional :class:`MemoryProvider` backend. When set, the
            ``Agent`` hydrates memory before each run and persists after.
        memory_strategy: How memory context is injected into the prompt.
            ``"auto"`` resolves to ``"prepended_message"`` (the strategy
            that requires no TOML variant support).
            ``"system_template"`` passes memory via ``input.system`` (the
            schema-described arguments dict).
            ``"prepended_message"`` prepends a text summary to the user message.
            ``"messages"`` threads the prior conversation as a TensorZero-
            native ``input.messages`` history; requires the memory
            provider to implement ``load_history(episode_id) -> list[Message]
            | None`` (e.g. :class:`~t0_agents.memory.TensorZeroMemory`).
            Recommended whenever the backend can supply structured
            history because it sidesteps the unbounded summary growth
            that ``prepended_message`` exhibits across turns.
        guardrails: Optional list of :class:`Guardrail` instances.  When
            non-empty, the ``Agent`` wraps the tool provider with a
            ``GuardedToolProvider`` that fires ``before_tool`` / ``after_tool``
            hooks around every tool call, tracks per-run counters, and
            propagates ``GuardrailBlockedError`` as a hard abort.  An empty
            list (the default) disables the wrapper entirely so behaviour
            remains byte-identical to the non-guardrail code path.
        sub_agents: Optional list of :class:`Agent` instances to expose
            as MCP tools on the orchestrator's FastMCP server.  When
            non-empty, ``Agent.__init__`` installs each sub-agent as a
            tool named ``f"{sub_agent.config.name}_ask"`` and raises
            :class:`ConfigurationError` on name collisions.  Requires
            the orchestrator's ``tool_provider`` to be a
            ``fastmcp.Client`` wrapping an in-process ``FastMCP``
            instance.  The default empty list leaves single-agent
            configurations unaffected.
        function_kind: Declares the TensorZero function type for
            dispatch.  ``"chat"`` (the default) preserves v0.9.0
            behaviour byte-identically.  ``"json"`` marks the function
            as a single-shot JSON function and routes ``Agent.run``
            through :func:`run_structured_inference`.
        output_schema: Optional dynamic schema for JSON functions.
            Accepts either an inline ``Mapping[str, Any]`` (sent
            verbatim) or a :class:`~pathlib.Path` (read as JSON at
            construction time; relative paths resolve against the
            current working directory — absolute paths recommended).
            Forbidden when ``function_kind == "chat"``.
        json_mode: TensorZero ``json_mode`` override for the variant
            (``"off"``, ``"on"``, ``"strict"``, ``"tool"``).  Required
            when ``function_kind == "json"``; forbidden when
            ``function_kind == "chat"`` (TZ guide §11.2 — setting
            ``json_mode`` on a chat function is a configuration
            error).
    """

    # Identity
    name: str
    function_name: str

    # Loop budget
    max_turns: int = 20
    tool_timeout: float | None = 30.0
    parallel_tools: bool = False

    # TensorZero knobs
    dryrun: bool = False
    tool_choice: ToolChoice | None = None
    tags: dict[str, str] | None = field(default=None)
    namespace: str | None = None

    # Memory
    memory: MemoryProvider | None = None
    memory_strategy: Literal[
        "auto", "system_template", "prepended_message", "messages"
    ] = "auto"

    # Guardrails
    guardrails: list[Guardrail] = field(default_factory=list)

    # Multi-agent composition
    sub_agents: list[Agent] = field(default_factory=list)

    # Structured outputs
    function_kind: Literal["chat", "json"] = "chat"
    output_schema: Mapping[str, Any] | Path | None = None
    json_mode: Literal["off", "on", "strict", "tool"] | None = None
    permissive_json_parsing: bool = True
    max_json_retries: int = 2

    # Middleware chain (v1.0).  ``None`` (default) materialises the
    # backwards-compatible chain in ``__post_init__`` from
    # ``memory=``/``guardrails=``/``sub_agents=``.  ``[]`` (empty list)
    # is the explicit "no middleware" override — power-user opt-in,
    # semantically distinct from ``None``.
    middleware: list[Middleware] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("AgentConfig.name must be a non-empty string")
        if not self.function_name:
            raise ConfigurationError(
                "AgentConfig.function_name must be a non-empty string"
            )
        if self.max_turns <= 0:
            raise ConfigurationError(
                f"AgentConfig.max_turns must be positive, got {self.max_turns}"
            )
        if self.tool_timeout is not None and self.tool_timeout <= 0:
            raise ConfigurationError(
                f"AgentConfig.tool_timeout must be positive or None, "
                f"got {self.tool_timeout}"
            )
        if self.memory_strategy not in _VALID_MEMORY_STRATEGIES:
            raise ConfigurationError(
                f"AgentConfig.memory_strategy must be one of "
                f"{sorted(_VALID_MEMORY_STRATEGIES)}, got {self.memory_strategy!r}"
            )
        if self.memory_strategy == "messages":
            if self.memory is None:
                raise ConfigurationError(
                    "memory_strategy='messages' requires a memory provider "
                    "that implements load_history(episode_id) -> "
                    "list[Message] | None. Pass AgentConfig(..., "
                    "memory=TensorZeroMemory(...)) or another compatible "
                    "backend."
                )
            if not hasattr(self.memory, "load_history"):
                raise ConfigurationError(
                    f"memory_strategy='messages' requires the memory "
                    f"provider to implement load_history(episode_id) -> "
                    f"list[Message] | None. "
                    f"{type(self.memory).__name__} does not — extend it or "
                    f"swap to TensorZeroMemory."
                )
        if self.sub_agents:
            # Lazy import to dodge the core.agent ↔ core.config cycle;
            # this only fires when the user opts in to multi-agent
            # composition.
            from t0_agents.core.agent import Agent as _Agent

            for idx, sa in enumerate(self.sub_agents):
                if not isinstance(sa, _Agent):
                    raise ConfigurationError(
                        f"AgentConfig.sub_agents[{idx}] must be an Agent "
                        f"instance, got {type(sa).__name__}"
                    )

        # Structured-output validations (contract §3.1 / §5.1)
        if self.max_json_retries < 0:
            raise ConfigurationError(
                f"AgentConfig.max_json_retries must be ≥ 0, "
                f"got {self.max_json_retries}"
            )
        if self.function_kind == "json":
            if self.json_mode is None:
                raise ConfigurationError(
                    "AgentConfig(function_kind='json') requires json_mode "
                    "to be one of 'off', 'on', 'strict', or 'tool'. TZ "
                    "guide §11.2: json_mode is required for json functions."
                )
            if self.sub_agents:
                raise ConfigurationError(
                    "AgentConfig(function_kind='json') is incompatible with "
                    "sub_agents. JSON functions are single-shot — sub-agent "
                    "tools would be unreachable. Compose chat → json "
                    "manually instead."
                )
        else:  # function_kind == "chat"
            if self.json_mode is not None:
                raise ConfigurationError(
                    "AgentConfig(function_kind='chat') forbids json_mode. "
                    "TZ guide §11.2: setting json_mode on a chat function "
                    "is a configuration error."
                )
            if self.output_schema is not None:
                raise ConfigurationError(
                    "AgentConfig(function_kind='chat') forbids output_schema. "
                    "Set function_kind='json' to use a JSON function."
                )

        # Resolve output_schema: Path → Mapping at construction time so
        # downstream code can treat the field as a Mapping.
        if isinstance(self.output_schema, Path):
            schema_path = self.output_schema
            if not schema_path.is_absolute():
                schema_path = (Path.cwd() / schema_path).resolve()
            try:
                schema_text = schema_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigurationError(
                    f"AgentConfig.output_schema points at {schema_path!s}, "
                    f"which could not be read: {exc}"
                ) from exc
            try:
                schema_data = json.loads(schema_text)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"AgentConfig.output_schema at {schema_path!s} is not "
                    f"valid JSON: {exc}"
                ) from exc
            if not isinstance(schema_data, dict):
                raise ConfigurationError(
                    f"AgentConfig.output_schema at {schema_path!s} must be a "
                    f"JSON object at the top level, got "
                    f"{type(schema_data).__name__}."
                )
            self.output_schema = schema_data

        # Materialise the default middleware chain when the caller did
        # not provide one.  ``middleware=[]`` (explicit "no middleware")
        # is preserved verbatim — power-user opt-in.  The lazy import
        # inside ``_build_default_chain`` dodges the
        # ``core.middleware`` ↔ ``core.config`` ↔ ``core.types`` cycle.
        if self.middleware is None:
            self.middleware = self._build_default_chain()

    def _build_default_chain(self) -> list[Middleware]:
        """Build the v0.12-equivalent middleware chain from existing kwargs.

        Order: ``Memory → SubAgents → Guardrails → CostTracking → JsonRetry``.

        ``Memory`` is included only when ``self.memory`` is set;
        ``Guardrails`` only when ``self.guardrails`` is non-empty;
        ``SubAgents`` only when ``self.sub_agents`` is non-empty.
        ``CostTracking`` and ``JsonRetry`` are always present —
        cost-tracking wires :meth:`Run.cost` / :meth:`Run.usage`
        unconditionally, and ``JsonRetry`` is a no-op placeholder in
        M2 (M3 activates it).

        **Order invariant (contract §3.2).** ``SubAgentMiddleware`` must
        wrap ``tool_provider`` *inside* ``GuardrailsMiddleware`` so
        orchestrator-level guardrails observe sub-agent delegations
        via their ``before_tool``/``after_tool`` hooks.  Because each
        middleware contributes one ``replace_tool_provider`` call in
        chain order, the wrap that happens LAST becomes the OUTERMOST.
        Putting ``SubAgents`` BEFORE ``Guardrails`` in the chain order
        means SubAgents wraps first (innermost), Guardrails wraps next
        (outermost) — preserving the v0.12 ``_execute_chat_legacy``
        wrap order (``GuardedToolProvider(_OrchestrationToolProvider(client))``).

        Returns the same list :meth:`default_middleware_for` would
        produce for the same field values.
        """
        return AgentConfig.default_middleware_for(
            memory=self.memory,
            memory_strategy=self.memory_strategy,
            guardrails=self.guardrails,
            sub_agents=self.sub_agents,
        )

    @staticmethod
    def default_middleware_for(
        *,
        memory: MemoryProvider | None = None,
        memory_strategy: Literal[
            "auto", "system_template", "prepended_message", "messages"
        ] = "auto",
        guardrails: list[Guardrail] | None = None,
        sub_agents: list[Agent] | None = None,
    ) -> list[Middleware]:
        """Build the default middleware chain for the given inputs.

        Mirrors :meth:`_build_default_chain` but takes explicit
        parameters so power users can compose custom chains that
        sandwich the default::

            AgentConfig(
                ...,
                middleware=[
                    MyOutermost(),
                    *AgentConfig.default_middleware_for(memory=m, guardrails=g),
                    MyInnermost(),
                ],
            )

        The static method is the single source of truth for the default
        chain shape so the helper and the auto-materialisation cannot
        drift apart.
        """
        # Lazy import to dodge ``core.middleware`` ↔ ``core.config``
        # ↔ ``core.types`` cycle.
        from t0_agents.core.middleware.builtins import (
            CostTrackingMiddleware,
            GuardrailsMiddleware,
            JsonRetryMiddleware,
            MemoryMiddleware,
            SubAgentMiddleware,
        )

        chain: list[Middleware] = []
        if memory is not None:
            chain.append(MemoryMiddleware(memory, memory_strategy))
        # SubAgent MUST appear before Guardrails so the GuardedToolProvider
        # wraps the _OrchestrationToolProvider — see ``_build_default_chain``
        # docstring for the wrap-order rationale.
        if sub_agents:
            chain.append(SubAgentMiddleware(list(sub_agents)))
        if guardrails:
            chain.append(GuardrailsMiddleware(list(guardrails)))
        chain.append(CostTrackingMiddleware())
        chain.append(JsonRetryMiddleware())
        return chain
