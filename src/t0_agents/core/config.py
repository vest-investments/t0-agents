"""Declarative configuration for an Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tensorzero import ToolChoice

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.types import Guardrail, MemoryProvider

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent
    from t0_agents.core.middleware.base import Middleware

_VALID_MEMORY_STRATEGIES = frozenset(
    {"auto", "system_template", "prepended_message", "messages"}
)


def _functionless_json_error(name: str, model: str, output_name: str) -> str:
    """Error text for ``model=`` + a JSON-function output mode.

    Carries a ready-to-paste tensorzero.toml stanza so the fix is a
    copy-paste plus swapping ``model=`` for ``function_name=``.
    """
    return (
        f"AgentConfig(model={model!r}) is incompatible with "
        f"output={output_name!r}: model= routes through the gateway's "
        f"built-in chat passthrough function and output_schema is ignored "
        f"there, so JSON-validated output can never be produced. Declare a "
        f"JSON function in tensorzero.toml and pass function_name= instead:\n"
        f"\n"
        f"[functions.{name}]\n"
        f'type = "json"\n'
        f'output_schema = "functions/{name}/output_schema.json"\n'
        f"\n"
        f"[functions.{name}.variants.default]\n"
        f'type = "chat_completion"\n'
        f'model = "{model}"\n'
        f'json_mode = "strict"\n'
    )


@dataclass(slots=True)
class AgentConfig:
    """Declarative configuration for an Agent.

    Captures every parameter that ``run_agent`` accepts so that callers
    configure once and call ``Agent.invoke(message)`` with no
    boilerplate.

    Attributes:
        name: Human-readable agent name; used for logging and tags (the
            ``t0_agents.agent`` observability tag carries it on every
            inference).
        function_name: TensorZero function name (must exist in
            ``tensorzero.toml``).  Mutually exclusive with ``model``.
            When neither is given, defaults to ``"agent"``.
        model: Model to call without any tensorzero.toml declaration —
            either a ``{provider}::{model}`` shorthand (e.g.
            ``"groq::llama-3.3-70b-versatile"``) or a ``[models.*]`` name.
            Routes through the gateway's built-in chat passthrough
            function ``tensorzero::default``; the loop, tools,
            episodes, and streaming all work unchanged.  Mutually
            exclusive with ``function_name`` and ``variant``, and
            incompatible with JSON-function output modes (the
            passthrough ignores ``output_schema`` — the error carries a
            ready-to-paste TOML stanza).  Note the memory caveat:
            inferences land under ``tensorzero::default``, so
            ``TensorZeroMemory(function_name=...)`` filters will not see
            them — filter on ``tag_filter={"t0_agents.agent": name}``
            instead.
        variant: Pin a specific variant of ``function_name`` (forwarded
            as ``variant_name``).  Requires the function path — the
            gateway rejects variant pinning for ``model``-routed calls.
        params: Per-agent inference parameters in the TensorZero shape,
            e.g. ``{"chat_completion": {"temperature": 0.2}}``.  Merged
            under any output-mode parameters (the mode wins on
            ``json_mode``).
        cache: TensorZero ``cache_options``, e.g.
            ``{"enabled": "on", "max_age_s": 3600}``.  ``enabled`` is one
            of ``"write_only"``/``"read_only"``/``"on"``/``"off"``; set it
            explicitly rather than relying on the gateway default.
        max_turns: Maximum inference turns before ``MaxTurnsExceededError``.
        tool_timeout: Per-tool-call timeout in seconds, or ``None`` to
            disable the per-call timeout entirely — required when a tool
            can block on human elicitation longer than any fixed budget.
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
        output: The :class:`OutputMode` driving inference shape +
            materialisation + recovery.  Default is :class:`ChatOutput`
            (the v0.12 chat agent).  Pass
            :class:`~t0_agents.core.outputs.JsonOutput` for JSON-mode
            agents, :class:`~t0_agents.core.outputs.StructuredOutput`
            for Pydantic-validated single-shot inference,
            :class:`~t0_agents.core.outputs.ToolCallOutput` for the
            router pattern.  Per-call override is available via
            :meth:`Agent.invoke` ``output=`` kwarg.

            v1.0 replaced the five v0.12 JSON-only fields
            (``function_kind``, ``json_mode``, ``output_schema``,
            ``permissive_json_parsing``, ``max_json_retries``) with this
            single canonical surface.  Passing any of the legacy kwargs
            raises ``TypeError`` (Tier 2 removal — contract §2.4).
        middleware: Override for the materialised middleware chain.
            ``None`` (default) builds the v0.12-equivalent chain from
            the kwargs above.  ``[]`` is the explicit "no middleware"
            opt-out — power users only.
    """

    # Identity
    name: str
    function_name: str | None = None

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

    # Output mode (v1.0).  Replaces the five v0.12 JSON-only fields.
    output: OutputMode[Any] = field(default_factory=ChatOutput)

    # Middleware chain (v1.0).  ``None`` (default) materialises the
    # backwards-compatible chain in ``__post_init__`` from
    # ``memory=``/``guardrails=``/``sub_agents=``.  ``[]`` (empty list)
    # is the explicit "no middleware" override — power-user opt-in,
    # semantically distinct from ``None``.
    middleware: list[Middleware] | None = None

    # Functionless routing + per-model inference knobs.  Appended after
    # the v1.x fields so positional construction stays stable.
    model: str | None = None
    variant: str | None = None
    params: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("AgentConfig.name must be a non-empty string")
        if self.model is not None and self.function_name is not None:
            raise ConfigurationError(
                f"AgentConfig accepts model= XOR function_name=, not both "
                f"(got model={self.model!r}, function_name={self.function_name!r}). "
                f"model= routes through the gateway's built-in passthrough "
                f"function; function_name= targets a tensorzero.toml function."
            )
        if self.model is not None and not self.model:
            raise ConfigurationError("AgentConfig.model must be a non-empty string")
        if self.model is None and self.function_name is None:
            self.function_name = "agent"
        if self.function_name is not None and not self.function_name:
            raise ConfigurationError(
                "AgentConfig.function_name must be a non-empty string"
            )
        if self.variant is not None and self.model is not None:
            raise ConfigurationError(
                f"AgentConfig.variant={self.variant!r} cannot be combined with "
                f"model= — the gateway rejects variant_name for model-routed "
                f"inference (functionless calls have no declared variants). "
                f"Pin a variant on a tensorzero.toml function via "
                f"function_name= + variant=."
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

        # Output mode validations (v1.0 — replaces ~80 LoC of v0.12
        # cross-field validation on function_kind/json_mode/output_schema).
        if not isinstance(self.output, OutputMode):
            raise ConfigurationError(
                f"AgentConfig.output must implement the OutputMode protocol "
                f"(shape_inference + materialize + recover + capability flags), "
                f"got {type(self.output).__name__}."
            )
        if not self.output.expects_tool_loop and self.sub_agents:
            raise ConfigurationError(
                f"AgentConfig.sub_agents is incompatible with "
                f"output={self.output.name!r} (single-shot mode — sub-agent "
                f"tools would be unreachable). Use ChatOutput() for the "
                f"orchestrator, or compose chat → single-shot manually."
            )
        if (
            not self.output.expects_tool_loop
            and self.memory_strategy == "messages"
        ):
            raise ConfigurationError(
                f"memory_strategy='messages' is incompatible with "
                f"output={self.output.name!r} (single-shot mode — there is no "
                f"multi-turn message history to thread). Use "
                f"memory_strategy='prepended_message' (the default) or "
                f"'system_template' for single-shot outputs."
            )
        # model= routes through the gateway's built-in chat passthrough
        # (``tensorzero::default``), which ignores per-call output_schema
        # (verified against gateways 2026.4.0 and 2026.6.0) — JSON-function
        # output modes
        # can never be satisfied on that path.
        if self.model is not None and getattr(
            self.output, "requires_json_function", False
        ):
            raise ConfigurationError(
                _functionless_json_error(self.name, self.model, self.output.name)
            )

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
        unconditionally, and ``JsonRetry`` activates for any non-chat
        output mode by consulting ``ctx.output.recover()``.

        **Order invariant.** ``SubAgentMiddleware`` must
        wrap ``tool_provider`` *inside* ``GuardrailsMiddleware`` so
        orchestrator-level guardrails observe sub-agent delegations
        via their ``before_tool``/``after_tool`` hooks.  Because each
        middleware contributes one ``replace_tool_provider`` call in
        chain order, the wrap that happens LAST becomes the OUTERMOST.
        :class:`SubAgentMiddleware` is now ContextVar-only and no longer
        wraps ``tool_provider``, so Guardrails is the sole wrapper at
        terminal time — running it inside the chain order ensures it
        wraps the :class:`SubAgentToolProvider` Agent.__init__ pre-installed
        on ``self._tool_provider``.

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
        # SubAgentMiddleware is ContextVar-only; placed before Guardrails
        # so the parent-episode binding is in scope when guardrails fire.
        if sub_agents:
            chain.append(SubAgentMiddleware(list(sub_agents)))
        if guardrails:
            chain.append(GuardrailsMiddleware(list(guardrails)))
        chain.append(CostTrackingMiddleware())
        chain.append(JsonRetryMiddleware())
        return chain
