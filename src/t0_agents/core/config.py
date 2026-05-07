"""Declarative configuration for an Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tensorzero import ToolChoice

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import Guardrail, MemoryProvider

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent

_VALID_MEMORY_STRATEGIES = frozenset(
    {"auto", "system_template", "prepended_message", "messages"}
)


@dataclass(slots=True)
class AgentConfig:
    """Declarative configuration for an Agent.

    Captures every parameter that ``run_agent`` accepts so that callers
    configure once and invoke ``Agent.run(message)`` with no boilerplate.

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
