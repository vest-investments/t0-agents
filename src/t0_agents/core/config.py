"""Declarative configuration for an Agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from tensorzero import ToolChoice

from t0_agents.core.exceptions import ConfigurationError


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
