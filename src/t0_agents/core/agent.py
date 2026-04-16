"""Declarative Agent wrapper around ``run_agent``."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.mcp.loop import run_agent

logger = logging.getLogger(__name__)

# Default tool timeout used when AgentConfig.tool_timeout is None.
# Matches run_agent's own default so behaviour is identical.
_DEFAULT_TOOL_TIMEOUT: float = 30.0


class Agent:
    """Declarative wrapper around :func:`run_agent`.

    Encapsulates configuration in an :class:`AgentConfig` dataclass and
    exposes a clean ``.run(message)`` method that delegates to the existing
    ``run_agent`` loop.  The caller owns the lifecycle of both the TensorZero
    gateway and the tool provider (typically a ``fastmcp.Client``).

    Example::

        config = AgentConfig(name="assistant", function_name="agent")
        agent = Agent(config, t0=gateway, tool_provider=mcp_client)
        result = await agent.run("What is 2 + 3?")
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider,
    ) -> None:
        if not isinstance(tool_provider, ToolProvider):
            raise ConfigurationError(
                f"tool_provider must implement the ToolProvider protocol "
                f"(list_tools + call_tool), got {type(tool_provider).__name__}"
            )
        self._config = config
        self._t0 = t0
        self._tool_provider = tool_provider
        logger.debug(
            "Agent %r created: function=%s, max_turns=%d",
            config.name,
            config.function_name,
            config.max_turns,
        )

    @property
    def config(self) -> AgentConfig:
        """Read-only access to the agent's configuration."""
        return self._config

    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
    ) -> AgentResult:
        """Run the agent loop and return the final response.

        Delegates to :func:`~t0_agents.mcp.loop.run_agent` with the
        configuration captured in :attr:`config`.

        Args:
            user_message: The user's query as a plain string.
            episode_id: Existing TensorZero episode id to continue, or
                ``None`` for a fresh episode.

        Returns:
            :class:`~t0_agents.core.types.AgentResult` with the joined
            text of the terminating turn, episode metadata, and turn count.

        Raises:
            MaxTurnsExceededError: If the loop exceeds ``config.max_turns``.
        """
        tool_timeout = (
            self._config.tool_timeout
            if self._config.tool_timeout is not None
            else _DEFAULT_TOOL_TIMEOUT
        )
        return await run_agent(
            self._t0,
            cast(Client[Any], self._tool_provider),
            user_message,
            self._config.function_name,
            episode_id=episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            tags=self._config.tags,
            namespace=self._config.namespace,
        )
