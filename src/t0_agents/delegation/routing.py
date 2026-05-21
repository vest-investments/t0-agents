"""Name conventions and collision detection for sub-agents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

from t0_agents.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


def default_tool_name(sub_agent: Agent) -> str:
    """Return ``"{config.name}_ask"`` — the ``_ask`` suffix hints to
    the orchestrator LLM that the tool is a question-answer
    delegation rather than a side-effecting action.
    """
    return f"{sub_agent.config.name}_ask"


def assert_unique_names(sub_agents: Iterable[Agent]) -> None:
    """Reject sub-agent :attr:`AgentConfig.name` collisions early.

    Runs at :class:`SubAgentToolProvider` construction time so the
    orchestrator never builds a tool surface with two handlers that
    would derive the same MCP tool name via :func:`default_tool_name`.
    """
    counts = Counter(sa.config.name for sa in sub_agents)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    if duplicates:
        raise ConfigurationError(
            f"Duplicate sub-agent names: {duplicates}. "
            "Each sub-agent's config.name must be unique."
        )
