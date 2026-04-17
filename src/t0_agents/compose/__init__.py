"""Multi-agent composition for t0-agents (Phase G).

Exposes :func:`agent_as_tool` for wrapping an :class:`~t0_agents.core.agent.Agent`
as an MCP tool the orchestrator's FastMCP server can advertise.  The
declarative path is ``AgentConfig.sub_agents=[...]``, which triggers
``install_sub_agents`` from :mod:`t0_agents.compose.orchestrator` at
``Agent.__init__`` time.
"""

from t0_agents.compose.delegation import agent_as_tool
from t0_agents.compose.orchestrator import install_sub_agents
from t0_agents.compose.routing import assert_unique_names, default_tool_name

__all__ = [
    "agent_as_tool",
    "assert_unique_names",
    "default_tool_name",
    "install_sub_agents",
]
