"""Sub-agent orchestration for t0-agents.

``AgentConfig.sub_agents=[...]`` registers each sub-agent as an MCP
tool via :func:`agent_as_tool` and :func:`install_sub_agents`.  The
orchestrator's LLM sees them as ``{name}_ask`` tools and decides when
to delegate.  Sub-agent inferences are tagged with
``t0_agents.parent_episode_id`` so ClickHouse can reconstruct the
delegation tree.
"""

from t0_agents.delegation.agent_as_tool import agent_as_tool
from t0_agents.delegation.routing import assert_unique_names, default_tool_name

__all__ = [
    "agent_as_tool",
    "assert_unique_names",
    "default_tool_name",
]
