"""Sub-agent orchestration for t0-agents.

``AgentConfig.sub_agents=[...]`` registers each sub-agent on a
:class:`SubAgentToolProvider` that synthesises an MCP-shaped tool
surface and dispatches matching ``call_tool`` invocations inline
through :meth:`Agent.invoke`.  The orchestrator's LLM sees them as
``{name}_ask`` tools and decides when to delegate.  Sub-agent
inferences are tagged with ``t0_agents.parent_episode_id`` so
ClickHouse can reconstruct the delegation tree.
"""

from t0_agents.delegation.orchestrator import (
    bind_parent_episode_id,
    get_parent_episode_id,
    update_parent_episode_id,
)
from t0_agents.delegation.provider import SubAgentToolProvider
from t0_agents.delegation.routing import assert_unique_names, default_tool_name

__all__ = [
    "SubAgentToolProvider",
    "assert_unique_names",
    "bind_parent_episode_id",
    "default_tool_name",
    "get_parent_episode_id",
    "update_parent_episode_id",
]
