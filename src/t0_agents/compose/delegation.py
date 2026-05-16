"""Wrap a sub-agent as an MCP tool.

The generated handler signature is ``async def <name>(message: str) -> str``
so FastMCP's :meth:`Tool.from_function` produces a flat, predictable
schema ``{ "type": "object", "properties": { "message": {"type":
"string"} }, "required": ["message"] }`` — the orchestrator LLM sees a
single-argument "ask" affordance per sub-agent.

Error handling is **hybrid fatal**: any
:class:`~t0_agents.core.exceptions.T0AgentsFatalError` (for example a
:class:`~t0_agents.core.exceptions.GuardrailBlockedError` raised by the
sub-agent's own guardrail) propagates unchanged so hard blocks reach the
caller of :meth:`Agent.invoke`.  Every other exception is wrapped in
:class:`~t0_agents.core.exceptions.SubAgentError`, which the orchestrator's
:mod:`~t0_agents.mcp.loop` stringifies as a tool error so the LLM can
decide to retry or re-delegate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.tools import Tool

from t0_agents.compose.routing import default_tool_name

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


def agent_as_tool(
    sub_agent: Agent,
    *,
    tool_name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Wrap ``sub_agent`` as a FastMCP :class:`Tool`.

    Args:
        sub_agent: The child :class:`~t0_agents.core.agent.Agent` to expose.
        tool_name: Override the default ``{name}_ask`` tool name.
        description: Override the default docstring surfaced to the LLM.

    Returns:
        A :class:`Tool` instance ready to register with ``server.add_tool``.
    """
    name = tool_name or default_tool_name(sub_agent)
    desc = description or (
        f"Delegate a question or sub-task to the "
        f"{sub_agent.config.name!r} agent. "
        f"Use when the request fits the {sub_agent.config.name!r} agent's "
        "specialisation; the returned string is the sub-agent's answer."
    )

    async def _handler(message: str) -> str:
        # Deferred imports keep this module import-light so the
        # orchestrator submodule stays the only place that touches the
        # ContextVar machinery.
        from t0_agents.compose.orchestrator import (
            PARENT_EPISODE_ID_TAG,
            get_parent_episode_id,
        )
        from t0_agents.core.exceptions import SubAgentError, T0AgentsFatalError

        parent_episode_id = get_parent_episode_id() or "unknown"
        delegation_tags = {PARENT_EPISODE_ID_TAG: parent_episode_id}

        try:
            # Dispatch via the canonical Agent.invoke API so the
            # sub-agent runs its own middleware chain (memory,
            # guardrails, cost-tracking, sub-agents).
            result = await sub_agent.invoke(message, tags=delegation_tags).result()
        except T0AgentsFatalError:
            raise
        except Exception as exc:
            raise SubAgentError(
                f"Sub-agent {sub_agent.config.name!r} failed: {exc}"
            ) from exc
        return result.response

    _handler.__name__ = name
    _handler.__doc__ = desc

    return Tool.from_function(_handler, name=name, description=desc)
