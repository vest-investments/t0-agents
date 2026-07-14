"""``SubAgentToolProvider`` — synthesise a tool surface from sub-agents.

Replaces the pre-v1.x dual-vía where (1) :func:`install_sub_agents`
registered each sub-agent on an in-process FastMCP server so
``list_tools()`` would surface them, and (2) ``_OrchestrationToolProvider``
intercepted ``call_tool`` mid-run to dispatch inline.  Both responsibilities
collapse into this single class:

- :meth:`list_tools` **synthesises** :class:`mcp.types.Tool` objects from
  the sub-agent list directly — no FastMCP server required.  When
  composed with an ``inner`` :class:`~t0_agents.core.types.ToolProvider`
  the inner tools come first, mirroring the legacy
  ``add_tool``-then-``list_tools`` ordering.
- :meth:`call_tool` dispatches matching sub-agent names inline through
  :meth:`~t0_agents.core.agent.Agent.invoke` — same asyncio task as the
  orchestrator — so the :data:`PARENT_EPISODE_ID_TAG` correlation tag
  reads the live ContextVar binding set by
  :class:`~t0_agents.core.middleware.builtins.SubAgentMiddleware`.

The class is auto-allocated by :class:`~t0_agents.core.agent.Agent` when
``config.sub_agents`` is non-empty; callers do not construct it directly.
The ``inner`` parameter carries whatever ``tool_provider`` the user passed
to the orchestrator (typically a :class:`fastmcp.Client` when the
orchestrator also exposes real MCP tools, or ``None`` for the pure
sub-agent case).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from t0_agents.core.exceptions import SubAgentError, T0AgentsFatalError
from t0_agents.core.types import (
    ToolProgressHandler,
    ToolProvider,
    encode_usage_meta,
)
from t0_agents.delegation.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    get_parent_episode_id,
)
from t0_agents.delegation.routing import assert_unique_names, default_tool_name

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


#: Flat ``message: str`` schema shared by every sub-agent tool.  Matches the
#: schema FastMCP's ``Tool.from_function`` produced from the legacy
#: ``agent_as_tool`` handler signature, so LLM-side tool descriptions remain
#: byte-identical to the pre-refactor contract.
_SUB_AGENT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
}


def _default_description(sub_agent: Agent) -> str:
    """Return the LLM-facing description for ``sub_agent``.

    Mirrors the legacy ``agent_as_tool`` default so the orchestrator's
    prompt history continues to address sub-agents with the same wording.
    """
    name = sub_agent.config.name
    return (
        f"Delegate a question or sub-task to the {name!r} agent. "
        f"Use when the request fits the {name!r} agent's specialisation; "
        "the returned string is the sub-agent's answer."
    )


class SubAgentToolProvider:
    """ToolProvider that exposes a list of sub-agents as MCP-shaped tools.

    Implements the :class:`~t0_agents.core.types.ToolProvider` protocol via
    the duck-typed :meth:`list_tools` / :meth:`call_tool` pair the loop
    consumes (`t0_agents.mcp.loop`).  No FastMCP server is constructed:
    :meth:`list_tools` synthesises :class:`mcp.types.Tool` objects from the
    sub-agent list, and :meth:`call_tool` dispatches matching names inline
    so the :data:`PARENT_EPISODE_ID_TAG` correlation tag stays attached to
    every delegated inference.

    ``inner`` composes with an existing :class:`ToolProvider` for agents
    that *also* expose real MCP tools — the inner provider's tools appear
    first in :meth:`list_tools`, and unmatched names fall through to
    :meth:`inner.call_tool`.  ``inner=None`` (the common case for pure
    orchestrators) keeps the class fully self-contained.
    """

    def __init__(
        self,
        sub_agents: Sequence[Agent],
        *,
        inner: ToolProvider | None = None,
    ) -> None:
        assert_unique_names(sub_agents)
        self._by_tool_name: dict[str, Agent] = {
            default_tool_name(sa): sa for sa in sub_agents
        }
        self._inner = inner

    async def list_tools(self) -> list[MCPTool]:
        """Return inner tools (if any) followed by one synthetic tool per sub-agent."""
        inner_tools: list[MCPTool] = []
        if self._inner is not None:
            inner_tools = cast(list[MCPTool], await self._inner.list_tools())

        synthetic: list[MCPTool] = [
            MCPTool(
                name=name,
                description=_default_description(sub),
                inputSchema=_SUB_AGENT_INPUT_SCHEMA,
            )
            for name, sub in self._by_tool_name.items()
        ]
        return [*inner_tools, *synthetic]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        progress_handler: ToolProgressHandler | None = None,
    ) -> CallToolResult:
        """Dispatch ``name`` to a sub-agent or fall through to ``inner``.

        Sub-agent dispatch uses :meth:`Agent.invoke` so the child runs its
        own middleware chain (memory, guardrails, cost-tracking, nested
        sub-agents).  :class:`T0AgentsFatalError` propagates unwrapped so
        hard blocks reach the orchestrator caller with their original
        type; every other exception is wrapped in :class:`SubAgentError`
        so the orchestrator LLM sees a clean tool error and can retry or
        re-delegate.

        ``progress_handler`` is forwarded on the fall-through path only:
        a sub-agent runs its own agent loop in a separate context and has
        no per-call progress channel to relay.
        """
        sub_agent = self._by_tool_name.get(name or "")
        if sub_agent is None:
            if self._inner is not None:
                return cast(
                    CallToolResult,
                    await self._inner.call_tool(
                        name,
                        arguments or {},
                        timeout=timeout,
                        progress_handler=progress_handler,
                    ),
                )
            return _error_result(
                f"Unknown tool {name!r}. "
                f"Available sub-agents: {sorted(self._by_tool_name)}."
            )

        args = arguments or {}
        message = args.get("message")
        if not isinstance(message, str):
            return _error_result(
                f"sub-agent tool {name!r} requires a 'message: str' argument; "
                f"got {args!r}"
            )

        parent_episode_id = get_parent_episode_id() or "unknown"
        delegation_tags = {PARENT_EPISODE_ID_TAG: parent_episode_id}
        try:
            child_run = sub_agent.invoke(message, tags=delegation_tags)
            result = await child_run.result()
        except T0AgentsFatalError:
            raise
        except Exception as exc:
            raise SubAgentError(
                f"Sub-agent {sub_agent.config.name!r} failed: {exc}"
            ) from exc

        # Carry the child's billing usage home on ``meta`` so the orchestrator's
        # CostTrackingMiddleware can roll it into the parent Run.  ``None`` until
        # the child produced an inference (or when its cost-tracking was opted
        # out).  Only *billing* travels up — the parent's context footprint
        # stays its own, since the child ran in a separate context window.
        child_usage = child_run.usage()
        meta = (
            encode_usage_meta(child_usage) if child_usage is not None else None
        )
        return CallToolResult(
            content=[TextContent(type="text", text=result.response)],
            structured_content=None,
            meta=meta,
            data=None,
            is_error=False,
        )


def _error_result(message: str) -> CallToolResult:
    """Wrap ``message`` in an ``is_error=True`` :class:`CallToolResult`."""
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structured_content=None,
        meta=None,
        data=None,
        is_error=True,
    )
