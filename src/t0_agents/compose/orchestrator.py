"""Install sub-agents on an orchestrator's in-process FastMCP server.

Registration happens at :meth:`Agent.__init__` time, **not** per run:
the orchestrator LLM sees the full set of delegated ``_ask`` tools
from the very first inference.  Dynamic mid-run registration is out
of scope — the gateway accepts per-call tool lists, but the LLM-side
cost of schema changes mid-conversation isn't worth it.

The in-process unwrap path uses ``client.transport.server`` when
``isinstance(client.transport, FastMCPTransport)`` — the only
FastMCP transport that exposes the underlying server instance so we
can call ``add_tool`` on it.  Remote transports (HTTP, SSE, WebSocket)
raise :class:`ConfigurationError` at construction time.  Collision
detection translates FastMCP's
``ValueError("Component already exists: ...")`` into
:class:`ConfigurationError` so callers see a predictable t0-agents
exception regardless of the server's ``on_duplicate=`` setting.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import FastMCPTransport
from fastmcp.tools import Tool as _FastMCPTool
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from t0_agents.compose.delegation import agent_as_tool
from t0_agents.delegation.routing import assert_unique_names, default_tool_name
from t0_agents.core.exceptions import (
    ConfigurationError,
    SubAgentError,
    T0AgentsFatalError,
)
from t0_agents.core.types import ToolProvider

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


logger = logging.getLogger(__name__)

#: Tag key carried on every sub-agent inference so ClickHouse can
#: reconstruct the delegation tree via ``tags[PARENT_EPISODE_ID_TAG]``.
PARENT_EPISODE_ID_TAG = "t0_agents.parent_episode_id"

# Kept isolated per :class:`contextvars.ContextVar` so concurrent
# ``Agent.invoke`` calls on one orchestrator don't cross-contaminate each
# other's parent-episode bindings.
_PARENT_EPISODE_ID: ContextVar[str | None] = ContextVar(
    "t0_agents_parent_episode_id",
    default=None,
)


def get_parent_episode_id() -> str | None:
    """Read the current orchestrator's episode id, if any.

    Returns ``None`` when called outside an orchestrator's run (e.g.
    from a sub-agent invoked directly by the user).  Callers should
    treat ``None`` as "no correlation available" rather than a
    configuration bug.
    """
    return _PARENT_EPISODE_ID.get()


@contextlib.contextmanager
def bind_parent_episode_id(episode_id: str | None) -> Iterator[None]:
    """Bind ``episode_id`` as the current parent episode for the block.

    ``None`` is a valid input — sub-agents will then see
    ``get_parent_episode_id() is None`` and fall back to the
    ``"unknown"`` correlation tag.  This keeps :meth:`Agent.invoke` from
    having to short-circuit the bind when the caller passes no
    ``episode_id`` for a fresh orchestrator run.
    """
    token = _PARENT_EPISODE_ID.set(episode_id)
    try:
        yield
    finally:
        _PARENT_EPISODE_ID.reset(token)


def update_parent_episode_id(episode_id: str) -> None:
    """Overwrite the currently-bound parent-episode id in place.

    :meth:`Agent.invoke` enters
    :func:`bind_parent_episode_id` at run start, but passes ``None`` when
    the caller did not supply one — TensorZero only allocates the real
    episode id on the first inference response.  Without a refresh,
    every sub-agent invocation during that same orchestrator run would
    tag ``t0_agents.parent_episode_id=unknown`` even after the id is
    known.

    Must be called inside an enclosing :func:`bind_parent_episode_id`
    context.  The returned :class:`~contextvars.Token` is intentionally
    discarded: :meth:`ContextVar.reset` on the outer bind's token
    reverts the ContextVar to its pre-bind value regardless of how
    many intermediate ``set`` calls landed in between, so the outer
    :func:`bind_parent_episode_id` still owns cleanup on exit.  This
    is a documented Python semantic (``contextvars`` module docs), not
    a quirk of this module.
    """
    _PARENT_EPISODE_ID.set(episode_id)


def install_sub_agents(parent: Agent) -> None:
    """Register each sub-agent as a tool on the orchestrator's server.

    No-op when :attr:`AgentConfig.sub_agents` is empty so single-agent
    constructions behave exactly as if the submodule were absent.
    Raises
    :class:`ConfigurationError` on any of:

    * a non-``fastmcp.Client`` ``tool_provider``;
    * a ``fastmcp.Client`` whose transport is not in-process;
    * duplicate ``config.name`` across the sub-agent list;
    * a tool name already present on the orchestrator's FastMCP server
      (declared via ``@mcp.tool`` or a prior ``add_tool`` call).

    Called from :meth:`Agent.__init__`; direct invocation by user code
    is supported but unusual.
    """
    sub_agents = parent.config.sub_agents
    if not sub_agents:
        return

    assert_unique_names(sub_agents)

    client = parent._tool_provider
    if not isinstance(client, Client):
        raise ConfigurationError(
            "AgentConfig.sub_agents only supports a fastmcp.Client as "
            f"tool_provider, got {type(client).__name__}. Remote tool "
            "providers cannot host runtime tools."
        )

    transport = getattr(client, "transport", None)
    if not isinstance(transport, FastMCPTransport):
        raise ConfigurationError(
            "AgentConfig.sub_agents requires the fastmcp.Client to use "
            "an in-process FastMCPTransport; remote MCP transports "
            f"cannot host runtime tools (got {type(transport).__name__})."
        )
    server = transport.server

    # FastMCP's default ``on_duplicate="warn"`` means ``add_tool`` will
    # silently replace an existing tool with a log warning instead of
    # raising.  We need the eager failure, so enumerate the existing
    # tools synchronously — ``LocalProvider._list_tools`` is async-only,
    # so we read the underlying ``_components`` dict directly.  No
    # synchronous public API exposes this; a future FastMCP rename of
    # ``_components`` would make the pre-check silently no-op, so we
    # log a warning to keep that failure mode loud.  The ValueError
    # catch on ``add_tool`` below is the second line of defence when
    # the server is configured with ``on_duplicate="error"``.
    existing_names: set[str] = set()
    local_provider = getattr(server, "local_provider", None)
    components = getattr(local_provider, "_components", None)
    if components is not None:
        existing_names = {
            c.name for c in components.values() if isinstance(c, _FastMCPTool)
        }
    else:
        logger.warning(
            "fastmcp.FastMCP.local_provider._components not found "
            "(local_provider=%r); sub-agent tool-name collision "
            "pre-check skipped. Pin fastmcp or file a bug if this "
            "persists — collisions will only surface if the server "
            "uses on_duplicate='error'.",
            local_provider,
        )

    for sub_agent in sub_agents:
        tool_name = default_tool_name(sub_agent)
        if tool_name in existing_names:
            raise ConfigurationError(
                f"Sub-agent tool {tool_name!r} collides with an "
                "existing tool on the orchestrator's FastMCP server. "
                "Rename the sub-agent or remove the existing tool."
            )
        tool = agent_as_tool(sub_agent)
        # ``add_tool(tool: Tool | Callable[..., Any])`` — mypy strict
        # can't see the ``Tool`` branch (no FastMCP stubs + no @overload),
        # so we pick the callable side explicitly.  Runtime dispatch is
        # unchanged.
        try:
            server.add_tool(cast(Callable[..., Any], tool))
        except ValueError as exc:
            # ``LocalProvider._add_component`` raises ``ValueError`` on
            # duplicate keys when the provider is configured with
            # ``on_duplicate="error"``.  Defence-in-depth: catch and
            # re-raise as ``ConfigurationError`` so callers see the
            # predictable t0-agents type either way.
            raise ConfigurationError(
                f"Sub-agent tool {tool_name!r} could not be registered "
                f"on the orchestrator's FastMCP server: {exc}"
            ) from exc
        existing_names.add(tool_name)


class _OrchestrationToolProvider:
    """Dispatch sub-agent tool calls in the orchestrator's asyncio task.

    FastMCP's in-process transport
    (``fastmcp/client/transports/memory.py::FastMCPTransport.connect_session``)
    runs the server in an anyio sub-task spawned by ``tg.start_soon``
    at ``async with Client(server):`` entry.  That sub-task snapshots
    the ``contextvars.Context`` at creation time, which happens
    **before** :meth:`Agent.invoke` can bind the parent-episode id via
    :func:`bind_parent_episode_id` / update it via the on-inference
    refresh hook.  A :class:`~contextvars.ContextVar`-only handoff
    therefore reads the pre-run default (``None``) inside the
    server-side handler, and every sub-agent inference tags
    ``t0_agents.parent_episode_id=unknown`` regardless of fixes in
    the orchestrator's task — confirmed against a live ClickHouse
    spot-check where all 16 sub-agent rows came back with
    ``parent=['unknown']``.

    This wrapper sits between :func:`~t0_agents.mcp.loop.run_agent`
    and the orchestrator's :class:`fastmcp.Client`.  When the tool
    name matches a registered sub-agent, the call is dispatched
    inline — same asyncio task as :meth:`Agent.invoke` — so
    :func:`get_parent_episode_id` reads the live binding set by the
    ``on_inference`` refresh wrapper.  All other tool calls fall
    through to the inner provider unchanged, preserving the MCP
    semantics for non-delegation tools.  The server-side handler in
    :mod:`t0_agents.compose.delegation` remains installed on the
    FastMCP server as a fallback for external callers that connect
    directly (those genuinely have no orchestrator context and
    correctly surface the ``"unknown"`` tag).
    """

    def __init__(
        self,
        inner: ToolProvider,
        sub_agents: dict[str, Agent],
    ) -> None:
        self._inner = inner
        self._sub_agents = sub_agents

    async def list_tools(self) -> list[MCPTool]:
        return cast(list[MCPTool], await self._inner.list_tools())

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CallToolResult:
        sub_agent = self._sub_agents.get(name or "")
        if sub_agent is None:
            return cast(
                CallToolResult,
                await self._inner.call_tool(
                    name, arguments or {}, timeout=timeout, **kwargs,
                ),
            )

        args = arguments or {}
        message = args.get("message")
        if not isinstance(message, str):
            # Mirror FastMCP's schema-validation error surface so the
            # orchestrator LLM sees a consistent tool-error payload and
            # can retry with the right argument shape.
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"sub-agent tool {name!r} requires a "
                            f"'message: str' argument; got {args!r}"
                        ),
                    ),
                ],
                structured_content=None,
                meta=None,
                data=None,
                is_error=True,
            )

        parent_episode_id = get_parent_episode_id() or "unknown"
        delegation_tags = {PARENT_EPISODE_ID_TAG: parent_episode_id}
        try:
            # Dispatch via Agent.invoke().result() so the sub-agent
            # runs its own middleware chain.
            result = await sub_agent.invoke(message, tags=delegation_tags).result()
        except T0AgentsFatalError:
            # Hybrid-fatal: T0AgentsFatalError subclasses (e.g.
            # GuardrailBlockedError raised by the sub-agent's own stack)
            # propagate unchanged so hard blocks reach the caller of
            # Agent.invoke with their original type.
            raise
        except Exception as exc:
            raise SubAgentError(
                f"Sub-agent {sub_agent.config.name!r} failed: {exc}"
            ) from exc
        return CallToolResult(
            content=[TextContent(type="text", text=result.response)],
            structured_content=None,
            meta=None,
            data=None,
            is_error=False,
        )
