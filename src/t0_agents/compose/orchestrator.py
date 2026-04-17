"""Install sub-agents on an orchestrator's in-process FastMCP server.

Registration happens at :meth:`Agent.__init__` time, **not** per run:
the orchestrator LLM sees the full set of delegated ``_ask`` tools
from the very first inference.  Dynamic mid-run registration is out
of scope — gateway-safe per T0 Q2, but the LLM-side cost of schema
changes mid-conversation isn't worth it yet
(see ``docs/internal/agentic-feature/02-open-questions.md``).

Source-validated correction vs the Phase G spec §4.3: the spec read
``client.local_provider`` on :class:`fastmcp.Client`, but that
attribute is a property on :class:`fastmcp.FastMCP` *servers*, not on
clients.  The in-process unwrap in fastmcp 2.x is
``client.transport.server`` when
``isinstance(client.transport, FastMCPTransport)``
(``.venv/.../fastmcp/client/transports/memory.py:29``).  Collision
detection relies on FastMCP's own
``ValueError("Component already exists: ...")`` from
``LocalProvider._add_component``
(``fastmcp/server/providers/local_provider/local_provider.py:190``),
which we translate into :class:`ConfigurationError` so callers see a
predictable t0-agents exception.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.tools import Tool as _FastMCPTool

from t0_agents.compose.delegation import agent_as_tool
from t0_agents.compose.routing import assert_unique_names, default_tool_name
from t0_agents.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


#: Tag key carried on every sub-agent inference so ClickHouse can
#: reconstruct the delegation tree via ``tags[PARENT_EPISODE_ID_TAG]``.
PARENT_EPISODE_ID_TAG = "t0_agents.parent_episode_id"

# Kept isolated per :class:`contextvars.ContextVar` so concurrent
# ``Agent.run`` calls on one orchestrator don't cross-contaminate each
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
    ``"unknown"`` correlation tag.  This keeps :meth:`Agent.run` from
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

    :meth:`Agent.run` / :meth:`Agent.stream` enter
    :func:`bind_parent_episode_id` at run start, but pass ``None`` when
    the caller did not supply one — TensorZero only allocates the real
    episode id on the first inference response.  Without a refresh,
    every sub-agent invocation during that same orchestrator run would
    tag ``t0_agents.parent_episode_id=unknown`` even after the id is
    known.  This helper replaces the value stored for the duration of
    the enclosing :func:`bind_parent_episode_id` context; the outer
    context still owns the reset on exit.
    """
    _PARENT_EPISODE_ID.set(episode_id)


def install_sub_agents(parent: Agent) -> None:
    """Register each sub-agent as a tool on the orchestrator's server.

    No-op when :attr:`AgentConfig.sub_agents` is empty so single-agent
    constructions keep Phase A byte-identity.  Raises
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
    # so we read the underlying ``_components`` dict directly.
    existing_names: set[str] = set()
    local_provider = getattr(server, "local_provider", None)
    components = getattr(local_provider, "_components", None)
    if components is not None:
        existing_names = {
            c.name for c in components.values() if isinstance(c, _FastMCPTool)
        }

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
