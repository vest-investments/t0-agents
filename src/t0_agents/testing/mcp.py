"""In-memory helpers for testing glow agents that mount MCP servers.

Driving a mounted MCP tool through the real fastmcp client boundary — the
integration test the guidelines call for, since an in-process
``registered_tools[name](**kwargs)`` call bypasses FastMCP's schema and
serialization layer and misses framework-level bugs — costs a few coupled
facts today: a stand-in gateway (no inference happens in a tool-surface test),
the App lifespan, and the private App-owned runtime.  :func:`glow_tools` hides
them behind one async context manager that yields the mounted tools as a
:class:`~t0_agents.core.types.ToolProvider`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, MCPServer

if TYPE_CHECKING:
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core.types import ToolProvider

__all__ = ["glow_tools"]


@asynccontextmanager
async def glow_tools(
    *servers: MCPServer | type[MCPServer],
    elicitation_handler: ElicitationHandler | None = None,
) -> AsyncIterator[ToolProvider]:
    """Mount ``servers`` on an in-memory glow App and yield their tools.

    The yielded :class:`~t0_agents.core.types.ToolProvider` is the App-owned
    MCP runtime: ``await tools.list_tools()`` returns the merged, namespace-
    prefixed inventory and ``await tools.call_tool(name, arguments)`` drives a
    real in-memory ``fastmcp.Client`` → server → schema → result round-trip.
    In-memory ``server=`` mounts connect with no sockets or transport; nothing
    is sent to TensorZero, so the gateway is a stand-in.

    Pass ``elicitation_handler=`` (for example
    :attr:`ScriptedElicitation.handler`) to test tools that call
    ``ctx.elicit(...)``.

    :param servers: ``MCPServer`` specs (instances or subclasses) to mount.
    :raises ValueError: if no server mounts any tool.
    """
    app = App(
        t0=cast(AsyncTensorZeroGateway, object()),
        servers=list(servers),
        elicitation_handler=elicitation_handler,
    )
    async with app.lifespan():
        provider = app._owned_client
        if provider is None:
            raise ValueError("glow_tools requires at least one MCP server")
        yield provider
