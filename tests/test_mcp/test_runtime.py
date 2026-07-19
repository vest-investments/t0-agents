"""MCPRuntime unit tests — handles, lifecycle, and the routing contract.

Every test drives through the real FastMCP boundary (in-memory servers via
``Client(FastMCP)``), never a hand-built provider fake — framework-layer
regressions (schema wrapping, elicitation capability negotiation) surface
here or not at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from t0_agents.mcp.elicitation import auto_approve_handler
from t0_agents.mcp.runtime import MCPRuntime


def _local_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("local")

    @server.tool
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool
    async def confirm(ctx: Context) -> str:
        """Ask for binary approval."""
        result = await ctx.elicit("Approve?", response_type=None)
        return "approved" if result.action == "accept" else "declined"

    return server


async def test_local_server_tools_listed_verbatim() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
    assert {"add", "confirm"} <= names


async def test_call_tool_routes_to_local_server() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        result = await runtime.call_tool("add", {"a": 2, "b": 3})
    assert result.data == 5


async def test_unknown_tool_matches_composite_error_contract() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        with pytest.raises(ToolError, match="Unknown tool: 'nope'"):
            await runtime.call_tool("nope", {})


async def test_empty_runtime_lists_nothing_and_rejects_calls() -> None:
    async with MCPRuntime() as runtime:
        assert await runtime.list_tools() == []
        with pytest.raises(ToolError, match="Unknown tool"):
            await runtime.call_tool("anything", {})


async def test_connect_is_lazy_and_aclose_idempotent() -> None:
    runtime = MCPRuntime(local_server=_local_server())
    (handle,) = runtime._handles
    assert not handle.client.is_connected()
    async with runtime:
        assert not handle.client.is_connected()
        await runtime.call_tool("add", {"a": 1, "b": 1})
        assert handle.client.is_connected()
    assert not handle.client.is_connected()
    await runtime.aclose()


async def test_elicitation_handler_bound_to_local_handle() -> None:
    async with MCPRuntime(
        local_server=_local_server(), elicitation_handler=auto_approve_handler
    ) as runtime:
        result = await runtime.call_tool("confirm", {})
    assert result.data == "approved"
