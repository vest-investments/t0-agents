"""Tests for the in-memory glow-MCP testing helpers."""

from __future__ import annotations

import pytest
from fastmcp import Context, FastMCP

from t0_agents.glow import MCPServer
from t0_agents.testing import ScriptedElicitation, glow_tools


async def test_glow_tools_surfaces_mounted_tool() -> None:
    server = FastMCP("calc")

    @server.tool
    async def ping() -> str:
        """Trivial tool."""
        return "pong"

    async with glow_tools(MCPServer(server=server, namespace="calc")) as tools:
        names = {t.name for t in await tools.list_tools()}

    assert "calc_ping" in names


async def test_glow_tools_calls_through_the_fastmcp_boundary() -> None:
    server = FastMCP("calc")

    @server.tool
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    async with glow_tools(MCPServer(server=server, namespace="calc")) as tools:
        result = await tools.call_tool("calc_add", {"a": 2, "b": 3})

    assert result.data == 5


async def test_glow_tools_mounts_multiple_servers() -> None:
    calc = FastMCP("calc")
    text = FastMCP("text")

    @calc.tool
    async def double(n: int) -> int:
        """Double an integer."""
        return n * 2

    @text.tool
    async def shout(s: str) -> str:
        """Upper-case a string."""
        return s.upper()

    async with glow_tools(
        MCPServer(server=calc, namespace="calc"),
        MCPServer(server=text, namespace="text"),
    ) as tools:
        names = {t.name for t in await tools.list_tools()}

    assert {"calc_double", "text_shout"} <= names


async def test_glow_tools_rejects_empty_mount() -> None:
    with pytest.raises(ValueError, match="at least one MCP server"):
        async with glow_tools():
            pass


def _hitl_server() -> FastMCP:
    server = FastMCP("hitl")

    @server.tool
    async def approve(ctx: Context) -> str:
        """Binary approval (response_type=None)."""
        result = await ctx.elicit("Approve?", response_type=None)
        return "approved" if result.action == "accept" else "rejected"

    @server.tool
    async def place(ctx: Context) -> str:
        """Bool-scalar confirm (hidden `value`)."""
        result = await ctx.elicit("Place the order?", response_type=bool)
        if result.action == "accept" and result.data is True:
            return "placed"
        return "skipped"

    return server


async def test_scripted_elicitation_accepts_binary_approval() -> None:
    scripted = ScriptedElicitation(["accept"])

    async with glow_tools(
        MCPServer(server=_hitl_server(), namespace="hitl"),
        elicitation_handler=scripted.handler,
    ) as tools:
        result = await tools.call_tool("hitl_approve", {})

    assert result.data == "approved"
    assert len(scripted.received) == 1
    assert scripted.received[0]["response_type"] is None


async def test_scripted_elicitation_declines() -> None:
    scripted = ScriptedElicitation(["decline"])

    async with glow_tools(
        MCPServer(server=_hitl_server(), namespace="hitl"),
        elicitation_handler=scripted.handler,
    ) as tools:
        result = await tools.call_tool("hitl_approve", {})

    assert result.data == "rejected"


async def test_scripted_elicitation_accepts_scalar_with_content() -> None:
    scripted = ScriptedElicitation([{"action": "accept", "content": {"value": True}}])

    async with glow_tools(
        MCPServer(server=_hitl_server(), namespace="hitl"),
        elicitation_handler=scripted.handler,
    ) as tools:
        result = await tools.call_tool("hitl_place", {})

    assert result.data == "placed"
    assert scripted.received[0]["response_type"] == "bool"


async def test_scripted_elicitation_consumes_responses_in_order() -> None:
    server = FastMCP("hitl")

    @server.tool
    async def two_step(ctx: Context) -> str:
        """Two sequential elicitations: name then confirm."""
        r1 = await ctx.elicit("Name?", response_type=str)
        r2 = await ctx.elicit("Confirm?", response_type=None)
        if r1.action == "accept" and r2.action == "accept":
            return f"done:{r1.data}"
        return "incomplete"

    scripted = ScriptedElicitation(
        [{"action": "accept", "content": {"value": "Ada"}}, "accept"]
    )

    async with glow_tools(
        MCPServer(server=server, namespace="hitl"),
        elicitation_handler=scripted.handler,
    ) as tools:
        result = await tools.call_tool("hitl_two_step", {})

    assert result.data == "done:Ada"
    assert len(scripted.received) == 2
