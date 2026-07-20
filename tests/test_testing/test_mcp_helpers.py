"""Tests for the in-memory glow-MCP testing helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from t0_agents.glow import MCPServer
from t0_agents.testing import glow_tools


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
