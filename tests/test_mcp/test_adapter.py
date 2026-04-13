"""Unit tests for the MCP tool adapter."""

from __future__ import annotations

from typing import Any

from fastmcp import Client

from t0_agents.mcp import convert_tools


class TestConvertTools:
    async def test_list_tools_returns_valid_structure(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.1: list_tools() returns objects with .name, .description, .inputSchema."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        assert len(tools) >= 2
        for tool in tools:
            assert isinstance(tool.name, str) and tool.name
            assert tool.description is None or isinstance(tool.description, str)
            assert isinstance(tool.inputSchema, dict)
            assert tool.inputSchema.get("type") == "object"

    async def test_convert_tools_produces_expected_shape(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.2: adapter maps list[Tool] → list[dict] with name/description/parameters."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        t0_tools = convert_tools(tools)
        assert len(t0_tools) == len(tools)

        for t0_tool in t0_tools:
            # Exactly these keys — no leaks of FastMCP-specific fields.
            assert set(t0_tool.keys()) == {"name", "description", "parameters"}
            assert isinstance(t0_tool["name"], str) and t0_tool["name"]
            # After the adapter, description is always a string (F1.6).
            assert isinstance(t0_tool["description"], str)
            assert isinstance(t0_tool["parameters"], dict)
            assert t0_tool["parameters"].get("type") == "object"

        # Sanity-check names carry through unchanged.
        names = {t["name"] for t in t0_tools}
        assert {"add", "greet", "search", "echo_no_docs"}.issubset(names)

    async def test_convert_tools_handles_none_description(
        self, mcp_client: Client[Any]
    ) -> None:
        """F1.6: tool with description=None at the FastMCP layer becomes ""."""
        async with mcp_client:
            tools = await mcp_client.list_tools()

        # FastMCP leaves description as None when there is no docstring and no
        # explicit description= on the decorator. This is the precondition for
        # F1.6 to be meaningful — if it changes upstream, the adapter is still
        # correct but this assertion anchors the scenario we're testing.
        echo = next(t for t in tools if t.name == "echo_no_docs")
        assert echo.description is None

        t0_tools = convert_tools(tools)
        echo_t0 = next(t for t in t0_tools if t["name"] == "echo_no_docs")
        assert echo_t0["description"] == ""


class TestSerializeToolResult:
    """Phase 2 — not yet implemented."""


class TestBuildToolResult:
    """Phase 2 — not yet implemented."""
