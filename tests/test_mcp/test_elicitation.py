"""Smoke tests for MCP Elicitation (Phase 0).

Validates that FastMCP's elicitation mechanism works end-to-end with
the in-memory test transport — no production code changes needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError

from tests.server import mcp


async def test_smoke_approve() -> None:
    """Handler accepts approval → tool returns 'Approved: deploy'."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("approve_action", {"action": "deploy"})
        assert result.data == "Approved: deploy"


async def test_smoke_structured_input() -> None:
    """Handler returns structured data → tool returns greeting."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={"name": "Alice", "age": 30})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("collect_info", {})
        assert result.data == "Hello Alice, age 30"


async def test_smoke_decline() -> None:
    """Handler declines → tool returns rejection message."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("approve_action", {"action": "deploy"})
        assert result.data == "Rejected: deploy"


async def test_smoke_no_handler() -> None:
    """No elicitation handler → ToolError raised from call_tool().

    The server-side McpError is wrapped by FastMCP into a
    CallToolResult(isError=True); the client then raises ToolError
    (not McpError) because raise_on_error defaults to True.
    """
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="Elicitation not supported"):
            await client.call_tool("approve_action", {"action": "deploy"})
