"""Shared test FastMCP server with sample tools."""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel

mcp = FastMCP("TestServer")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class SearchParams(BaseModel):
    """Search parameters with a query string and a result limit."""

    query: str
    limit: int = 10


@mcp.tool
def search(params: SearchParams) -> list[dict[str, Any]]:
    """Search using a structured Pydantic parameter model."""
    return [{"query": params.query, "limit": params.limit}]


# `echo_no_docs` intentionally has no docstring and no explicit description on
# the decorator so FastMCP leaves `tool.description` as `None`. It exists to
# exercise the None-description fallback in convert_tools (F1.6 / F5.8).
@mcp.tool
def echo_no_docs(value: str) -> str:
    return value


# --- Phase 5 robustness fixtures ------------------------------------------
# These tools are deliberately degenerate: they exist to exercise the error
# contract in `src/t0_agents/mcp/loop.py`, not to model realistic tool logic.


@mcp.tool
def fail_always(reason: str) -> str:
    """Always raise ToolError with the caller-supplied reason (F5.1 anchor).

    FastMCP converts unhandled exceptions raised in a tool function into an
    MCP error response; ``Client.call_tool(..., raise_on_error=True)`` (the
    default) then re-raises it as ``fastmcp.exceptions.ToolError`` on the
    client side. This tool is the canonical hook for F5.1, F5.5, and F5.6
    integration tests.
    """
    raise ToolError(f"fail_always: {reason}")


@mcp.tool
async def slow_tool(delay_seconds: float) -> str:
    """Sleep for ``delay_seconds`` and return ``"done"`` (F5.4 anchor).

    Pair with ``Client.call_tool(..., timeout=N)`` where ``N < delay_seconds``
    to force the protocol-level timeout → ``McpError`` code ``-32000`` path
    that ``_execute_single_tool_call`` handles. Async so ``asyncio.sleep``
    actually yields rather than blocking the event loop.
    """
    await asyncio.sleep(delay_seconds)
    return "done"
