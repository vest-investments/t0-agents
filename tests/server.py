"""Shared test FastMCP server with sample tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastmcp import Context, FastMCP
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
# exercise the None-description fallback in convert_tools.
@mcp.tool
def echo_no_docs(value: str) -> str:
    return value


# --- Robustness test fixtures ----------------------------------------------
# These tools are deliberately degenerate: they exist to exercise the error
# contract in `src/t0_agents/mcp/loop.py`, not to model realistic tool logic.


@mcp.tool
def fail_always(reason: str) -> str:
    """Always raise ToolError with the caller-supplied reason.

    FastMCP converts unhandled exceptions raised in a tool function into an
    MCP error response; ``Client.call_tool(..., raise_on_error=True)`` (the
    default) then re-raises it as ``fastmcp.exceptions.ToolError`` on the
    client side. This tool is used by the robustness and error-handling test suite.
    """
    raise ToolError(f"fail_always: {reason}")


@mcp.tool
async def slow_tool(delay_seconds: float) -> str:
    """Sleep for ``delay_seconds`` and return ``"done"``.

    Pair with ``Client.call_tool(..., timeout=N)`` where ``N < delay_seconds``
    to force the protocol-level timeout → ``McpError`` code ``-32000`` path
    that ``_execute_single_tool_call`` handles. Async so ``asyncio.sleep``
    actually yields rather than blocking the event loop.
    """
    await asyncio.sleep(delay_seconds)
    return "done"


# --- Elicitation test fixtures ------------------------------------------------
# These tools exercise MCP Elicitation (human-in-the-loop approval and
# structured input).  They require a Client created with an
# `elicitation_handler` — without one the default MCP callback returns
# McpError(-32600, "Elicitation not supported").


@mcp.tool
async def approve_action(action: str, ctx: Context) -> str:
    """Request approval before performing an action."""
    result = await ctx.elicit(f"Approve: {action}?", response_type=None)
    if result.action == "accept":
        return f"Approved: {action}"
    return f"Rejected: {action}"


@dataclass
class UserInfo:
    """Flat dataclass for structured elicitation (name + age)."""

    name: str
    age: int


@mcp.tool
async def collect_info(ctx: Context) -> str:
    """Collect user information via elicitation."""
    result = await ctx.elicit(
        "Please provide your information",
        response_type=UserInfo,
    )
    if result.action == "accept":
        return f"Hello {result.data.name}, age {result.data.age}"
    return "Info not provided"
