"""MCP server with deliberately failing and slow tools for error handling demos."""
from __future__ import annotations

import asyncio

from fastmcp import FastMCP

mcp = FastMCP("ErrorDemoServer")


@mcp.tool
def get_data(query: str) -> str:
    """Fetch data for a query (always succeeds)."""
    return f"Results for '{query}': [item_1, item_2, item_3]"


@mcp.tool
def unreliable_api(endpoint: str) -> str:
    """Call an external API that always fails (for demo purposes)."""
    raise ValueError(f"Connection refused: {endpoint} is unavailable")


@mcp.tool
async def slow_process(seconds: float) -> str:
    """A process that takes a while to complete."""
    await asyncio.sleep(seconds)
    return f"Completed after {seconds}s"
