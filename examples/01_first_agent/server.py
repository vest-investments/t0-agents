"""FastMCP server: the coffee shop concierge's two tools."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("CoffeeShopConcierge")


@mcp.tool
def get_specialty_today() -> str:
    """Today's specialty coffee at the shop."""
    return "Ethiopian Yirgacheffe pour-over"


@mcp.tool
def estimate_wait_minutes() -> int:
    """How many minutes the queue is right now."""
    return 7
