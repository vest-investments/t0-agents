"""Example FastMCP server used by the HTTP agent demo."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("HttpAgentDemo")


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22C."


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
