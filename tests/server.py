"""Shared test FastMCP server with sample tools."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("TestServer")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
