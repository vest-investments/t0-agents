"""Math tools MCP server."""
from fastmcp import FastMCP

mcp = FastMCP("MathServer")


@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b
