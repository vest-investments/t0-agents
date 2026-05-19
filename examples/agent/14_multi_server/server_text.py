"""Text processing MCP server."""
from fastmcp import FastMCP

mcp = FastMCP("TextServer")


@mcp.tool
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


@mcp.tool
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]
