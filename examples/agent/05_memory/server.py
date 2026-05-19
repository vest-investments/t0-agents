"""Example MCP server with remember/recall tools.

These tools are *agent-visible* (the LLM calls them); they are separate
from the framework-managed ``MemoryProvider`` that persists state
between ``Agent.invoke()`` calls transparently.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("memory-demo")

# Simple key-value store the LLM can use as a scratchpad.
_store: dict[str, str] = {}


@mcp.tool()
def remember(key: str, value: str) -> str:
    """Store a value under a key for later retrieval."""
    _store[key] = value
    return f"Stored: {key}={value}"


@mcp.tool()
def recall(key: str) -> str:
    """Retrieve a previously stored value by key."""
    if key in _store:
        return _store[key]
    return f"No value found for key: {key}"
