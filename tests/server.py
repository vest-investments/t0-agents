"""Shared test FastMCP server with sample tools."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
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
# exercise the None-description fallback in convert_tools (F1.6).
@mcp.tool
def echo_no_docs(value: str) -> str:
    return value
