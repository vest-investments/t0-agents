"""Adapter between FastMCP tool types and TensorZero tool format."""

from __future__ import annotations

from typing import Any

from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool
from tensorzero import ToolCall, ToolResult


def convert_tools(mcp_tools: list[MCPTool]) -> list[dict[str, Any]]:
    """Convert FastMCP tools to the TensorZero ``additional_tools`` format.

    Mapping:

    * ``tool.name`` → ``"name"``
    * ``tool.description`` → ``"description"`` (with ``None`` coerced to ``""``)
    * ``tool.inputSchema`` → ``"parameters"``

    The TensorZero native inference API treats ``description`` as required, but
    FastMCP's ``mcp.types.Tool.description`` is ``Optional[str]`` — it is
    ``None`` when a ``@mcp.tool`` function has no docstring and no explicit
    ``description=`` on the decorator. We coerce ``None`` (and empty strings)
    to ``""`` defensively so T0 always receives a string.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        }
        for tool in mcp_tools
    ]


def serialize_tool_result(result: CallToolResult) -> str:
    """Serialize a FastMCP ``CallToolResult`` to the ``str`` T0 expects.

    Phase 2 — not yet implemented.
    """
    raise NotImplementedError("Phase 2")


def build_tool_result(tool_call: ToolCall, mcp_result: CallToolResult) -> ToolResult:
    """Build a TensorZero ``ToolResult`` from a ``ToolCall`` and a FastMCP result.

    Phase 2 — not yet implemented.
    """
    raise NotImplementedError("Phase 2")
