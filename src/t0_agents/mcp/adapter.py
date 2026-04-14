"""Adapter between FastMCP tool types and TensorZero tool format."""

from __future__ import annotations

import json
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

    TensorZero's ``ToolResult.result`` is always a string, so every branch of
    this function must produce one. The serialization strategy handles each case:

    * ``is_error=True`` → ``"ERROR: "`` + joined ``.content[*].text``
      (``"Unknown tool error"`` if no textual block is present). Callers should
      invoke ``client.call_tool(..., raise_on_error=False)`` so that FastMCP
      surfaces errors through the result rather than as exceptions.
    * ``.data: str`` → passthrough.
    * ``.data`` anything else (primitives, dicts, lists, and complex types via
      ``default=str``) → ``json.dumps(data, default=str,
      ensure_ascii=False)``. Booleans become JSON-cased (``"true"`` /
      ``"false"``); ``datetime`` and ``UUID`` are stringified by ``default=str``
      and embedded as JSON strings.
    * ``.data is None`` → concatenate ``.text`` from every content block
      that exposes it, joined with ``"\\n"``. Non-textual blocks
      (``ImageContent``, ``AudioContent``, ``EmbeddedResource``) are skipped
      intentionally.

    The ``try``/``except`` around ``json.dumps`` is a defensive guard for the
    rare case where ``default=str`` cannot rescue an unserializable type (e.g.
    an object whose ``__str__`` raises): we fall through to the content-block
    fallback rather than letting the loop crash.
    """
    if result.is_error:
        error_texts = [
            block.text for block in result.content if hasattr(block, "text")
        ]
        message = "\n".join(error_texts) if error_texts else "Unknown tool error"
        return f"ERROR: {message}"

    data = result.data
    if data is not None:
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            pass  # fall through to the content-block fallback

    text_parts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join(text_parts)


def build_tool_result(tool_call: ToolCall, mcp_result: CallToolResult) -> ToolResult:
    """Build a TensorZero ``ToolResult`` from a ``ToolCall`` and a FastMCP result.

    ``ToolCall.name`` is ``Optional[str]`` (``None`` when the LLM hallucinated a
    name that is not in the schema), but ``ToolResult.name`` must be ``str``.
    We fall back to ``tool_call.raw_name`` — the raw string produced by the
    model, which is always populated — so T0 always receives a string.
    Upstream callers that want to reject hallucinations before execution (see
    ``execute_tool_calls``) should do so before reaching this function; this
    function only needs to produce a well-shaped ``ToolResult``.
    """
    return ToolResult(
        name=tool_call.name or tool_call.raw_name,
        result=serialize_tool_result(mcp_result),
        id=tool_call.id,
    )
