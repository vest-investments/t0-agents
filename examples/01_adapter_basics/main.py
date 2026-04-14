"""Adapter basics: explore how t0-agents converts FastMCP tools to TensorZero format.

No Docker, no API keys, no gateway required.

    uv run python examples/01_adapter_basics/main.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

from t0_agents.mcp import convert_tools, serialize_tool_result

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]


async def main() -> None:
    async with Client(mcp) as client:
        # 1. Discover tools from the MCP server
        mcp_tools = await client.list_tools()
        print(f"Discovered {len(mcp_tools)} tool(s):\n")
        for tool in mcp_tools:
            print(f"  {tool.name}: {tool.description}")

        # 2. Convert to TensorZero format (what gets sent to the gateway)
        t0_tools = convert_tools(mcp_tools)
        print("\nTensorZero-compatible schemas:\n")
        for t0_tool in t0_tools:
            print(json.dumps(t0_tool, indent=2))

        # 3. Call a tool and serialize the result
        result = await client.call_tool("add", {"a": 3, "b": 7})
        serialized = serialize_tool_result(result)
        print(f"\nadd(3, 7) = {serialized}")
        print(f"  type: {type(serialized).__name__} (always str for TensorZero)")

        result2 = await client.call_tool("get_weather", {"city": "Tokyo"})
        serialized2 = serialize_tool_result(result2)
        print(f"\nget_weather('Tokyo') = {serialized2}")


if __name__ == "__main__":
    asyncio.run(main())
