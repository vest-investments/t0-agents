"""Multi-server composition: combine tools from multiple MCP servers.

Demonstrates FastMCP's mount() to compose servers with namespace prefixes.
The agent sees all tools as a flat list (math_add, math_multiply, text_word_count, text_reverse).

Prerequisites:
    1. Set GROQ_API_KEY in your environment.
    2. Start the gateway:
         docker compose --env-file .env \\
             -f examples/agent/14_multi_server/docker-compose.yml up -d
    3. Run: uv run python examples/agent/14_multi_server/main.py
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from t0_agents.mcp import convert_tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server_math import mcp as math_mcp  # type: ignore[import-not-found]
from server_text import mcp as text_mcp  # type: ignore[import-not-found]

# Compose servers with namespace prefixes
composed = FastMCP("ComposedServer")
composed.mount(math_mcp, namespace="math")
composed.mount(text_mcp, namespace="text")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(composed)

    async with await gateway as t0, client:
        # Show the namespaced tool list
        tools = await client.list_tools()
        t0_tools = convert_tools(tools)
        print("Available tools:")
        for tool in t0_tools:
            print(f"  {tool['name']}: {tool['description']}")

        # Build the Agent and ask a question that requires both math and text tools
        agent = Agent(
            AgentConfig(
                name="composed-tools-agent",
                function_name="agent",
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, client),
        )
        result = await agent.invoke(
            "How many words are in 'The quick brown fox jumps over the lazy dog'? "
            "Also, what is 7 multiplied by 6?"
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")


if __name__ == "__main__":
    asyncio.run(main())
