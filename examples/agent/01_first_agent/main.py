"""01 — Your first agent: a coffee shop concierge.

A single user message, two MCP tools, one final answer.  The LLM
figures out it needs both ``get_specialty_today`` and
``estimate_wait_minutes``, picks the order itself, and synthesises a
two-clause reply.

Prerequisites:
    1. ``GROQ_API_KEY=...`` in ``.env`` at the project root.
    2. Boot the gateway::

           docker compose -f examples/agent/01_first_agent/docker-compose.yml \\
               --env-file .env up -d

    3. Run::

           uv run python examples/agent/01_first_agent/main.py
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    async with await gateway as t0, mcp_client:
        agent = Agent(
            AgentConfig(name="barista", function_name="agent"),
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        # invoke() returns a Run — awaitable for the final result.
        result = await agent.invoke(
            "What should I order, and how long will I wait?"
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Episode ID: {result.episode_id}")
        print(f"Inference IDs: {result.inference_ids}")


if __name__ == "__main__":
    asyncio.run(main())
