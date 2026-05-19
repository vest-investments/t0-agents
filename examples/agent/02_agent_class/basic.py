"""Agent class: declarative agent with AgentConfig.

The declarative Agent wrapper. Configure once with AgentConfig, then
call agent.invoke(message) with no boilerplate per call.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/agent/02_agent_class/docker-compose.yml up -d
    3. Run: uv run python examples/agent/02_agent_class/basic.py
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
    # 1. Configure the agent once.
    config = AgentConfig(
        name="weather-assistant",
        function_name="agent",
        max_turns=10,
        tags={"example": "02_agent_class", "tier": "basic"},
    )

    # 2. Build the collaborators (caller owns their lifecycle).
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    # 3. Run the agent.
    async with await gateway as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        result = await agent.invoke("What is the weather in Tokyo?")

    # 4. Inspect the result.
    print(f"\nResponse: {result.response}")
    print(f"Turns: {result.turns}")
    print(f"Episode ID: {result.episode_id}")
    print(f"Inference IDs: {result.inference_ids}")
    print(f"Response source: {result.response_source}")


if __name__ == "__main__":
    asyncio.run(main())
