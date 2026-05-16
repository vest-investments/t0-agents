"""System prompt example: demonstrates how to configure a TensorZero system
prompt that guides the LLM's tool-use behavior.

Prerequisites:
    1. Set GROQ_API_KEY in your environment.
    2. Start the gateway::

           docker compose -f examples/02_agent_class/docker-compose.yml up -d

    3. Run::

           uv run python examples/02_agent_class/intermediate.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import run_agent

# The directory name starts with a digit (03_...) so it can't be a normal
# Python import.  Add the example dir to sys.path so ``server`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import mcp  # type: ignore[import-not-found]

# Enable INFO logging to see the agent loop in action.
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    async with await gateway as t0, mcp_client:
        result = await run_agent(
            t0,
            mcp_client,
            "What's the weather in Paris, and convert that temperature to Fahrenheit?",
            "agent",
            max_turns=6,
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Response source: {result.response_source}")
        print(f"Episode ID: {result.episode_id}")


if __name__ == "__main__":
    asyncio.run(main())
