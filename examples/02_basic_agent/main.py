"""Basic agent: minimal end-to-end run_agent loop.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway: docker compose -f examples/02_basic_agent/docker-compose.yml up -d
    3. Run: uv run python examples/02_basic_agent/main.py
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import run_agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]


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
            "What is the weather in Tokyo?",
            "agent",
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Episode ID: {result.episode_id}")
        print(f"Inference IDs: {result.inference_ids}")
        print(f"Response source: {result.response_source}")


if __name__ == "__main__":
    asyncio.run(main())
