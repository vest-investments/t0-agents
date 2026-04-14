"""Error handling: how the agent loop recovers from tool failures.

Demonstrates:
- Tool errors are caught and sent to the LLM as error strings
- The LLM can retry with different tools/arguments after seeing an error
- Timeout behavior with tool_timeout parameter
- MaxTurnsExceededError safety valve

Prerequisites:
    1. Set GROQ_API_KEY in your environment.
    2. Start the gateway: docker compose -f examples/05_error_handling/docker-compose.yml up -d
    3. Run: uv run python examples/05_error_handling/main.py
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

from t0_agents import MaxTurnsExceededError, run_agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(mcp)

    async with await gateway as t0, client:
        # Scenario 1: The LLM tries unreliable_api, sees the error, falls back to get_data
        print("=== Scenario 1: Error recovery ===")
        result = await run_agent(
            t0,
            client,
            "Try calling unreliable_api with endpoint '/users'. "
            "If it fails, use get_data instead with query 'users'.",
            "agent",
            max_turns=5,
        )
        print(f"Response: {result.response}")
        print(f"Turns used: {result.turns}\n")

        # Scenario 2: Timeout handling
        print("=== Scenario 2: Timeout ===")
        result2 = await run_agent(
            t0,
            client,
            "Run slow_process with 1 second delay, then get_data for 'status'.",
            "agent",
            tool_timeout=2.0,
            max_turns=5,
        )
        print(f"Response: {result2.response}")
        print(f"Turns used: {result2.turns}\n")

        # Scenario 3: MaxTurnsExceededError
        print("=== Scenario 3: Max turns safety ===")
        try:
            await run_agent(
                t0,
                client,
                "Keep calling unreliable_api with '/retry' until it works.",
                "agent",
                max_turns=3,
            )
        except MaxTurnsExceededError as e:
            print(f"Caught MaxTurnsExceededError: loop exceeded {e.max_turns} turns")
            print("This prevents infinite retry loops.\n")


if __name__ == "__main__":
    asyncio.run(main())
