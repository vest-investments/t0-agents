"""Feedback loop: capture observability data and send feedback to TensorZero.

Demonstrates:
- episode_id and inference_ids tracking from AgentResult
- Sending boolean, float, and comment feedback via the TensorZero SDK
- Structured logging configuration
- The dryrun caveat (dryrun=True IDs are ephemeral; feedback against them fails)

Prerequisites:
    1. Set GROQ_API_KEY in your environment.
    2. Start the gateway:
         docker compose --env-file .env \\
             -f examples/agent/13_observability/docker-compose.yml up -d
    3. Run: uv run python examples/agent/13_observability/main.py
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

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

# Enable logging to see the agent loop details
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(mcp)

    async with await gateway as t0, client:
        agent = Agent(
            AgentConfig(
                name="docs-search-agent",
                function_name="agent",
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, client),
        )
        # Default dryrun=False so the inference IDs are stored in ClickHouse
        # and feedback calls against them can land.
        result = await agent.invoke(
            "Search the docs for 'installation' and show me the Getting Started content."
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Episode ID: {result.episode_id}")
        print(f"Inference IDs: {result.inference_ids}")

        # Send boolean feedback (episode-level: was the task successful?)
        fb1 = await t0.feedback(
            episode_id=result.episode_id,
            metric_name="task_success",
            value=True,
            dryrun=False,
        )
        print(f"\nBoolean feedback sent: {fb1.feedback_id}")

        # Send float feedback (inference-level: quality of the last turn)
        last_inference_id = result.inference_ids[-1]
        fb2 = await t0.feedback(
            inference_id=last_inference_id,
            metric_name="quality_score",
            value=0.9,
            dryrun=False,
        )
        print(f"Float feedback sent: {fb2.feedback_id}")

        # Send comment feedback (free-form text for review)
        fb3 = await t0.feedback(
            episode_id=result.episode_id,
            metric_name="comment",
            value="Agent correctly found and displayed the Getting Started doc.",
            dryrun=False,
        )
        print(f"Comment feedback sent: {fb3.feedback_id}")

        print("\nFeedback is now stored in ClickHouse.")
        print("TensorZero can use this data to optimize model selection and prompts.")


if __name__ == "__main__":
    asyncio.run(main())
