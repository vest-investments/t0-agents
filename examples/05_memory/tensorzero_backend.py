"""TensorZero-backed memory example with the ``messages`` strategy.

Drives two turns of a conversation through the same ``episode_id``.  The
first turn plants a fact; the second one reuses the episode and the
agent recalls it via TensorZero's own observability log — no Postgres,
no SQLite, no in-process state.  The native ``messages`` strategy
threads the prior conversation through ``input.messages`` so the user
text is never inflated with a rendered summary.

Pre-requisites:

    docker compose up -d              # boots TZ gateway + ClickHouse
    GROQ_API_KEY in .env

This example ships in the base distribution because ``tensorzero`` is
already a required dependency — there is nothing extra to install.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from fastmcp import Client
from server import mcp
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import TensorZeroMemory

CUSTOMER_ID = "demo-tensorzero-backend"


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0, Client(mcp) as client:
        memory = TensorZeroMemory(
            t0,
            function_name="agent",
            tag_filter={"customer_id": CUSTOMER_ID},
        )
        agent = Agent(
            AgentConfig(
                name="tensorzero-memory-demo",
                function_name="agent",
                memory=memory,
                memory_strategy="messages",
                tags={"customer_id": CUSTOMER_ID},
            ),
            t0=t0,
            tool_provider=client,
        )

        print("=== Turn 1 (fresh episode) ===")
        r1 = await agent.invoke(
            "My favourite colour is teal. Please remember that.",
        )
        print(f"  episode_id: {r1.episode_id}")
        print(f"  reply:      {r1.response}\n")

        # Give TZ a moment to flush the inference to ClickHouse before the
        # next turn reads it back.  Production code should NOT need this
        # (the prior turn always lands before the next request arrives via
        # network); it is here only because back-to-back local calls can
        # outrun the gateway's async writer.
        await asyncio.sleep(1.5)

        print("=== Turn 2 (same episode) ===")
        r2 = await agent.invoke(
            "What is my favourite colour?",
            episode_id=r1.episode_id,
        )
        print(f"  reply:      {r2.response}")


if __name__ == "__main__":
    asyncio.run(main())
