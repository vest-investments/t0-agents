"""Stateful Agent example — Milestone M2.

Demonstrates cross-session memory using both ``InMemoryMemory`` (ephemeral)
and ``SqliteMemory`` (persistent).  The agent remembers context across
successive ``.run()`` calls via the configured ``MemoryProvider``.

Requirements:
    docker compose up -d          # boots TensorZero gateway + ClickHouse
    GROQ_API_KEY in .env
    pip install 't0-agents[memory-sqlite]'   # for SqliteMemory
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import InMemoryMemory, SqliteMemory

from server import mcp  # type: ignore[import-untyped]


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[object] = Client(mcp)

    async with await gateway as t0, client:
        # --- InMemoryMemory (ephemeral) ----------------------------------
        print("=== InMemoryMemory ===")
        in_mem = InMemoryMemory()
        agent = Agent(
            AgentConfig(
                name="ephemeral-assistant",
                function_name="agent",
                memory=in_mem,
            ),
            t0=t0,
            tool_provider=client,
        )

        r1 = await agent.run("My name is Alice. Please remember that.")
        print(f"Turn 1: {r1.response}")
        print(f"  episode_id: {r1.episode_id}")

        r2 = await agent.run(
            "What is my name?",
            episode_id=r1.episode_id,
        )
        print(f"Turn 2: {r2.response}")

        records = await in_mem.load(r1.episode_id or "")
        print(f"  Memory records: {len(records)}")

        # --- SqliteMemory (persistent) -----------------------------------
        if SqliteMemory is not None:
            print("\n=== SqliteMemory ===")
            db_path = "./agent_memory.db"
            sqlite_mem = SqliteMemory(db_path)
            agent2 = Agent(
                AgentConfig(
                    name="persistent-assistant",
                    function_name="agent",
                    memory=sqlite_mem,
                ),
                t0=t0,
                tool_provider=client,
            )

            r3 = await agent2.run("Remember: the project deadline is Friday.")
            print(f"Turn 1: {r3.response}")
            print(f"  episode_id: {r3.episode_id}")

            # Simulate process restart — new SqliteMemory instance, same DB
            sqlite_mem2 = SqliteMemory(db_path)
            agent3 = Agent(
                AgentConfig(
                    name="persistent-assistant",
                    function_name="agent",
                    memory=sqlite_mem2,
                ),
                t0=t0,
                tool_provider=client,
            )

            r4 = await agent3.run(
                "When is the project deadline?",
                episode_id=r3.episode_id,
            )
            print(f"Turn 2 (new instance): {r4.response}")
        else:
            print(
                "\nSkipping SqliteMemory demo. Install with: "
                "pip install 't0-agents[memory-sqlite]'"
            )


if __name__ == "__main__":
    asyncio.run(main())
