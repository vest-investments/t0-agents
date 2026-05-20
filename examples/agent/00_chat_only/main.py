"""00 — Toolless chat: the absolute minimum agent.

One user message, one inference, one text response.  No MCP server,
no MCP client, no ``list_tools()`` roundtrip — just an ``Agent``
wired to a TensorZero gateway.

This is the *shape* a chat-only agent takes after v1.0 made the
``tool_provider`` parameter optional.  Compare with ``01_first_agent``
to see what tools add on top of this baseline.

Prerequisites:
    1. ``GROQ_API_KEY=...`` in ``.env`` at the project root.
    2. Boot the gateway::

           docker compose -f examples/agent/00_chat_only/docker-compose.yml \\
               --env-file .env up -d

    3. Run::

           uv run python examples/agent/00_chat_only/main.py
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        agent = Agent(
            AgentConfig(name="tutor", function_name="agent"),
            t0=t0,
        )
        # invoke() returns a Run — awaitable for the final result.
        result = await agent.invoke("Explain monads in two sentences.")
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Episode ID: {result.episode_id}")
        print(f"Inference IDs: {result.inference_ids}")


if __name__ == "__main__":
    asyncio.run(main())
