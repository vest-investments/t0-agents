"""JSON-function dispatch through the declarative ``Agent``.

Companion to ``main.py`` — same TensorZero function
(``extract_customer_info``) and same gateway config, but driven through
the high-level ``Agent`` wrapper instead of the low-level
``run_structured_inference`` helper.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Start the gateway with this example's config:
         docker compose -f examples/17_structured/docker-compose.yml \\
             --env-file .env up -d
    3. Run:
         uv run python examples/17_structured/structured_agent.py

The agent declares ``function_kind="json"`` and ``json_mode="strict"``
in its :class:`AgentConfig`.  ``Agent.run(...)`` then routes through
``run_structured_inference`` automatically and returns an
:class:`AgentResult` with ``function_kind == "json"``, ``parsed``
populated, and ``response`` set to the raw JSON string.

JSON functions do not call MCP tools, so the in-process FastMCP server
attached to the agent is intentionally empty — present only because
``Agent`` requires a ``ToolProvider``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def main() -> None:
    config = AgentConfig(
        name="customer-extractor",
        function_name="extract_customer_info",
        function_kind="json",
        json_mode="strict",
    )
    # Empty in-process MCP server — JSON functions don't dispatch tools,
    # but Agent still requires a ToolProvider at construction time.
    noop_server: FastMCP[Any] = FastMCP("noop")
    mcp_client: Client[Any] = Client(noop_server)

    messages = [
        "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
        "This is Diego Pérez (diego@example.com) — please follow up.",
    ]

    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        for idx, message in enumerate(messages, start=1):
            result = await agent.run(message)
            print(f"--- run {idx} ---")
            print(f"function_kind: {result.function_kind}")
            print(f"parsed:        {result.parsed}")
            print(f"response:      {result.response}")
            print(f"episode_id:    {result.episode_id}")
            print(f"inference_ids: {result.inference_ids}")
            print(f"turns:         {result.turns}")

    # Per-call schema/mode override (commented out: the demo TOML only
    # defines the ``strict`` variant, so flipping ``json_mode`` at
    # call time would error against this config — uncomment once you
    # add a ``tool``-mode variant).
    #
    # result = await agent.run(
    #     "...",
    #     json_mode="tool",
    # )


if __name__ == "__main__":
    asyncio.run(main())
