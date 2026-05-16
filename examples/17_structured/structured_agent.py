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

The agent declares ``output=JsonOutput(json_mode="strict")`` in its
:class:`AgentConfig` (v1.0 canonical surface — replaces the v0.12
``function_kind="json", json_mode="strict"`` pair).
``agent.invoke(...)`` then routes through the single-shot terminal and
returns an :class:`AgentResult` with ``parsed`` populated (and the
v0.12 read-side properties ``function_kind``, ``response``, etc. still
available).

Single-shot agents do not dispatch MCP tools, so ``Agent`` accepts no
``tool_provider`` here — the constructor takes just the configuration
and the TensorZero gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, JsonOutput


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def main() -> None:
    config = AgentConfig(
        name="customer-extractor",
        function_name="extract_customer_info",
        output=JsonOutput(json_mode="strict"),
    )

    messages = [
        "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
        "This is Diego Pérez (diego@example.com) — please follow up.",
    ]

    async with await _gateway_factory() as t0:
        agent = Agent(config, t0=t0)
        for idx, message in enumerate(messages, start=1):
            result = await agent.invoke(message).result()
            print(f"--- run {idx} ---")
            print(f"function_kind: {result.function_kind}")
            print(f"parsed:        {result.parsed}")
            print(f"response:      {result.response}")
            print(f"episode_id:    {result.episode_id}")
            print(f"inference_ids: {result.inference_ids}")
            print(f"turns:         {result.turns}")

    # Per-call output= override is also supported.  Uncomment to flip
    # to a different JSON mode for a single invocation (the demo TOML
    # only defines the ``strict`` variant, so a ``tool`` override
    # against this config would surface a gateway error):
    #
    # result = await agent.invoke(
    #     "...",
    #     output=JsonOutput(json_mode="tool"),
    # ).result()


if __name__ == "__main__":
    asyncio.run(main())
