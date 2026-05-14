"""Real-provider demo of the M3 permissive parsing + retry layers.

Companion to ``recovery_demo.py`` — same ``Agent.run`` invocation, but
driven against a live TensorZero gateway and the real model behind it.
The prompt is deliberately under-specified and ``json_mode`` is set to
``"off"`` per call, so the model is free to wrap its JSON in prose or
a fenced code block.  When that happens, the agent's permissive layer
recovers the JSON in-Python; when the model emits something the
permissive layer cannot recover, the retry layer re-issues the
inference with a corrective hint.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Start the gateway with this example's config:
         docker compose -f examples/17_structured/docker-compose.yml \\
             --env-file .env up -d
    3. Run:
         uv run python examples/17_structured/recovery_real_provider.py

Model behaviour is non-deterministic — the same prompt may exit on
``strict``, ``permissive``, or ``retry`` on different runs.  Re-run a
few times to see the recovery layers exercised.  ``"none"`` is also
possible (and reported) when every attempt failed; this is expected
on weak models and a signal that you should either tighten the prompt
or raise ``max_json_retries``.
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
        name="customer-extractor-loose",
        function_name="extract_customer_info",
        function_kind="json",
        json_mode="strict",  # config baseline; overridden per call below
        permissive_json_parsing=True,
        max_json_retries=2,
    )
    noop_server: FastMCP[Any] = FastMCP("noop")
    mcp_client: Client[Any] = Client(noop_server)

    prompt = (
        "Tell me about Sarah Johnson (sarah.j@example.com). "
        "Feel free to wrap your reply in a brief explanation."
    )

    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        # ``json_mode="off"`` lets the model decide on output formatting,
        # which is exactly what M3's recovery layers are built to absorb.
        result = await agent.run(prompt, json_mode="off")

    print(f"parse_strategy:      {result.parse_strategy}")
    print(f"json_parse_attempts: {result.json_parse_attempts}")
    print(f"turns:               {result.turns}")
    print(f"inference_ids:       {result.inference_ids}")
    print(f"episode_id:          {result.episode_id}")
    print(f"parsed:              {result.parsed}")
    print(f"final raw:           {result.response!r}")


if __name__ == "__main__":
    asyncio.run(main())
