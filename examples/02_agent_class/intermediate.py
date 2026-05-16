"""Agent class with a baked-in system prompt.

Demonstrates the canonical TensorZero pattern: the system prompt lives
in ``tensorzero.toml`` (as a Minijinja template attached to the
function variant), and the Python caller just points
``AgentConfig.function_name`` at it.  The persona is preserved across
every ``agent.invoke(...)`` call without the caller ever having to
thread the prompt through — which is exactly what
``examples/02_agent_class/`` promises in its README.

Compare with ``advanced.py`` (same example folder) which takes the
opposite approach: bypass the template and supply a raw ``system="..."``
string per call.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/02_agent_class/docker-compose.yml \\
           --env-file .env up -d
    3. Run: uv run python examples/02_agent_class/intermediate.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider

# The directory name starts with a digit (02_...) so it can't be a normal
# Python import.  Add the example dir to sys.path so ``server`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]


async def main() -> None:
    # The persona is configured on the TensorZero function — see
    # ``config/functions/agent_with_persona/system_template.minijinja``.
    # Python never sees the prompt string; it just points at the function.
    config = AgentConfig(
        name="weather-concierge",
        function_name="agent_with_persona",
        max_turns=4,
        tags={"example": "02_agent_class", "tier": "intermediate"},
    )

    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    async with await gateway as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )

        # Two back-to-back invocations.  Notice the same warm,
        # conversational tone in both answers — that's the system
        # template doing its job on every turn without any per-call
        # work from the caller.
        for question in (
            "What is the weather in Tokyo?",
            "And in Paris?",
        ):
            result = await agent.invoke(question)
            print(f"\nQ: {question}")
            print(f"A: {result.response}")


if __name__ == "__main__":
    asyncio.run(main())
