"""Raw system prompt: PaaS-style ``system=`` kwarg on ``Agent.invoke``.

Demonstrates the escape hatch added in 0.13.0: instead of putting the
system prompt in ``tensorzero.toml`` (the recommended pattern shown by
``examples/02_agent_class/``), the caller supplies a raw string per
run.  Useful for prompt-as-a-service style products where the prompt
body itself is request data.

The same agent runs three times with three different runtime-supplied
personalities — same model, same TensorZero function, same MCP
provider, only the ``system="..."`` argument changes.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway::

           docker compose -f examples/02_agent_class/docker-compose.yml \\
               --env-file .env up -d

    3. Run::

           uv run python examples/02_agent_class/advanced.py

    4. Tear down when done::

           docker compose -f examples/02_agent_class/docker-compose.yml down
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

# The directory name starts with a digit (18_...) so it can't be a normal
# Python import.  Add the example dir to sys.path so ``server`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

# Personalities supplied at runtime — the whole point of the demo is
# that these strings are *data*, not configuration.  In a real PaaS
# product they would come from a database or the request body.
PERSONALITIES: dict[str, str] = {
    "haiku": "Reply only with a single 3-line haiku.  No prose, no preamble.",
    "pirate": (
        "You are a pirate captain.  Reply in exaggerated pirate slang, "
        "two sentences max."
    ),
    "lawyer": (
        "You are a precise contracts lawyer.  Answer in at most two "
        "sentences using plain language."
    ),
}

QUESTIONS: list[tuple[str, str]] = [
    ("haiku", "the ocean at dawn"),
    ("pirate", "where is the treasure?"),
    ("lawyer", "what is a force majeure clause?"),
]


async def main() -> None:
    config = AgentConfig(
        name="paas-demo",
        function_name="agent",
        max_turns=3,
        tags={"example": "18_raw_system_prompt"},
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
        for personality, question in QUESTIONS:
            result = await agent.invoke(
                question,
                system=PERSONALITIES[personality],
            )
            print(f"\n[{personality}] {question}")
            print(f"  → {result.response}")


if __name__ == "__main__":
    asyncio.run(main())
