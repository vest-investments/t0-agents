"""System prompt example: demonstrates how to configure a TensorZero system
prompt that guides the LLM's tool-use behavior.

Prerequisites:
    1. Set GROQ_API_KEY in your environment.
    2. Copy the system prompt into the project's config directory::

           mkdir -p config/functions/agent/baseline
           cp examples/02_system_prompt/config/functions/agent/baseline/system.txt \
              config/functions/agent/baseline/system.txt

       Then add ``system_template`` to the variant in ``config/tensorzero.toml``::

           [functions.agent.variants.gpt_oss_20]
           type = "chat_completion"
           model = "gpt_oss_20"
           system_template = "functions/agent/baseline/system.txt"

       Restart the gateway: ``docker compose down && docker compose up -d``

    3. Run::

           uv run python examples/02_system_prompt/main.py

The system prompt in config/functions/agent/baseline/system.txt instructs the
model to prefer tool calls over clarification questions, keep reasoning brief,
and give up after repeated failures.  This addresses observed failure modes
O3 (ambiguity -> clarify instead of call) and O5 (tool selection skipped)
from the Phase 7 harness engineering notes.
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

from t0_agents import run_agent

# The directory name starts with a digit (02_...) so it can't be a normal
# Python import.  Add the example dir to sys.path so ``server`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import mcp  # type: ignore[import-not-found]

# Enable INFO logging to see the agent loop in action.
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    async with await gateway as t0, mcp_client:
        result = await run_agent(
            t0,
            mcp_client,
            "What's the weather in Paris, and convert that temperature to Fahrenheit?",
            "agent",
            max_turns=6,
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Response source: {result.response_source}")
        print(f"Episode ID: {result.episode_id}")


if __name__ == "__main__":
    asyncio.run(main())
