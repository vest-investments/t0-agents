"""Demo 02: Multi-agent parallel composition (L3 in docs/concurrency.md).

Builds an orchestrator with two sub-agents (``researcher``, ``writer``)
and asks it to produce a short report.  The orchestrator's
``parallel_tools=True`` setting means that when the model delegates to
``researcher_ask`` and ``writer_ask`` in the same assistant turn, the
two sub-agents run concurrently via ``asyncio.gather`` — same flag as
demo 01, since sub-agents are exposed as MCP tools.

Each sub-agent's underlying tool sleeps for ~1 second, so when both
delegations happen in the same turn the total wall-clock cost
approaches the slower of the two rather than the sum.  Models that
delegate one sub-agent at a time still produce a correct report; the
demo just won't show overlap.

Prerequisites:
    1. ``GROQ_API_KEY`` in your environment (or in ``.env``).
    2. ``docker compose -f examples/16_parallel/docker-compose.yml up -d``.
    3. ``uv run python examples/16_parallel/02_parallel_subagents.py``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_server import mcp as research_server  # type: ignore[import-not-found]
from writing_server import mcp as writing_server  # type: ignore[import-not-found]

QUERY = (
    "Produce a short report on chain-of-thought prompting.  Use "
    "researcher_ask to find papers and writer_ask to draft an outline.  "
    "Call them together in the same turn so they run in parallel, then "
    "merge the results."
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    orchestrator_server = FastMCP("parallel-demo-orchestrator")

    async with (
        await gateway as t0,
        Client(research_server) as research_client,
        Client(writing_server) as writing_client,
        Client(orchestrator_server) as orchestrator_client,
    ):
        researcher = Agent(
            AgentConfig(
                name="researcher",
                function_name="research",
                max_turns=4,
                tags={"example": "16_parallel", "demo": "02", "role": "researcher"},
            ),
            t0=t0,
            tool_provider=cast(Any, research_client),
        )
        writer = Agent(
            AgentConfig(
                name="writer",
                function_name="writing",
                max_turns=4,
                tags={"example": "16_parallel", "demo": "02", "role": "writer"},
            ),
            t0=t0,
            tool_provider=cast(Any, writing_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="report_generator",
                function_name="orchestrator",
                max_turns=8,
                sub_agents=[researcher, writer],
                parallel_tools=True,
                tags={"example": "16_parallel", "demo": "02", "role": "orchestrator"},
            ),
            t0=t0,
            tool_provider=cast(Any, orchestrator_client),
        )

        print("=== Demo 02: multi-agent parallel composition ===\n")
        start = time.monotonic()
        result = await orchestrator.run(QUERY)
        elapsed = time.monotonic() - start

    print(f"\n--- Orchestrator response ---\n{result.response}")
    print(
        f"\n--- Run metadata ---\n"
        f"Episode ID    : {result.episode_id}\n"
        f"Turns         : {result.turns}\n"
        f"Inference IDs : {len(result.inference_ids)} total\n"
        f"Wall-clock    : {elapsed:.2f}s\n"
    )
    print(
        "Look at the INFO log above: when the orchestrator delegates "
        "to researcher_ask and writer_ask in the same turn, you should "
        "see their tool spans overlap rather than chain."
    )


if __name__ == "__main__":
    asyncio.run(main())
