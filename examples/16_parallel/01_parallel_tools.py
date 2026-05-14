"""Demo 01: Intra-turn parallel tool execution (L1 in docs/concurrency.md).

Runs the same fetch-heavy query twice against the same FastMCP server:
once with ``parallel_tools=False`` (the default), once with
``parallel_tools=True``.  Prints the wall-clock time of each run so the
speed-up is visible when the model decides to call ``fetch_url``
multiple times in a single turn.

The MCP server's ``fetch_url`` tool sleeps for ``LATENCY_S`` seconds per
call.  With three concurrent calls in one turn the parallel run should
finish in ~``LATENCY_S`` seconds and the sequential run in ~``3 *
LATENCY_S`` seconds.  Smaller models that emit one tool call per turn
will not show a difference here — see the README for the L2 "batch
tool" mitigation.

Prerequisites:
    1. ``GROQ_API_KEY`` in your environment (or in ``.env``).
    2. ``docker compose -f examples/16_parallel/docker-compose.yml up -d``.
    3. ``uv run python examples/16_parallel/01_parallel_tools.py``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

QUERY = (
    "Fetch metadata for these three URLs and tell me how long each "
    "response was: https://example.com/a, https://example.com/b, "
    "https://example.com/c.  Issue the fetches together."
)


async def _run_once(t0: Any, mcp_client: Any, *, parallel_tools: bool) -> float:
    config = AgentConfig(
        name="parallel-demo",
        function_name="fetch_agent",
        max_turns=5,
        parallel_tools=parallel_tools,
        tags={"example": "16_parallel", "demo": "01", "parallel": str(parallel_tools)},
    )
    agent = Agent(config, t0=t0, tool_provider=mcp_client)
    start = time.monotonic()
    result = await agent.run(QUERY)
    elapsed = time.monotonic() - start
    print(
        f"  parallel_tools={parallel_tools!s:<5} -> {elapsed:5.2f}s "
        f"({result.turns} turn(s), episode {result.episode_id})"
    )
    return elapsed


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    mcp_client: Client[Any] = Client(mcp)

    print("=== Demo 01: intra-turn parallel tool execution ===")
    async with await gateway as t0, mcp_client:
        print("\nRun 1 (sequential, the default):")
        seq = await _run_once(t0, mcp_client, parallel_tools=False)
        print("\nRun 2 (parallel):")
        par = await _run_once(t0, mcp_client, parallel_tools=True)

    speedup = seq / par if par else float("inf")
    print(f"\nSpeed-up: {speedup:.2f}x (sequential / parallel)")
    print(
        "Tip: a sub-1x ratio means the model emitted one tool call per "
        "turn; see the README and L2 in docs/concurrency.md for the "
        "batch-tool pattern."
    )


if __name__ == "__main__":
    asyncio.run(main())
