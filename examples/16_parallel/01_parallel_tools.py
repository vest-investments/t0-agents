"""Demo 01: Intra-turn parallel tool execution (L1 in docs/concurrency.md).

Runs the same agent loop twice — once with ``parallel_tools=False`` and
once with ``parallel_tools=True`` — and prints the wall-clock time of
each.  The LLM is replaced by a :class:`~t0_agents.testing.MockGateway`
scripted to emit **three** ``fetch_url`` ``ToolCall`` blocks in a single
assistant turn, so the loop's executor selection is exercised
deterministically regardless of which real model you would otherwise
use.

The MCP server's ``fetch_url`` tool sleeps for ``LATENCY_S = 1.5``
seconds per call.  Three sequential calls finish in ~4.5 s; three
parallel calls finish in ~1.5 s.

Runs offline — no docker compose, no API key.

Usage:
    uv run python examples/16_parallel/01_parallel_tools.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents import Agent, AgentConfig
from t0_agents.testing import MockGateway

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

URLS = ["https://example.com/a", "https://example.com/b", "https://example.com/c"]


def _build_gateway() -> MockGateway:
    """Two-turn script: emit three fetch_url calls, then a final text."""
    tool_calls = [
        ToolCall(
            id=f"call-{i}",
            raw_name="fetch_url",
            name="fetch_url",
            raw_arguments=f'{{"url": "{u}"}}',
            arguments={"url": u},
        )
        for i, u in enumerate(URLS)
    ]
    final = Text(text="Fetched all three URLs; each responded 200 OK.")
    return MockGateway.script([tool_calls, [final]])


async def _run_once(mcp_client: Any, *, parallel_tools: bool) -> float:
    gateway = _build_gateway()
    config = AgentConfig(
        name="parallel-demo",
        function_name="fetch_agent",
        max_turns=5,
        parallel_tools=parallel_tools,
    )
    agent = Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, gateway),
        tool_provider=mcp_client,
    )
    start = time.monotonic()
    result = await agent.run("Fetch a, b, c.")
    elapsed = time.monotonic() - start
    print(
        f"  parallel_tools={parallel_tools!s:<5} -> {elapsed:5.2f}s "
        f"({result.turns} turn(s))"
    )
    return elapsed


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mcp_client: Client[Any] = Client(mcp)

    print("=== Demo 01 (mocked LLM): intra-turn parallel tool execution ===")
    async with mcp_client:
        print("\nRun 1 (sequential, the default):")
        seq = await _run_once(mcp_client, parallel_tools=False)
        print("\nRun 2 (parallel):")
        par = await _run_once(mcp_client, parallel_tools=True)

    speedup = seq / par if par else float("inf")
    print(f"\nSpeed-up: {speedup:.2f}x (sequential / parallel)")
    print(
        "Each fetch_url sleeps for 1.5 s, so sequential ~= 4.5 s and "
        "parallel ~= 1.5 s.  The MockGateway emits three ToolCall blocks "
        "in one turn so the executor selection is the only thing changing "
        "between the two runs."
    )


if __name__ == "__main__":
    asyncio.run(main())
