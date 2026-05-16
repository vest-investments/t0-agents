"""Demo 02: Multi-agent parallel composition (L3 in docs/concurrency.md).

Builds an orchestrator with two sub-agents (``researcher``, ``writer``)
and runs it twice — once with ``parallel_tools=False`` and once with
``True``.  The orchestrator's LLM is replaced by a
:class:`~t0_agents.testing.MockGateway` scripted to delegate to both
sub-agents **in the same assistant turn**, so the orchestrator's
``parallel_tools`` flag is the only thing that decides whether the two
delegations overlap.  Each sub-agent's LLM is mocked too, so the demo
runs offline and deterministically.

Each sub-agent's tool sleeps for 1 s.  Sequential dispatch sums the two
sleeps (~2 s); parallel dispatch overlaps them (~1 s).

Runs offline — no docker compose, no API key.

Usage:
    uv run python examples/12_concurrency/02_parallel_subagents.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents import Agent, AgentConfig
from t0_agents.testing import MockGateway

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_server import mcp as research_server  # type: ignore[import-not-found]
from writing_server import mcp as writing_server  # type: ignore[import-not-found]


def _orchestrator_gateway() -> MockGateway:
    """Two-turn script: delegate to researcher_ask + writer_ask, then text."""
    return MockGateway.script(
        [
            [
                ToolCall(
                    id="call-research",
                    raw_name="researcher_ask",
                    name="researcher_ask",
                    raw_arguments='{"message": "Find papers on chain-of-thought"}',
                    arguments={"message": "Find papers on chain-of-thought"},
                ),
                ToolCall(
                    id="call-write",
                    raw_name="writer_ask",
                    name="writer_ask",
                    raw_arguments='{"message": "Draft an outline for CoT report"}',
                    arguments={"message": "Draft an outline for CoT report"},
                ),
            ],
            [Text(text="Report assembled from researcher and writer outputs.")],
        ]
    )


def _research_gateway() -> MockGateway:
    """Two-turn script: call search_papers, then return a summary."""
    return MockGateway.script(
        [
            [
                ToolCall(
                    id="call-search",
                    raw_name="search_papers",
                    name="search_papers",
                    raw_arguments='{"query": "chain-of-thought"}',
                    arguments={"query": "chain-of-thought"},
                ),
            ],
            [Text(text="Found three relevant CoT papers.")],
        ]
    )


def _writing_gateway() -> MockGateway:
    """Two-turn script: call format_outline, then return the outline."""
    return MockGateway.script(
        [
            [
                ToolCall(
                    id="call-format",
                    raw_name="format_outline",
                    name="format_outline",
                    raw_arguments='{"topic": "chain-of-thought"}',
                    arguments={"topic": "chain-of-thought"},
                ),
            ],
            [Text(text="Markdown outline drafted.")],
        ]
    )


async def _run_once(*, parallel_tools: bool) -> float:
    orchestrator_server = FastMCP("parallel-demo-orchestrator")
    async with (
        Client(research_server) as research_client,
        Client(writing_server) as writing_client,
        Client(orchestrator_server) as orchestrator_client,
    ):
        researcher = Agent(
            AgentConfig(name="researcher", function_name="research", max_turns=4),
            t0=cast(AsyncTensorZeroGateway, _research_gateway()),
            tool_provider=cast(Any, research_client),
        )
        writer = Agent(
            AgentConfig(name="writer", function_name="writing", max_turns=4),
            t0=cast(AsyncTensorZeroGateway, _writing_gateway()),
            tool_provider=cast(Any, writing_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="report_generator",
                function_name="orchestrator",
                max_turns=8,
                sub_agents=[researcher, writer],
                parallel_tools=parallel_tools,
            ),
            t0=cast(AsyncTensorZeroGateway, _orchestrator_gateway()),
            tool_provider=cast(Any, orchestrator_client),
        )
        start = time.monotonic()
        result = await orchestrator.run("Produce a short report on CoT.")
        elapsed = time.monotonic() - start
    print(
        f"  parallel_tools={parallel_tools!s:<5} -> {elapsed:5.2f}s "
        f"({result.turns} turn(s))"
    )
    return elapsed


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("=== Demo 02 (mocked LLM): multi-agent parallel composition ===")
    print("\nRun 1 (sequential, the default):")
    seq = await _run_once(parallel_tools=False)
    print("\nRun 2 (parallel):")
    par = await _run_once(parallel_tools=True)
    speedup = seq / par if par else float("inf")
    print(f"\nSpeed-up: {speedup:.2f}x (sequential / parallel)")
    print(
        "Each sub-agent's tool sleeps for 1 s.  Under parallel dispatch "
        "the orchestrator's researcher_ask and writer_ask delegations run "
        "concurrently, so the wall-clock approaches the slower of the "
        "two (~1 s) rather than the sum (~2 s)."
    )


if __name__ == "__main__":
    asyncio.run(main())
