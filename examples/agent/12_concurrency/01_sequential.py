"""Canonical Sequential pattern (Agent legacy API).

Two agents chained as a pipeline: the output of the first is the input
of the second.  Plain ``async/await`` does the wiring; no framework
abstraction stands between the two steps, so you can drop any ad-hoc
code (a transform, a validation, a log, an early return) in the gap
and it composes naturally.

The pipeline here:

    summarizer  ──▶  (ad-hoc truncate)  ──▶  translator

Both agents are toolless chat: chat-mode ``AgentConfig`` accepts
``tool_provider=None``, so you can pipeline two LLM calls without
standing up a FastMCP server.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway (shared with the glow examples):
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/12_concurrency/01_sequential.py``

Nesting: just call ``await summarize_then_translate(query)`` from
another ``async def``, expose it as a FastMCP tool via
``fastmcp.Tool.from_function``, or embed it inside an orchestrator
that lists this whole pipeline as a sub_agent.  No abstraction is
forcing a shape; the pattern composes by being plain Python.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        summarizer = Agent(
            AgentConfig(name="summarizer", function_name="agent"),
            t0=t0,
        )
        translator = Agent(
            AgentConfig(name="translator", function_name="agent"),
            t0=t0,
        )

        query = (
            "The James Webb Space Telescope is a large infrared observatory "
            "launched in December 2021.  It orbits the Sun at the second "
            "Lagrange point and has produced detailed images of distant "
            "galaxies, exoplanet atmospheres, and star-forming nebulae since "
            "its commissioning."
        )

        print("=== Step 1: summarize ===")
        summary_result = await summarizer.invoke(
            query,
            system="Summarize the input in one short paragraph (3 sentences max).",
        )
        summary = summary_result.response
        print(summary)

        # Ad-hoc step — anything you want; no abstraction forces a shape here.
        # Here we cap the summary at 500 chars before handing it on; in a
        # real pipeline this is where you would normalise, validate, branch
        # on a condition, emit a log, or short-circuit.
        if len(summary) > 500:
            summary = summary[:500] + "..."
            print("\n[ad-hoc] summary truncated to 500 chars")

        print("\n=== Step 2: translate ===")
        translation_result = await translator.invoke(
            summary,
            system="Translate the input to Spanish, preserving its meaning.",
        )
        print(translation_result.response)


if __name__ == "__main__":
    asyncio.run(main())
