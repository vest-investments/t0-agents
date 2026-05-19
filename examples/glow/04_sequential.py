"""Canonical Sequential pattern (glow API).

The glow counterpart of
``examples/12_concurrency/01_sequential.py``: two ``@app.agent``-decorated
callables chained as a pipeline.  The output of ``summarizer`` becomes the
input of ``translator``; an ad-hoc truncate step sits in between.

Glow's decorator surface trims the construction:

- ``@app.agent("name")`` registers the spec; the docstring becomes the
  system prompt automatically (no per-invocation ``system=``).
- ``-> str`` return annotation infers ``ChatOutput``.
- ``await summarizer(query)`` returns the unwrapped ``str`` (the
  Pythonic fast path); use ``.invoke(query).result()`` to get the
  full ``AgentResult``.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway:
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/glow/04_sequential.py``

Nesting: ``await summarize_then_translate(query)`` is just another
``async def`` — call it from another async function, wrap it with
``@app.tool`` to expose it as an MCP tool, or embed inside an
orchestrator's ``Depends(...)`` parameters by promoting the chain to
its own ``@app.agent``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("summarizer")
        async def summarizer(query: str) -> str:  # type: ignore[empty-body]
            """Summarize the input in one short paragraph (3 sentences max)."""

        @app.agent("translator")
        async def translator(text: str) -> str:  # type: ignore[empty-body]
            """Translate the input to Spanish, preserving its meaning."""

        async with app.lifespan():
            query = (
                "The James Webb Space Telescope is a large infrared "
                "observatory launched in December 2021.  It orbits the Sun "
                "at the second Lagrange point and has produced detailed "
                "images of distant galaxies, exoplanet atmospheres, and "
                "star-forming nebulae since its commissioning."
            )

            print("=== Step 1: summarize ===")
            summary = await summarizer(query)
            print(summary)

            # Ad-hoc step — anything you want; no abstraction forces a shape
            # here.  Truncate, normalize, branch on a condition, emit a log,
            # short-circuit; the gap is plain Python.
            if len(summary) > 500:
                summary = summary[:500] + "..."
                print("\n[ad-hoc] summary truncated to 500 chars")

            print("\n=== Step 2: translate ===")
            translation = await translator(summary)
            print(translation)


if __name__ == "__main__":
    asyncio.run(main())
