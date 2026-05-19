"""Canonical Parallel pattern (glow API).

The glow counterpart of ``examples/agent/12_concurrency/02_parallel.py``:
multiple ``@app.agent``-decorated callables fanned out with
``asyncio.gather`` and combined by a plain Python reducer.  Because
``await wrapper(query)`` returns the unwrapped value directly,
gather over wrappers reads almost exactly like gather over plain
``async def`` functions — no ``.invoke().result()`` ceremony.

Topology:

    optimist  ┐
    pessimist ┼─▶  asyncio.gather  ──▶  (ad-hoc reducer)  ──▶  merged text
    pragmatist┘

The reducer joins the three takes with ``\\n---\\n`` and gracefully
drops any branches that raised (``return_exceptions=True``).

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway:
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/glow/05_parallel.py``

Nesting: ``await fan_out(query)`` is just another awaitable.  Wrap
the function with ``@app.tool`` to expose it as an MCP tool, or
embed it inside an orchestrator's ``Depends(...)`` parameters by
promoting the merged-output function to its own ``@app.agent``.
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

        @app.agent("optimist")
        async def optimist(query: str) -> str:  # type: ignore[empty-body]
            """Reply with a one-sentence optimistic take on the topic."""

        @app.agent("pessimist")
        async def pessimist(query: str) -> str:  # type: ignore[empty-body]
            """Reply with a one-sentence pessimistic take on the topic."""

        @app.agent("pragmatist")
        async def pragmatist(query: str) -> str:  # type: ignore[empty-body]
            """Reply with a one-sentence pragmatic take on the topic."""

        async with app.lifespan():
            query = (
                "What does the rise of generative AI mean for software jobs?"
            )

            # Fan-out: three independent inferences scheduled concurrently.
            # return_exceptions=True keeps a failure on one branch from
            # cancelling the others — the reducer decides what to do.
            results = await asyncio.gather(
                optimist(query),
                pessimist(query),
                pragmatist(query),
                return_exceptions=True,
            )

            # Ad-hoc reducer — anything you want; no abstraction forces a
            # shape here.  Common variants: concatenate, vote, JSON-merge,
            # pick first non-empty.  This one drops failed branches and
            # joins the survivors with a separator.
            labels = ("Optimist", "Pessimist", "Pragmatist")
            parts: list[str] = []
            for label, r in zip(labels, results, strict=True):
                if isinstance(r, BaseException):
                    parts.append(
                        f"- {label} (branch failed: {type(r).__name__})"
                    )
                else:
                    parts.append(f"- {label}: {r}")
            merged = "\n---\n".join(parts)

            print("=== Fan-out result ===")
            print(merged)


if __name__ == "__main__":
    asyncio.run(main())
