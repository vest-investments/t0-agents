"""Canonical Parallel pattern (Agent legacy API).

Multiple agents run on the same input concurrently via
``asyncio.gather``; a plain Python reducer combines their answers.
The pattern replaces the deleted ``parallel(*nodes, reducer=...)``
topology factory: ``asyncio.gather`` already does fan-out and
``return_exceptions=True`` already gives you continue-on-error
semantics, so no framework abstraction is needed.

The example here:

    optimist  ┐
    pessimist ┼─▶  asyncio.gather  ──▶  (ad-hoc reducer)  ──▶  merged text
    pragmatist┘

Each agent is toolless chat (chat-mode ``AgentConfig`` accepts
``tool_provider=None``).  The reducer joins the three takes with a
``\\n---\\n`` separator and gracefully drops any branches that raised.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway (shared with the glow examples):
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/12_concurrency/02_parallel.py``

Nesting: ``await fan_out(query)`` is just another awaitable.  Wrap it
in a ``@fastmcp.Tool.from_function`` to expose it as an MCP tool, or
embed it inside an orchestrator's ``sub_agents=[...]`` by promoting
the merged-output function to its own ``Agent``-as-tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, AgentResult


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        optimist = Agent(
            AgentConfig(name="optimist", function_name="agent"), t0=t0
        )
        pessimist = Agent(
            AgentConfig(name="pessimist", function_name="agent"), t0=t0
        )
        pragmatist = Agent(
            AgentConfig(name="pragmatist", function_name="agent"), t0=t0
        )

        query = "What does the rise of generative AI mean for software jobs?"

        # Fan-out: three independent inferences run concurrently.  Each
        # Agent.invoke() returns an awaitable that resolves to AgentResult;
        # asyncio.gather schedules them on the event loop in parallel.
        # return_exceptions=True keeps a failure on one branch from
        # cancelling the others — the reducer below decides what to do.
        results = await asyncio.gather(
            optimist.invoke(
                query, system="Reply with a one-sentence optimistic take."
            ),
            pessimist.invoke(
                query, system="Reply with a one-sentence pessimistic take."
            ),
            pragmatist.invoke(
                query, system="Reply with a one-sentence pragmatic take."
            ),
            return_exceptions=True,
        )

        # Ad-hoc reducer — anything you want; no abstraction forces a shape
        # here.  Common variants: pick one, majority vote, JSON-merge.  The
        # reducer here drops failed branches and joins the surviving texts
        # with a separator.
        labels = ("Optimist", "Pessimist", "Pragmatist")
        parts: list[str] = []
        for label, r in zip(labels, results, strict=True):
            if isinstance(r, BaseException):
                parts.append(f"- {label} (branch failed: {type(r).__name__})")
            else:
                assert isinstance(r, AgentResult)
                parts.append(f"- {label}: {r.response}")
        merged = "\n---\n".join(parts)

        print("=== Fan-out result ===")
        print(merged)


if __name__ == "__main__":
    asyncio.run(main())
