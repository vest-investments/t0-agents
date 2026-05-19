"""Canonical Orchestrator pattern (glow API).

The glow counterpart of
``examples/agent/12_concurrency/03_orchestrator.py``.  Declare each
sub-agent as a normal ``@app.agent``; declare the orchestrator with
``Depends(<wrapper>)`` parameters for each sub-agent.  Glow records
the dependencies on the orchestrator's ``_AgentSpec.sub_agent_specs``
and v1.0 ``install_sub_agents`` (now in ``t0_agents.delegation``)
installs each as a ``<name>_ask`` MCP tool on the orchestrator's
shared, App-owned FastMCP server at lifespan time.

Topology:

    orchestrator
      ├──▶  researcher_ask (from Depends(researcher))
      └──▶  writer_ask     (from Depends(writer))

Anti-recursion guardrail — ``max_turns``
========================================

The orchestrator's LLM can in principle call the same sub-agent tool
repeatedly (an LLM-driven loop, not a code-driven loop).  Glow uses
v1.0's ``AgentConfig.max_turns`` default (20) unless overridden.  If
you need a tighter or looser bound, drop down to the v1.0 escape
hatch (``orchestrator._agent.config.max_turns = N``) before
``app.lifespan()`` builds the underlying ``Agent`` — or, in production
configurations, declare a dedicated TensorZero function with a
template that hard-codes the budget.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway:
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/glow/06_orchestrator.py``

Nesting: the same ``researcher`` / ``writer`` wrappers can be
``Depends(...)``-injected into a different orchestrator wrapper —
reuse is free because each ``_AgentSpec`` is just an object.  Two
orchestrators can also be composed in series (the Sequential
pattern), or wrapped as ``@app.tool`` callables for use as MCP tools
in a larger graph.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, Depends


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("researcher")
        async def researcher(query: str) -> str:  # type: ignore[empty-body]
            """Summarize the requested topic in a single sentence."""

        @app.agent("writer")
        async def writer(query: str) -> str:  # type: ignore[empty-body]
            """Format the provided summary as a single markdown bullet."""

        @app.agent("orchestrator", function_name="orchestrator")
        async def orchestrator(  # type: ignore[empty-body]
            query: str,
            research: Any = Depends(researcher),
            write: Any = Depends(writer),
        ) -> str:
            """Coordinate research then formatting via the available _ask tools.

            Call researcher_ask first with the topic, then writer_ask
            with the researcher's output, and return the writer's
            output verbatim.
            """

        async with app.lifespan():
            topic = (
                "transformer architectures for retrieval-augmented generation"
            )
            response = await orchestrator(topic)

        print("=== Orchestrator output ===")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
