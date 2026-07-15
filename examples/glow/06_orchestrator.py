"""Canonical Orchestrator pattern (glow API).

The glow counterpart of
``examples/agent/12_concurrency/03_orchestrator.py``.  Declare each
sub-agent as a normal ``@app.agent``; declare the orchestrator with
``Depends(<wrapper>)`` parameters for each sub-agent.  Glow records
the dependencies on the orchestrator's ``_AgentSpec.sub_agent_specs``;
``Agent.__init__`` auto-allocates a
:class:`~t0_agents.delegation.SubAgentToolProvider` that exposes each
sub-agent as a ``<name>_ask`` MCP tool to the orchestrator LLM.  No
FastMCP server is needed when the App has no ``@app.tool`` or
:class:`MCPServer` specs (this example).

Topology:

    orchestrator
      ├──▶  researcher_ask (from Depends(researcher))
      └──▶  writer_ask     (from Depends(writer))

Anti-recursion guardrail — ``max_turns``
========================================

The orchestrator's LLM can in principle call the same sub-agent tool
repeatedly (an LLM-driven loop, not a code-driven loop).  Glow uses
v1.0's ``AgentConfig.max_turns`` default (20) unless overridden.  Pass
``max_turns=`` directly to ``@app.agent(...)`` for a tighter or looser
bound (this example uses ``max_turns=8`` below) — or, in production
configurations, declare a dedicated TensorZero function with a template
that hard-codes the budget.

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

from t0_agents.glow import AgentHandle, App, Depends, llm


async def main() -> None:
    app = App(gateway="http://localhost:3000")

    @app.agent("researcher")
    async def researcher(query: str) -> str:
        """Summarize the requested topic in a single sentence."""
        return llm()

    @app.agent("writer")
    async def writer(query: str) -> str:
        """Format the provided summary as a single markdown bullet."""
        return llm()

    @app.agent("orchestrator", function_name="orchestrator", max_turns=8)
    async def orchestrator(
        query: str,
        research: AgentHandle[..., str] = Depends(researcher),
        write: AgentHandle[..., str] = Depends(writer),
    ) -> str:
        """Coordinate research then formatting via the available _ask tools.

        Call researcher_ask first with the topic, then writer_ask
        with the researcher's output, and return the writer's
        output verbatim.
        """
        return llm()

    async with app:
        topic = (
            "transformer architectures for retrieval-augmented generation"
        )
        response = await orchestrator(topic)

    print("=== Orchestrator output ===")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
