"""Canonical Orchestrator pattern (Agent legacy API).

An orchestrator delegates work to sub-agents declared via
``AgentConfig(sub_agents=[...])``.  Each sub-agent is auto-installed
as a ``<name>_ask`` MCP tool on the orchestrator's tool provider (the
v1.0 :class:`SubAgentMiddleware` wires this at Agent construction
time via :func:`t0_agents.delegation.install_sub_agents`).  The
orchestrator's LLM picks which sub-agent to call and in what order —
that's the "LLM in the routing decision" model.  Use it when the
right delegate depends on the input; for fixed topologies, use the
Sequential or Parallel patterns instead.

Topology:

    orchestrator
      ├──▶  researcher_ask (sub-agent)
      └──▶  writer_ask     (sub-agent)

For demonstration clarity each sub-agent runs toolless chat — the
``sub_agents=`` wiring works the same whether or not the leaves have
their own MCP tool providers.  Production-grade configurations
typically declare per-agent TensorZero functions with system-template
files so each leaf has a stable identity; here the orchestrator's
system prompt is rich enough to direct dispatch on its own.

Anti-recursion guardrail — ``max_turns``
========================================

The orchestrator's LLM can in principle call the same sub-agent tool
repeatedly (an LLM-driven loop, not a code-driven loop).  Bound it
with ``max_turns=N`` on the parent ``AgentConfig``.  Sub-agents are
likewise bounded by their own ``max_turns``.  Without this, a
malformed prompt or a stubborn delegation policy can run away.  Pick
a budget that exceeds the longest sensible plan by a comfortable
margin and let it fail loud if it overruns.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway (shared with the glow examples):
       ``docker compose -f examples/glow/docker-compose.yml --env-file .env up -d``
    3. Run: ``uv run python examples/12_concurrency/03_orchestrator.py``

Nesting: the same researcher and writer Agents can be embedded in a
different orchestrator (``AgentConfig(sub_agents=[researcher, writer])``
again) — reuse is free because each Agent is just an object.  Two
orchestrators can also be composed in series with a plain Sequential
chain, or in a router that picks one of them via a top-level LLM
decision (a nested orchestrator where each "leaf" sub-agent is itself
an orchestrator).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

_ORCHESTRATOR_SYSTEM = (
    "Coordinate a research-then-format pipeline.  First call "
    "researcher_ask with a request like 'Summarize the topic in one "
    "sentence: <topic>'.  Then call writer_ask with a request like "
    "'Format the following sentence as a single markdown bullet: "
    "<researcher output>'.  Reply with the writer's output verbatim — "
    "do not add commentary."
)


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    # SubAgentMiddleware installs each sub-agent as a `<name>_ask` MCP tool
    # on the orchestrator's tool_provider.  Empty FastMCP servers are enough —
    # they expose no tools of their own; only the sub-agent ones.
    orchestrator_server: FastMCP[Any] = FastMCP("orchestrator-tools")
    researcher_server: FastMCP[Any] = FastMCP("researcher-noop")
    writer_server: FastMCP[Any] = FastMCP("writer-noop")

    async with (
        await gateway as t0,
        Client(researcher_server) as researcher_client,
        Client(writer_server) as writer_client,
        Client(orchestrator_server) as orchestrator_client,
    ):
        researcher = Agent(
            AgentConfig(
                name="researcher", function_name="agent", max_turns=4
            ),
            t0=t0,
            tool_provider=cast(Any, researcher_client),
        )
        writer = Agent(
            AgentConfig(
                name="writer", function_name="agent", max_turns=4
            ),
            t0=t0,
            tool_provider=cast(Any, writer_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="orchestrator",
                function_name="orchestrator",
                # max_turns is the anti-recursion guardrail — see the
                # module docstring.  8 turns gives the LLM enough room to
                # call researcher_ask, read the result, call writer_ask,
                # read the result, and emit the final response, with slack
                # for retries.
                max_turns=8,
                sub_agents=[researcher, writer],
            ),
            t0=t0,
            tool_provider=cast(Any, orchestrator_client),
        )

        topic = "transformer architectures for retrieval-augmented generation"
        result = await orchestrator.invoke(topic, system=_ORCHESTRATOR_SYSTEM)

    print("=== Orchestrator output ===")
    print(result.response)


if __name__ == "__main__":
    asyncio.run(main())
