"""Agent factory for the multi-agent composition demo.

Invoked via:

.. code-block:: bash

    uv run t0-agents run examples.08_multi_agent.agents:orchestrator_factory

The factory yields an orchestrator :class:`Agent` that has been wired
with two sub-agents (``researcher``, ``writer``).  Each sub-agent has
its own FastMCP server + ``fastmcp.Client``; the ``install_sub_agents``
hook registers a ``researcher_ask`` and ``writer_ask`` tool on the
orchestrator's own in-process FastMCP server at construction time.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

# Path juggling so the example modules can be imported as top-level names
# the same way ``examples/10_cli/agent.py`` does.  Keeps the example
# runnable both as a Python script and via the ``t0-agents run`` loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_server import mcp as arxiv_server  # type: ignore[import-not-found]
from markdown_server import mcp as markdown_server  # type: ignore[import-not-found]


def _build_orchestrator_server() -> FastMCP:
    """Return a bare FastMCP for the orchestrator.

    The server starts empty; ``install_sub_agents`` adds the
    ``researcher_ask`` and ``writer_ask`` tools when the orchestrator
    :class:`Agent` is constructed.  Leaving it empty here makes the
    sub-agent registration the sole source of orchestrator-level
    tools, which keeps the collision-detection code path exercised.
    """
    return FastMCP("orchestrator-server")


@asynccontextmanager
async def orchestrator_factory() -> AsyncIterator[Agent]:
    """Yield a composed orchestrator Agent, then tear down the stack."""
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    async with (
        await gateway as t0,
        Client(arxiv_server) as arxiv_client,
        Client(markdown_server) as markdown_client,
        Client(_build_orchestrator_server()) as orchestrator_client,
    ):
        researcher = Agent(
            AgentConfig(
                name="researcher",
                function_name="research",
                max_turns=6,
                tags={"example": "08_multi_agent", "role": "researcher"},
            ),
            t0=t0,
            tool_provider=cast(Any, arxiv_client),
        )
        writer = Agent(
            AgentConfig(
                name="writer",
                function_name="writing",
                max_turns=6,
                tags={"example": "08_multi_agent", "role": "writer"},
            ),
            t0=t0,
            tool_provider=cast(Any, markdown_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="report_generator",
                function_name="orchestrator",
                max_turns=10,
                sub_agents=[researcher, writer],
                tags={"example": "08_multi_agent", "role": "orchestrator"},
            ),
            t0=t0,
            tool_provider=cast(Any, orchestrator_client),
        )
        yield orchestrator
