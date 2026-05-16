"""Agent factory for the TUI demo.

Invoked via:

.. code-block:: bash

    uv run t0-agents run examples.10_cli.agent:assistant
    uv run t0-agents serve examples.10_cli.agent:assistant
    uv run t0-agents info examples.10_cli.agent:assistant

The factory is decorated with :func:`contextlib.asynccontextmanager` so
the CLI can open both the TensorZero gateway and the FastMCP ``Client``
as a single ``async with`` block around the REPL lifetime.  This is the
factory-only loader convention for :func:`t0_agents.cli.loader.load_agent_factory`.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.cli import rich_elicitation_handler
from t0_agents.memory import InMemoryMemory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]  # noqa: E402


@asynccontextmanager
async def assistant() -> AsyncIterator[Agent]:
    """Build and yield a fresh :class:`Agent` with streaming + elicitation.

    The Client is wired with :func:`t0_agents.cli.rich_elicitation_handler`
    so a server-side ``ctx.elicit`` call surfaces as a Rich prompt in the
    same terminal the panel is running in.
    """
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(mcp, elicitation_handler=rich_elicitation_handler)
    async with await gateway as t0, client:
        yield Agent(
            AgentConfig(
                name="tui-assistant",
                function_name="agent",
                max_turns=10,
                memory=InMemoryMemory(),
                tags={"example": "10_cli"},
            ),
            t0=t0,
            tool_provider=client,
        )
