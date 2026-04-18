"""HTTP agent demo — exposes an Agent over FastAPI.

Launch with::

    docker compose up -d          # from repo root (boots gateway + clickhouse)
    uv run uvicorn examples.11_http_agent.agent_app:app --port 8080

Then exercise the four endpoints with :file:`client.sh` or ``curl``.

Notes
=====

* The TensorZero gateway is built **synchronously** via
  ``AsyncTensorZeroGateway.build_http(..., async_setup=False)`` so the
  FastAPI ``app`` object is fully constructed at module-import time —
  that is what ``uvicorn agent_app:app`` expects.
* The per-session ``Client`` factory ``lambda: Client(mcp)`` creates a
  fresh in-process FastMCP client for every request.  That's what keeps
  elicitation handlers demultiplexed — ``fastmcp.Client`` binds a
  single ``elicitation_handler`` at construction, so a shared client
  cannot route concurrent prompts back to the right caller (see
  ``docs/serving.md`` § Concepts).  No elicitation tool is declared in
  this demo, so the forwarder's auto-decline default is never
  triggered.
* ``deployment_label="demo"`` is echoed on every inference's tags as
  ``t0_agents.deployment=demo`` — verify it via ``curl /info`` or
  ClickHouse.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import InMemoryMemory
from t0_agents.serve import create_app

# Make the sibling ``server`` module importable no matter which working
# directory uvicorn was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

# Synchronous handle to the gateway — `async_setup=False` tells the SDK
# to return the gateway directly instead of an Awaitable.  The cast
# narrows the Union return type for mypy strict mode.
_t0 = cast(
    AsyncTensorZeroGateway,
    AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000", async_setup=False,
    ),
)

_agent = Agent(
    AgentConfig(
        name="http-agent",
        function_name="agent",
        max_turns=10,
        # In-memory memory so ``client.sh``'s episode continuation demo
        # actually carries context across the two `/chat` calls.  Swap for
        # ``SqliteMemory("./demo.db")`` in production (behind the
        # ``[memory-sqlite]`` extra) — see ``docs/memory.md``.
        memory=InMemoryMemory(),
    ),
    t0=_t0,
    tool_provider=Client(mcp),
)

app = create_app(
    _agent,
    client_factory=lambda: Client(mcp),
    deployment_label="demo",
)
