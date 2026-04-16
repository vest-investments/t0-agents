"""Streaming HTTP agent demo — SSE + WebSocket + elicitation.

Launch with::

    docker compose up -d          # boots gateway + clickhouse at repo root
    uv run uvicorn examples.12_streaming_http.agent_app:app --port 8080

Open http://localhost:8080/static/index.html in a browser and chat.  The
page uses a WebSocket connection for full bidirectional traffic so tools
that call ``ctx.elicit()`` can pop up an approval modal.

Highlights
==========

* ``create_app(streaming=True, ...)`` mounts both ``GET /chat/stream``
  (SSE) and ``WS /chat/ws`` (bidirectional).
* ``client_factory`` mints fresh Clients for the non-streaming ``/chat``
  route and the SSE route.
* ``elicitation_client_factory`` mints Clients that register the
  per-connection elicitation handler (required by FastMCP Q3 — one
  handler per Client).
* ``deployment_label='streaming-demo'`` is echoed on every inference's
  tags so you can filter by surface in ClickHouse.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from fastapi.staticfiles import StaticFiles
from fastmcp import Client
from fastmcp.client.elicitation import ElicitationHandler
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.memory import InMemoryMemory
from t0_agents.serve import create_app

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

_t0 = cast(
    AsyncTensorZeroGateway,
    AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000", async_setup=False,
    ),
)


def _client_factory() -> Client[Any]:
    """Fresh Client per session for /chat + /chat/stream."""
    return Client(mcp)


def _elicitation_client_factory(handler: ElicitationHandler) -> Client[Any]:
    """Fresh Client wired with a per-connection elicitation handler."""
    return Client(mcp, elicitation_handler=handler)


_agent = Agent(
    AgentConfig(
        name="streaming-agent",
        function_name="agent",
        max_turns=10,
        memory=InMemoryMemory(),
    ),
    t0=_t0,
    tool_provider=Client(mcp),
)

app = create_app(
    _agent,
    client_factory=_client_factory,
    elicitation_client_factory=_elicitation_client_factory,
    deployment_label="streaming-demo",
    streaming=True,
)

# Serve the browser client from /static/index.html.
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
