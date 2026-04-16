"""Integration tests for the streaming routes — requires a live TensorZero gateway.

Boot the gateway with ``docker compose up -d`` and an LLM provider key
(``GROQ_API_KEY`` etc.) exported before running.

Marked ``@pytest.mark.integration`` module-wide so offline CI skips them
with ``pytest -m "not integration"``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client
from httpx import ASGITransport, AsyncClient
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.serve import create_app
from tests.server import mcp as demo_mcp

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_gateway() -> AsyncTensorZeroGateway:
    gateway = await cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    try:
        yield gateway
    finally:
        await gateway.close()


async def test_sse_endpoint_live_gateway(live_gateway: AsyncTensorZeroGateway) -> None:
    agent = Agent(
        AgentConfig(name="live-sse", function_name="agent", dryrun=True),
        t0=live_gateway,
        tool_provider=Client(demo_mcp),
    )
    app = create_app(
        agent,
        client_factory=lambda: Client(demo_mcp),
        streaming=True,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        async with http.stream(
            "GET", "/chat/stream", params={"message": "Say hello in one word."},
        ) as resp:
            assert resp.status_code == 200
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
    decoded = body.decode("utf-8")
    assert "event: done" in decoded


def test_ws_endpoint_live_gateway(live_gateway: AsyncTensorZeroGateway) -> None:
    def _client_factory() -> Client[Any]:
        return Client(demo_mcp)

    def _elicit_factory(
        handler: Callable[..., Any],
    ) -> Client[Any]:
        return Client(demo_mcp, elicitation_handler=handler)

    agent = Agent(
        AgentConfig(name="live-ws", function_name="agent", dryrun=True),
        t0=live_gateway,
        tool_provider=Client(demo_mcp),
    )
    app = create_app(
        agent,
        client_factory=_client_factory,
        elicitation_client_factory=_elicit_factory,
        streaming=True,
    )
    client = TestClient(app)
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json({"type": "user_message", "message": "Add 2 and 3."})
        final: dict[str, Any] | None = None
        while True:
            frame = ws.receive_json()
            if frame["type"] == "done":
                final = frame
                break
        assert final is not None
        assert final["result"]["episode_id"]
        # Sanity check: the full DoneEvent frame is JSON-serialisable
        # (asdict recursion handled AgentResult's nested dataclasses).
        assert json.dumps(final)
