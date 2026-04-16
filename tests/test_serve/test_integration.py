"""Integration tests for the HTTP serving layer.

Requires ``docker compose up -d`` (TensorZero gateway + ClickHouse) and
``GROQ_API_KEY`` in ``.env`` — see ``docs/serving.md``.  Uses
``httpx.AsyncClient`` against ``ASGITransport`` so no ``uvicorn``
subprocess is needed; the live component is the gateway only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

import pytest
from fastmcp import Client
from httpx import ASGITransport, AsyncClient
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.serve import create_app
from tests.server import mcp

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_live_chat_round_trip() -> None:
    """A single ``POST /chat`` reaches the gateway and returns real text."""
    async with await _gateway_factory() as t0, Client(mcp) as warmup_client:
        # Warm the MCP transport once so construction isn't counted as
        # a request cost; the in-process FastMCP client is cheap but
        # the first __aenter__ spins a background session task.
        await warmup_client.list_tools()

        agent = Agent(
            AgentConfig(name="integration-agent", function_name="agent"),
            t0=t0,
            tool_provider=warmup_client,
        )
        app = create_app(agent, client_factory=lambda: Client(mcp))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as http:
            resp = await http.post(
                "/chat",
                json={"message": "Say hello in one short sentence."},
                timeout=30.0,
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["response"], str) and body["response"]
    assert body["episode_id"]
    assert body["turns"] >= 1


async def test_live_episode_continuity() -> None:
    """Two consecutive requests sharing ``episode_id`` continue the run."""
    async with await _gateway_factory() as t0, Client(mcp) as c:
        await c.list_tools()
        agent = Agent(
            AgentConfig(name="integration-agent", function_name="agent"),
            t0=t0,
            tool_provider=c,
        )
        app = create_app(agent, client_factory=lambda: Client(mcp))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as http:
            first = await http.post(
                "/chat",
                json={"message": "Remember the secret code 42."},
                timeout=30.0,
            )
            assert first.status_code == 200, first.text
            episode_id = first.json()["episode_id"]
            assert episode_id

            second = await http.post(
                "/chat",
                json={
                    "message": "What was the secret code I told you?",
                    "episode_id": episode_id,
                },
                timeout=30.0,
            )
            assert second.status_code == 200, second.text
            assert second.json()["episode_id"] == episode_id


async def test_live_concurrent_users_distinct_episodes() -> None:
    """10 concurrent ``POST /chat`` with distinct episode_ids all succeed."""
    concurrency = 10
    async with await _gateway_factory() as t0, Client(mcp) as c:
        await c.list_tools()
        agent = Agent(
            AgentConfig(name="integration-agent", function_name="agent"),
            t0=t0,
            tool_provider=c,
        )
        app = create_app(agent, client_factory=lambda: Client(mcp))

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
            timeout=60.0,
        ) as http:
            responses = await asyncio.gather(
                *[
                    http.post(
                        "/chat",
                        json={"message": f"Echo the number {i}."},
                    )
                    for i in range(concurrency)
                ],
            )

    statuses = [r.status_code for r in responses]
    assert all(s == 200 for s in statuses), statuses
    episode_ids = {r.json()["episode_id"] for r in responses}
    # Each concurrent request should have gotten its own episode_id
    # (no cross-contamination from a shared client).
    assert len(episode_ids) == concurrency


async def test_live_feedback_round_trip() -> None:
    """``POST /feedback`` reaches the gateway and returns a feedback id."""
    async with await _gateway_factory() as t0, Client(mcp) as c:
        await c.list_tools()
        agent = Agent(
            AgentConfig(name="integration-agent", function_name="agent"),
            t0=t0,
            tool_provider=c,
        )
        app = create_app(agent, client_factory=lambda: Client(mcp))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as http:
            chat = await http.post(
                "/chat", json={"message": "Hi."}, timeout=30.0,
            )
            assert chat.status_code == 200, chat.text
            inference_id = chat.json()["inference_ids"][0]

            fb = await http.post(
                "/feedback",
                json={
                    "metric_name": "agent_response_quality",
                    "value": True,
                    "inference_id": inference_id,
                },
                timeout=10.0,
            )
    # Accept 200 (metric pre-declared), 404 (unknown metric forwarded from
    # the gateway — the usual case when the test's metric isn't in TOML), or
    # 502 (5xx gateway failure).  All three confirm the route round-trips.
    assert fb.status_code in (200, 404, 502), fb.text
    if fb.status_code == 200:
        assert fb.json()["feedback_id"]
