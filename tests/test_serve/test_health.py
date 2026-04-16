"""Unit tests for ``GET /health``."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.serve import create_app
from tests.test_serve.conftest import FakeFastMCPClient


async def test_health_per_session_mode(base_agent: Agent) -> None:
    """Per-session mode reports ``tool_provider: 'per-session'`` (no probe)."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        resp = await http.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"gateway": "ok", "tool_provider": "per-session"}


async def test_health_share_client_connected(base_agent: Agent) -> None:
    """Shared mode with a connected client reports ``'ok'``."""
    # Simulate a connected shared client by flipping is_connected()
    base_agent._tool_provider._connected = True  # type: ignore[attr-defined]
    app = create_app(base_agent, share_client=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        resp = await http.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"gateway": "ok", "tool_provider": "ok"}


async def test_health_share_client_disconnected(base_agent: Agent) -> None:
    """Shared mode with a disconnected client reports ``'down'``."""
    # Default FakeFastMCPClient is disconnected until __aenter__
    app = create_app(base_agent, share_client=True)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        resp = await http.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"gateway": "ok", "tool_provider": "down"}
