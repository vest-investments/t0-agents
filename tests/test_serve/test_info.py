"""Unit tests for ``GET /info``."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.guardrails import block_tools, max_tool_calls
from t0_agents.memory import InMemoryMemory
from t0_agents.serve import create_app
from tests._fake_t0 import FakeT0
from tests.test_serve.conftest import FakeFastMCPClient


async def test_info_minimal_config(base_agent: Agent) -> None:
    """Bare config → ``has_memory=False`` and empty guardrail list."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        resp = await http.get("/info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "serve-test"
    assert body["function_name"] == "agent"
    assert body["namespace"] is None
    assert body["tags"] is None
    assert body["has_memory"] is False
    assert body["guardrails"] == []


async def test_info_reports_namespace_and_tags(
    fake_t0: FakeT0, fake_tool_provider: FakeFastMCPClient,
) -> None:
    """Namespace and tags appear in the redacted snapshot."""
    cfg = AgentConfig(
        name="mobile",
        function_name="agent",
        namespace="mobile",
        tags={"team": "core"},
    )
    agent = Agent(cfg, t0=fake_t0, tool_provider=fake_tool_provider)  # type: ignore[arg-type]
    app = create_app(
        agent, client_factory=lambda: FakeFastMCPClient(), deployment_label="mobile",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        body = (await http.get("/info")).json()

    assert body["namespace"] == "mobile"
    assert body["tags"] == {"team": "core", "t0_agents.deployment": "mobile"}


async def test_info_lists_guardrails_by_name(
    fake_t0: FakeT0, fake_tool_provider: FakeFastMCPClient,
) -> None:
    """Configured guardrails appear as a list of names."""
    cfg = AgentConfig(
        name="t",
        function_name="agent",
        guardrails=[block_tools(["delete_db"]), max_tool_calls(per_run=5)],
    )
    agent = Agent(cfg, t0=fake_t0, tool_provider=fake_tool_provider)  # type: ignore[arg-type]
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        body = (await http.get("/info")).json()

    # Guardrail names include their parameters for audit purposes — that's
    # the built-in convention, not a secret.
    assert any("block_tools" in name for name in body["guardrails"])
    assert any("max_tool_calls" in name for name in body["guardrails"])
    assert len(body["guardrails"]) == 2


async def test_info_has_memory_true(
    fake_t0: FakeT0, fake_tool_provider: FakeFastMCPClient,
) -> None:
    """``has_memory=True`` when a ``MemoryProvider`` is configured."""
    cfg = AgentConfig(
        name="t", function_name="agent", memory=InMemoryMemory(),
    )
    agent = Agent(cfg, t0=fake_t0, tool_provider=fake_tool_provider)  # type: ignore[arg-type]
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        body = (await http.get("/info")).json()

    assert body["has_memory"] is True
