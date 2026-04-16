"""Unit tests for :func:`mount_agent`."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.serve import mount_agent
from tests._fake_t0 import FakeChatResponse, FakeT0
from tests.test_serve.conftest import FakeFastMCPClient


async def test_mount_agent_exposes_endpoints_under_prefix(
    base_agent: Agent,
) -> None:
    """Mounting under ``/mobile`` makes the routes available at that prefix."""
    parent = FastAPI(title="parent")
    mount_agent(
        parent, base_agent, "/mobile",
        client_factory=lambda: FakeFastMCPClient(),
    )

    async with AsyncClient(
        transport=ASGITransport(app=parent), base_url="http://test",
    ) as http:
        info_resp = await http.get("/mobile/info")
        health_resp = await http.get("/mobile/health")
        chat_resp = await http.post(
            "/mobile/chat", json={"message": "hello"},
        )

    assert info_resp.status_code == 200
    assert info_resp.json()["name"] == "serve-test"
    assert health_resp.status_code == 200
    assert chat_resp.status_code == 200
    assert chat_resp.json()["response"] == "hi"


async def test_mount_multiple_agents_disjoint_prefixes(
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """Two agents on two prefixes don't cross-contaminate."""
    from tensorzero.types import Text

    t0_a = FakeT0(responses=[
        FakeChatResponse(content=[Text(type="text", text="A")]),
    ])
    t0_b = FakeT0(responses=[
        FakeChatResponse(content=[Text(type="text", text="B")]),
    ])
    agent_a = Agent(
        AgentConfig(name="a", function_name="agent"),
        t0=t0_a,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )
    agent_b = Agent(
        AgentConfig(name="b", function_name="agent"),
        t0=t0_b,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )

    parent = FastAPI()
    mount_agent(parent, agent_a, "/a", client_factory=lambda: FakeFastMCPClient())
    mount_agent(parent, agent_b, "/b", client_factory=lambda: FakeFastMCPClient())

    async with AsyncClient(
        transport=ASGITransport(app=parent), base_url="http://test",
    ) as http:
        r_a = await http.post("/a/chat", json={"message": "x"})
        r_b = await http.post("/b/chat", json={"message": "y"})

    assert r_a.json()["response"] == "A"
    assert r_b.json()["response"] == "B"
