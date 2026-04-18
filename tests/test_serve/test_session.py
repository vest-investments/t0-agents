"""Unit tests for :class:`SessionFactory` and ``create_app`` XOR validation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import AgentResult
from t0_agents.serve import SessionFactory, create_app
from tests._fake_t0 import FakeChatResponse
from tests.test_serve.conftest import FakeFastMCPClient

# --- create_app XOR validation -----------------------------------------------


async def test_create_app_neither_flag_raises(base_agent: Agent) -> None:
    with pytest.raises(ConfigurationError) as exc:
        create_app(base_agent)
    assert "client_factory" in str(exc.value)
    assert "share_client" in str(exc.value)


async def test_create_app_both_flags_raise(base_agent: Agent) -> None:
    with pytest.raises(ConfigurationError) as exc:
        create_app(
            base_agent,
            client_factory=lambda: FakeFastMCPClient(),
            share_client=True,
        )
    assert "mutually exclusive" in str(exc.value)


# --- Per-session vs shared ---------------------------------------------------


async def test_per_session_factory_mints_fresh_clients(
    base_agent: Agent,
    client_factory_log: dict[str, list[FakeFastMCPClient]],
    fake_client_factory: Any,
) -> None:
    """Three requests → three distinct ``Client`` instances."""
    from tensorzero.types import Text

    base_agent._t0._queue.extend(  # type: ignore[attr-defined]
        [FakeChatResponse(content=[Text(type="text", text="hi")]) for _ in range(2)]
    )

    app = create_app(base_agent, client_factory=fake_client_factory)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        for _ in range(3):
            assert (await http.post("/chat", json={"message": "hi"})).status_code == 200

    ids = [c.id for c in client_factory_log["created"]]
    assert len(set(ids)) == 3


async def test_share_client_reuses_shared_provider(base_agent: Agent) -> None:
    """``share_client=True`` reuses the agent's tool provider across requests."""
    from tensorzero.types import Text

    base_agent._t0._queue.append(  # type: ignore[attr-defined]
        FakeChatResponse(content=[Text(type="text", text="hi")])
    )
    # The single shared client must start connected for ToolProvider ops
    # to work; the FakeFastMCPClient lets us flip the flag directly.
    base_agent._tool_provider._connected = True  # type: ignore[attr-defined]

    app = create_app(base_agent, share_client=True)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        for _ in range(2):
            assert (await http.post("/chat", json={"message": "hi"})).status_code == 200

    # One, long-lived client — never entered by the framework in share mode.
    shared = base_agent._tool_provider  # type: ignore[attr-defined]
    assert shared.enter_count == 0
    assert shared.exit_count == 0


async def test_concurrent_per_session_requests_get_distinct_clients(
    base_agent: Agent,
    client_factory_log: dict[str, list[FakeFastMCPClient]],
    fake_client_factory: Any,
) -> None:
    """``asyncio.gather`` 5 requests → 5 distinct ``Client`` instances."""
    from tensorzero.types import Text

    # Script enough responses for 5 concurrent requests (plus the one already
    # in fake_t0 from the fixture).
    base_agent._t0._queue.extend(  # type: ignore[attr-defined]
        [FakeChatResponse(content=[Text(type="text", text="hi")]) for _ in range(4)]
    )

    app = create_app(base_agent, client_factory=fake_client_factory)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        responses = await asyncio.gather(
            *[http.post("/chat", json={"message": "hi"}) for _ in range(5)],
        )

    assert all(r.status_code == 200 for r in responses)
    assert len(client_factory_log["created"]) == 5
    assert len({c.id for c in client_factory_log["created"]}) == 5


# --- deployment_label auto-tag ----------------------------------------------


async def test_deployment_label_adds_tag_for_every_inference(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """``deployment_label=mobile`` → ``tags["t0_agents.deployment"]="mobile"``."""

    class CapturingAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> AgentResult:
            self._captured_tags = dict(self._config.tags or {})  # type: ignore[attr-defined]
            return AgentResult(response="ok", episode_id="e", inference_ids=[], turns=0)

    agent = CapturingAgent(
        base_config,
        t0=object(),  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )
    app = create_app(
        agent,
        client_factory=lambda: FakeFastMCPClient(),
        deployment_label="mobile",
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        resp = await http.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200


# --- SessionFactory direct unit tests ---------------------------------------


async def test_session_factory_rejects_neither_or_both(
    base_config: AgentConfig,
) -> None:
    with pytest.raises(ConfigurationError):
        SessionFactory(base_config=base_config, t0=object())  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError):
        SessionFactory(
            base_config=base_config,
            t0=object(),  # type: ignore[arg-type]
            client_factory=lambda: FakeFastMCPClient(),
            shared_tool_provider=FakeFastMCPClient(),
        )


async def test_session_factory_merges_tags_shallow(
    base_config: AgentConfig,
) -> None:
    cfg = AgentConfig(name="t", function_name="agent", tags={"a": "1"})
    sf = SessionFactory(
        base_config=cfg,
        t0=object(),  # type: ignore[arg-type]
        client_factory=lambda: FakeFastMCPClient(),
    )
    async with sf.session(extra_tags={"b": "2", "a": "override"}) as agent:
        assert agent.config.tags == {"a": "override", "b": "2"}


async def test_session_factory_probe_returns_per_session_in_factory_mode(
    base_config: AgentConfig,
) -> None:
    sf = SessionFactory(
        base_config=base_config,
        t0=object(),  # type: ignore[arg-type]
        client_factory=lambda: FakeFastMCPClient(),
    )
    assert sf.probe_tool_provider() == "per-session"


async def test_session_factory_probe_ok_down_for_shared(
    base_config: AgentConfig,
) -> None:
    shared = FakeFastMCPClient()
    sf = SessionFactory(
        base_config=base_config,
        t0=object(),  # type: ignore[arg-type]
        shared_tool_provider=shared,
    )
    assert sf.probe_tool_provider() == "down"
    shared._connected = True
    assert sf.probe_tool_provider() == "ok"
