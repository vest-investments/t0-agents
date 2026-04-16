"""Unit tests for ``POST /chat``.

Stays offline: goes through ``httpx.AsyncClient`` backed by
``ASGITransport`` (no uvicorn) and runs the full ``Agent.run`` loop
against :class:`FakeT0` + :class:`FakeFastMCPClient`.  Live-gateway
coverage lives in ``tests/test_serve/test_integration.py``.
"""

from __future__ import annotations

from typing import Any

from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    GuardrailBlockedError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.serve import create_app
from tests.test_serve.conftest import FakeFastMCPClient


async def _post_chat(app: Any, payload: dict[str, Any]) -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        return await http.post("/chat", json=payload)


async def test_chat_round_trip(base_agent: Agent) -> None:
    """Happy path: response mirrors ``AgentResult`` shape."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_chat(app, {"message": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "hi"
    assert body["turns"] == 1
    assert body["response_source"] == "text"
    assert isinstance(body["episode_id"], str) and body["episode_id"]
    assert isinstance(body["inference_ids"], list)
    assert body["guardrail_log"] == []


async def test_chat_maxturns_returns_408(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """``MaxTurnsExceededError`` maps to HTTP 408."""

    class FailingAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> Any:
            raise MaxTurnsExceededError("ran out of turns")

    agent = FailingAgent(
        base_config,
        t0=object(),  # type: ignore[arg-type] - never called
        tool_provider=fake_tool_provider,
    )
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_chat(app, {"message": "x"})
    assert resp.status_code == 408
    assert "ran out of turns" in resp.json()["detail"]


async def test_chat_guardrail_blocked_returns_409(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """``GuardrailBlockedError`` maps to HTTP 409."""

    class BlockingAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> Any:
            raise GuardrailBlockedError("cost cap exceeded")

    agent = BlockingAgent(
        base_config,
        t0=object(),  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_chat(app, {"message": "x"})
    assert resp.status_code == 409
    assert "cost cap exceeded" in resp.json()["detail"]


async def test_chat_generic_t0agents_error_returns_400(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """Any other ``T0AgentsError`` maps to HTTP 400."""

    class ErroringAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> Any:
            raise T0AgentsError("something else")

    agent = ErroringAgent(
        base_config,
        t0=object(),  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_chat(app, {"message": "x"})
    assert resp.status_code == 400


async def test_chat_tags_merge(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """Per-request tags merge on top of base config + deployment_label."""
    captured: dict[str, Any] = {}

    class ObservingAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> Any:
            captured["tags"] = dict(self._config.tags or {})
            # Return a minimal AgentResult shape
            from t0_agents.core.types import AgentResult
            return AgentResult(
                response="ok", episode_id="e-1", inference_ids=[], turns=0,
            )

    cfg = AgentConfig(
        name="t", function_name="agent",
        tags={"base": "yes"},
    )
    agent = ObservingAgent(
        cfg,
        t0=object(),  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )
    app = create_app(
        agent,
        client_factory=lambda: FakeFastMCPClient(),
        deployment_label="mobile",
    )

    resp = await _post_chat(app, {
        "message": "x", "tags": {"per_req": "abc"},
    })
    assert resp.status_code == 200
    assert captured["tags"] == {
        "base": "yes",
        "t0_agents.deployment": "mobile",
        "per_req": "abc",
    }


async def test_chat_guardrail_log_serialized(
    base_config: AgentConfig,
    fake_tool_provider: FakeFastMCPClient,
) -> None:
    """``AgentResult.guardrail_log`` serialises to plain dicts in JSON."""
    from t0_agents.core.types import AgentResult, GuardrailLogEntry

    class LoggingAgent(Agent):
        async def run(  # type: ignore[override]
            self, user_message: str, *, episode_id: str | None = None,
        ) -> Any:
            return AgentResult(
                response="ok",
                episode_id="e-1",
                inference_ids=[],
                turns=0,
                guardrail_log=[
                    GuardrailLogEntry(
                        guardrail_name="max_cost",
                        hook="before_run",
                        kind="pass",
                    ),
                    GuardrailLogEntry(
                        guardrail_name="block_tools",
                        hook="before_tool",
                        kind="block",
                        tool_name="delete_db",
                        detail="forbidden",
                    ),
                ],
            )

    agent = LoggingAgent(
        base_config,
        t0=object(),  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )
    app = create_app(agent, client_factory=lambda: FakeFastMCPClient())

    resp = await _post_chat(app, {"message": "hi"})
    assert resp.status_code == 200
    entries = resp.json()["guardrail_log"]
    assert len(entries) == 2
    assert entries[0] == {
        "guardrail_name": "max_cost",
        "hook": "before_run",
        "kind": "pass",
        "tool_name": None,
        "detail": "",
    }
    assert entries[1]["tool_name"] == "delete_db"
    assert entries[1]["detail"] == "forbidden"


async def test_chat_episode_id_forwarded(base_agent: Agent) -> None:
    """``episode_id`` from the request reaches ``Agent.run`` and then ``run_agent``."""
    app = create_app(base_agent, client_factory=lambda: FakeFastMCPClient())

    await _post_chat(app, {"message": "hi", "episode_id": "prev-episode"})

    calls = base_agent._t0.calls  # type: ignore[attr-defined]
    assert len(calls) == 1
    assert calls[0]["episode_id"] == "prev-episode"


async def test_chat_uses_fresh_client_per_request(
    base_agent: Agent,
    client_factory_log: dict[str, list[FakeFastMCPClient]],
    fake_client_factory: Any,
) -> None:
    """Each request opens + closes a fresh ``Client``."""
    # Re-script the FakeT0 with three responses, one per request.
    from tensorzero.types import Text

    from tests._fake_t0 import FakeChatResponse

    base_agent._t0._queue.extend(  # type: ignore[attr-defined]
        [
            FakeChatResponse(content=[Text(type="text", text="hi")])
            for _ in range(2)
        ]
    )

    app = create_app(base_agent, client_factory=fake_client_factory)

    # Use a single AsyncClient to exercise the app lifespan only once.
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        for _ in range(3):
            assert (await http.post("/chat", json={"message": "hi"})).status_code == 200

    assert len(client_factory_log["created"]) == 3
    for client in client_factory_log["created"]:
        assert client.enter_count == 1
        assert client.exit_count == 1
        assert client.is_connected() is False
