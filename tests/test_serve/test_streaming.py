"""Offline tests for the SSE + WebSocket streaming routes."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.serve import create_app
from tests._fake_t0 import (
    FakeT0,
    chat_chunk,
    final_chunk,
    text_chunk,
)
from tests.test_serve.conftest import FakeFastMCPClient


@pytest.fixture
def streaming_client_factory() -> Callable[[], FakeFastMCPClient]:
    """Zero-arg factory minting fresh FakeFastMCPClients for streaming tests."""

    def _factory() -> FakeFastMCPClient:
        return FakeFastMCPClient()

    return _factory


@pytest.fixture
def elicitation_client_factory() -> Callable[[Any], FakeFastMCPClient]:
    """Factory that accepts an elicitation handler per FastMCP Q3."""

    def _factory(handler: Any) -> FakeFastMCPClient:
        client = FakeFastMCPClient()
        client._elicitation_handler = handler  # type: ignore[attr-defined]
        return client

    return _factory


# -------------------------------------------------------------------------
# SSE
# -------------------------------------------------------------------------


async def test_sse_streams_events_until_done(
    streaming_client_factory: Callable[[], FakeFastMCPClient],
) -> None:
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("hola ", block_id="t1")]),
                chat_chunk([text_chunk("mundo", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(name="sse", function_name="agent"),
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )
    app = create_app(
        agent,
        client_factory=streaming_client_factory,
        streaming=True,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as http:
        async with http.stream(
            "GET", "/chat/stream", params={"message": "hi"},
        ) as resp:
            assert resp.status_code == 200
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk

    decoded = body.decode("utf-8")
    assert "event: text_delta" in decoded
    assert "event: text_end" in decoded
    assert "event: turn_end" in decoded
    assert "event: done" in decoded

    # Verify the DoneEvent payload carries the joined response text.
    done_idx = decoded.find("event: done")
    assert done_idx != -1
    data_line_start = decoded.find("data:", done_idx)
    # SSE line separator is configurable (\r\n, \n, etc.) — find either.
    data_end = decoded.find("\n", data_line_start)
    done_payload = json.loads(
        decoded[data_line_start + len("data:") : data_end].strip(),
    )
    assert done_payload["type"] == "done"
    assert done_payload["result"]["response"] == "hola mundo"


# -------------------------------------------------------------------------
# WebSocket
# -------------------------------------------------------------------------


def test_ws_streams_events_and_closes_cleanly(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("¡listo!", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(name="ws", function_name="agent"),
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )
    app = create_app(
        agent,
        client_factory=lambda: FakeFastMCPClient(),
        elicitation_client_factory=elicitation_client_factory,
        streaming=True,
    )

    client = TestClient(app)
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json({"type": "user_message", "message": "hola"})
        collected: list[dict[str, Any]] = []
        while True:
            frame = ws.receive_json()
            collected.append(frame)
            if frame["type"] == "done":
                break

    types = [f["type"] for f in collected]
    assert "text_delta" in types
    assert "text_end" in types
    assert "turn_end" in types
    assert types[-1] == "done"
    assert collected[-1]["result"]["response"] == "¡listo!"


def test_ws_rejects_invalid_frames_without_closing(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    fake_t0 = FakeT0(responses=[])
    agent = Agent(
        AgentConfig(name="ws-bad", function_name="agent"),
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )
    app = create_app(
        agent,
        client_factory=lambda: FakeFastMCPClient(),
        elicitation_client_factory=elicitation_client_factory,
        streaming=True,
    )
    client = TestClient(app)
    with client.websocket_connect("/chat/ws") as ws:
        # Missing type field — WSErrorFrame comes back, connection stays open.
        ws.send_json({"foo": "bar"})
        frame = ws.receive_json()
        assert frame["type"] == "ws_error"


# -------------------------------------------------------------------------
# create_app validation for streaming kwargs
# -------------------------------------------------------------------------


def test_create_app_rejects_elicitation_factory_without_client_factory(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    fake_t0 = FakeT0(responses=[])
    agent = Agent(
        AgentConfig(name="bad", function_name="agent"),
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )
    with pytest.raises(ConfigurationError) as exc:
        create_app(
            agent,
            share_client=True,
            elicitation_client_factory=elicitation_client_factory,
        )
    assert "additive to 'client_factory'" in str(exc.value)


def test_session_provider_mode_via_create_app(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    """Passing session_provider=... should skip the XOR dispatch branches."""
    from contextlib import AbstractAsyncContextManager, asynccontextmanager

    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.serve.session import SessionProvider

    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("provider path", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(name="provider", function_name="agent"),
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=FakeFastMCPClient(),
    )

    class MyProvider:
        def build_client(
            self,
            *,
            elicitation_handler: ElicitationHandler | None = None,
        ) -> AbstractAsyncContextManager[Any]:
            @asynccontextmanager
            async def _mgr() -> Any:
                client = FakeFastMCPClient()
                async with client:
                    yield client

            return _mgr()

    provider = MyProvider()
    assert isinstance(provider, SessionProvider)

    app = create_app(agent, session_provider=provider, streaming=True)
    client = TestClient(app)
    with client.websocket_connect("/chat/ws") as ws:
        ws.send_json({"type": "user_message", "message": "hi"})
        collected: list[dict[str, Any]] = []
        while True:
            frame = ws.receive_json()
            collected.append(frame)
            if frame["type"] == "done":
                break
    assert collected[-1]["result"]["response"] == "provider path"
