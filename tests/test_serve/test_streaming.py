"""Offline tests for the SSE + WebSocket streaming routes."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

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
    """Factory that accepts a per-connection elicitation handler."""

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
                final_chunk(input_tokens=120, output_tokens=8),
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
    # Additive usage block on the done frame; the `result` projection above is
    # unchanged. SSE is single-invoke, so there is no per-session block.
    usage = done_payload["usage"]
    assert usage is not None
    assert usage["footprint"]["input_tokens"] == 120
    assert usage["footprint"]["output_tokens"] == 8
    assert usage["cumulative"]["input_tokens"] == 120
    assert usage["session"] is None


# -------------------------------------------------------------------------
# WebSocket
# -------------------------------------------------------------------------


def _drain_to_done(ws: Any) -> dict[str, Any]:
    """Read WS frames until the terminal ``done`` frame, which is returned."""
    while True:
        frame = ws.receive_json()
        if frame["type"] == "done":
            return frame


def test_ws_streams_events_and_closes_cleanly(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("¡listo!", block_id="t1")]),
                final_chunk(input_tokens=90, output_tokens=6),
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
    # Additive usage block; `result` keys unchanged. WS carries the per-session
    # running total — one turn so far, so total == footprint.
    usage = collected[-1]["usage"]
    assert usage is not None
    assert usage["footprint"]["input_tokens"] == 90
    assert usage["cumulative"]["input_tokens"] == 90
    assert usage["session"]["total"]["input_tokens"] == 90
    assert usage["session"]["runs_recorded"] == 1


def test_ws_session_usage_accumulates_across_turns(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    """Two turns on one socket sharing an episode: footprint tracks the latest
    prompt size (a level) while the per-session billing total accumulates."""
    episode = uuid4()
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("turn one", block_id="t1")]),
                final_chunk(input_tokens=100, output_tokens=10, episode_id=episode),
            ],
            [
                chat_chunk([text_chunk("turn two", block_id="t1")]),
                final_chunk(input_tokens=250, output_tokens=20, episode_id=episode),
            ],
        ]
    )
    agent = Agent(
        AgentConfig(name="ws-accum", function_name="agent"),
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
        ws.send_json(
            {"type": "user_message", "message": "uno", "episode_id": str(episode)},
        )
        done1 = _drain_to_done(ws)
        ws.send_json(
            {"type": "user_message", "message": "dos", "episode_id": str(episode)},
        )
        done2 = _drain_to_done(ws)

    u1, u2 = done1["usage"], done2["usage"]
    # Per-turn footprint is a level — the latest prompt size, not a sum.
    assert u1["footprint"]["input_tokens"] == 100
    assert u2["footprint"]["input_tokens"] == 250
    # Per-session billing total accumulates across both turns of the episode.
    assert u1["session"]["total"]["input_tokens"] == 100
    assert u1["session"]["runs_recorded"] == 1
    assert u2["session"]["total"]["input_tokens"] == 350  # 100 + 250
    assert u2["session"]["runs_recorded"] == 2
    # Session footprint is the latest level, not the accumulated sum.
    assert u2["session"]["footprint"]["input_tokens"] == 250


async def test_sse_tags_query_param_reaches_gateway(
    streaming_client_factory: Callable[[], FakeFastMCPClient],
) -> None:
    """SSE ``?tags=<json>`` propagates to ``agent.invoke(tags=...).events()``."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(name="sse-tags", function_name="agent"),
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
            "GET",
            "/chat/stream",
            params={"message": "hi", "tags": json.dumps({"env": "staging"})},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_bytes():
                pass

    assert fake_t0.calls[0]["tags"] == {"env": "staging"}


async def test_sse_invalid_tags_yields_error_event(
    streaming_client_factory: Callable[[], FakeFastMCPClient],
) -> None:
    """Non-JSON ``?tags=`` surfaces as the first SSE frame (ErrorEvent)."""
    fake_t0 = FakeT0(responses=[])
    agent = Agent(
        AgentConfig(name="sse-bad", function_name="agent"),
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
            "GET",
            "/chat/stream",
            params={"message": "hi", "tags": "not-json"},
        ) as resp:
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk

    decoded = body.decode("utf-8")
    assert "event: error" in decoded
    assert '"kind": "invalid_tags"' in decoded
    # No inference was attempted — the parser rejected before session opened.
    assert fake_t0.calls == []


def test_ws_user_message_tags_reach_gateway(
    elicitation_client_factory: Callable[[Any], FakeFastMCPClient],
) -> None:
    """``WSUserMessage.tags`` must propagate to the T0 inference call."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ]
    )
    agent = Agent(
        AgentConfig(
            name="ws-tags",
            function_name="agent",
            tags={"env": "prod"},
        ),
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
        ws.send_json({
            "type": "user_message",
            "message": "hola",
            "tags": {"request_id": "r-42"},
        })
        while True:
            frame = ws.receive_json()
            if frame["type"] == "done":
                break

    assert fake_t0.calls[0]["tags"] == {
        "env": "prod",
        "request_id": "r-42",
    }


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
