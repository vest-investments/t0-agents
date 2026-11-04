"""Authed remote MCP mounts on the glow ``App``.

Two first-class capabilities land here:

* ``App(servers=[...])`` accepts ``MCPServer`` **instances**, not just
  subclasses — so a per-mount config (url + headers/auth) needs no
  module-level subclass.
* A ``url`` mount carries ``headers=`` (static) and ``auth=`` (an
  :class:`httpx.Auth`, applied per-request), replacing the pilot's
  ``create_proxy(MCPConfig(...))`` + dynamic-subclass bridge.

The unit tests pin glow's contract (which transport it builds); the
guarded end-to-end witness proves the four identity headers actually
reach a server over real HTTP — mirroring the M9 pilot witness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any, cast

import httpx
import pytest
from fastmcp import FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, MCPServer
from t0_agents.mcp.runtime import _remote_transport as _remote_target


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


class _ServiceAuth(httpx.Auth):
    """Inject a service-auth header per request (the pilot's ``ServiceAuth``
    shape — kept test-local; glow only needs *an* ``httpx.Auth``)."""

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["X-Service-Auth-Token"] = self._token
        yield request


_MOUNT_SERVER: FastMCP[Any] = FastMCP("mount-tools")


@_MOUNT_SERVER.tool
async def ping() -> str:
    """Trivial tool for the instance-mount assertion."""
    return "pong"


# --- Instances in servers= -------------------------------------------------


def test_app_accepts_mcp_server_instance_in_servers() -> None:
    """An ``MCPServer`` **instance** in ``servers=`` seeds a spec via
    ``_to_spec`` — the subclass is no longer mandatory."""
    inst = MCPServer(url="https://foo/mcp", namespace="foo", headers={"X-A": "1"})
    app = App(t0=_fake_t0(), servers=[inst])
    (spec,) = app._mcp_specs
    assert spec.url == "https://foo/mcp"
    assert spec.namespace == "foo"
    assert spec.headers == {"X-A": "1"}


async def test_app_mounts_instance_server_through_owned_client() -> None:
    """An in-memory-server instance mounts end-to-end: its tool surfaces on
    the App-owned client (real FastMCP boundary, offline)."""
    app = App(
        t0=_fake_t0(),
        servers=[MCPServer(server=_MOUNT_SERVER, namespace="calc")],
    )
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        tool_names = {t.name for t in await client.list_tools()}
    assert "calc_ping" in tool_names


# --- Validation ------------------------------------------------------------


def test_headers_rejected_on_in_memory_server() -> None:
    with pytest.raises(ConfigurationError, match="only to a remote"):
        MCPServer(server=_MOUNT_SERVER, headers={"X-A": "1"})


def test_auth_rejected_on_in_memory_server() -> None:
    with pytest.raises(ConfigurationError, match="only to a remote"):
        MCPServer(server=_MOUNT_SERVER, auth=_ServiceAuth("tok"))


# --- _remote_target: which transport glow builds ---------------------------


def test_remote_target_bare_url_when_unauthed() -> None:
    """No headers/auth → a bare URL string, letting create_proxy infer."""
    spec = MCPServer(url="https://x/mcp", namespace="x")._to_spec()
    assert _remote_target(spec) == "https://x/mcp"


def test_remote_target_builds_http_transport_with_headers_and_auth() -> None:
    auth = _ServiceAuth("svc-tok")
    spec = MCPServer(
        url="https://x/mcp", headers={"X-A": "1"}, auth=auth
    )._to_spec()
    target = _remote_target(spec)
    from fastmcp.client.transports import StreamableHttpTransport

    assert isinstance(target, StreamableHttpTransport)
    assert target.headers == {"X-A": "1"}
    assert target.auth is auth


def test_remote_target_builds_sse_transport() -> None:
    spec = MCPServer(
        url="https://x/sse", transport="sse", headers={"X-A": "1"}
    )._to_spec()
    from fastmcp.client.transports import SSETransport

    assert isinstance(_remote_target(spec), SSETransport)


def test_remote_target_rejects_authed_non_http_transport() -> None:
    spec = MCPServer(
        url="stdio://x", transport="stdio", headers={"X-A": "1"}
    )._to_spec()
    with pytest.raises(ConfigurationError, match="require transport 'http' or 'sse'"):
        _remote_target(spec)


def test_remote_target_timeout_alone_builds_explicit_transport() -> None:
    """timeout= forces the explicit-transport path even without headers/auth."""
    spec = MCPServer(url="https://x/mcp", timeout=2.0)._to_spec()
    target = _remote_target(spec)
    from fastmcp.client.transports import StreamableHttpTransport

    assert isinstance(target, StreamableHttpTransport)
    assert target.httpx_client_factory is not None


async def test_remote_target_timeout_factory_pins_read_and_mints_fresh_clients() -> None:
    """The factory sets httpx read=timeout and returns a NEW client per call
    (fastmcp closes each client on disconnect — a shared one would arrive
    already-closed on the next connect)."""
    spec = MCPServer(
        url="https://x/mcp", headers={"X-A": "1"}, timeout=2.0
    )._to_spec()
    target = _remote_target(spec)
    from fastmcp.client.transports import StreamableHttpTransport

    assert isinstance(target, StreamableHttpTransport)
    factory = target.httpx_client_factory
    assert factory is not None
    c1 = factory(headers={"X-A": "1"})
    c2 = factory(headers={"X-A": "1"})
    try:
        assert c1 is not c2
        assert c1.timeout.read == 2.0
        assert c1.timeout.connect == 30.0
    finally:
        await c1.aclose()
        await c2.aclose()


# --- End-to-end identity-header witness (real HTTP, guarded) ---------------


async def test_authed_mount_propagates_identity_headers() -> None:
    """The four identity headers (3 static + 1 via ``httpx.Auth``) reach a
    server over real HTTP through glow's mount — the M9 witness, in-loop.

    Skipped when the ``serve`` extra (uvicorn) is absent."""
    uvicorn = pytest.importorskip("uvicorn")

    seen: dict[str, str] = {}
    recorder: FastMCP[Any] = FastMCP("recorder")

    @recorder.tool
    async def whoami() -> str:
        return "ok"

    inner = recorder.http_app(path="/mcp")

    async def capturing_app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            for key, value in scope.get("headers", []):
                seen[key.decode().lower()] = value.decode()
        await inner(scope, receive, send)

    config = uvicorn.Config(
        capturing_app, host="127.0.0.1", port=0, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]

        app = App(
            t0=_fake_t0(),
            servers=[
                MCPServer(
                    url=f"http://127.0.0.1:{port}/mcp",
                    namespace="svc",
                    headers={
                        "X-Tenant-Id": "c-1",
                        "X-Tenant-Token": "tok-1",
                        "X-Session-Source": "api",
                    },
                    auth=_ServiceAuth("service-token"),
                )
            ],
        )
        async with app.lifespan():
            client = app._owned_client
            assert client is not None
            result = await client.call_tool("svc_whoami", {})
        assert result.data == "ok"
    finally:
        server.should_exit = True
        await serve_task

    assert seen.get("x-tenant-id") == "c-1"
    assert seen.get("x-tenant-token") == "tok-1"
    assert seen.get("x-session-source") == "api"
    assert seen.get("x-service-auth-token") == "service-token"


async def test_per_message_app_lifecycle_over_shared_server() -> None:
    """vest-ai builds a fresh App per user message; consecutive lifespans
    against one server must each mount, list, and call cleanly (the
    compat proof for the runtime swap).

    Skipped when the ``serve`` extra (uvicorn) is absent."""
    uvicorn = pytest.importorskip("uvicorn")

    recorder: FastMCP[Any] = FastMCP("recorder")

    @recorder.tool
    async def whoami() -> str:
        return "ok"

    config = uvicorn.Config(
        recorder.http_app(path="/mcp"),
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]

        for message in range(3):
            app = App(
                t0=_fake_t0(),
                servers=[
                    MCPServer(
                        url=f"http://127.0.0.1:{port}/mcp",
                        namespace="svc",
                        headers={"X-Tenant-Id": f"c-{message}"},
                        auth=_ServiceAuth("service-token"),
                    )
                ],
            )
            async with app.lifespan():
                client = app._owned_client
                assert client is not None
                tool_names = {t.name for t in await client.list_tools()}
                assert "svc_whoami" in tool_names
                result = await client.call_tool("svc_whoami", {})
                assert result.data == "ok"
    finally:
        server.should_exit = True
        await serve_task
