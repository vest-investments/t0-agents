"""MCPRuntime unit tests — handles, lifecycle, and the routing contract.

Every test drives through the real FastMCP boundary (in-memory servers via
``Client(FastMCP)``), never a hand-built provider fake — framework-layer
regressions (schema wrapping, elicitation capability negotiation) surface
here or not at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import mcp.types
import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.tasks import TaskNotificationHandler
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from fastmcp.exceptions import ToolError

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import MCPServer
from t0_agents.mcp.elicitation import auto_approve_handler
from t0_agents.mcp.runtime import (
    MCPRuntime,
    ServerCredentials,
    _InventoryMessageHandler,
    _OverlaidSpec,
    _remote_transport,
    _ServerHandle,
)


def _named_server(server_name: str, *tool_names: str) -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP(server_name)
    for tool_name in tool_names:

        def make(tool: str) -> Any:
            async def impl() -> str:
                return f"{server_name}:{tool}"

            impl.__name__ = tool
            impl.__doc__ = f"Return the {tool} marker."
            return impl

        server.tool(make(tool_name))
    return server


def _spec(server: FastMCP[Any], **kwargs: Any) -> Any:
    return MCPServer(server=server, **kwargs)._to_spec()


def _local_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("local")

    @server.tool
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool
    async def confirm(ctx: Context) -> str:
        """Ask for binary approval."""
        result = await ctx.elicit("Approve?", response_type=None)
        return "approved" if result.action == "accept" else "declined"

    return server


async def test_local_server_tools_listed_verbatim() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
    assert {"add", "confirm"} <= names


async def test_call_tool_routes_to_local_server() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        result = await runtime.call_tool("add", {"a": 2, "b": 3})
    assert result.data == 5


async def test_unknown_tool_matches_composite_error_contract() -> None:
    async with MCPRuntime(local_server=_local_server()) as runtime:
        with pytest.raises(ToolError, match="Unknown tool: 'nope'"):
            await runtime.call_tool("nope", {})


async def test_empty_runtime_lists_nothing_and_rejects_calls() -> None:
    async with MCPRuntime() as runtime:
        assert await runtime.list_tools() == []
        with pytest.raises(ToolError, match="Unknown tool"):
            await runtime.call_tool("anything", {})


async def test_connect_is_lazy_and_aclose_idempotent() -> None:
    runtime = MCPRuntime(local_server=_local_server())
    (handle,) = runtime._handles
    assert not handle.client.is_connected()
    async with runtime:
        assert not handle.client.is_connected()
        await runtime.call_tool("add", {"a": 1, "b": 1})
        assert handle.client.is_connected()
    assert not handle.client.is_connected()
    await runtime.aclose()


async def test_elicitation_handler_bound_to_local_handle() -> None:
    async with MCPRuntime(
        local_server=_local_server(), elicitation_handler=auto_approve_handler
    ) as runtime:
        result = await runtime.call_tool("confirm", {})
    assert result.data == "approved"


# --- Inventory: prefixes, mount parity, collisions -------------------------


async def test_namespaced_spec_prefixes_names_with_mount_parity() -> None:
    spec = _spec(_named_server("calc-tools", "ping"), namespace="calc")
    async with MCPRuntime([spec]) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
        result = await runtime.call_tool("calc_ping", {})
    assert names == {"calc_ping"}
    assert result.data == "calc-tools:ping"


async def test_prefix_optout_presents_names_verbatim() -> None:
    spec = _spec(_named_server("solo", "lookup"), namespace="solo", prefix=False)
    async with MCPRuntime([spec]) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
        result = await runtime.call_tool("lookup", {})
    assert names == {"lookup"}
    assert result.data == "solo:lookup"


async def test_local_and_namespaced_specs_merge() -> None:
    spec = _spec(_named_server("calc-tools", "ping"), namespace="calc")
    async with MCPRuntime([spec], local_server=_local_server()) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
    assert {"add", "confirm", "calc_ping"} <= names


async def test_longest_namespace_wins_routing() -> None:
    fred = _spec(_named_server("fred-srv", "data"), namespace="fred")
    fred_api = _spec(_named_server("fred-api-srv", "x"), namespace="fred_api")
    async with MCPRuntime([fred, fred_api]) as runtime:
        names = {tool.name for tool in await runtime.list_tools()}
        via_long = await runtime.call_tool("fred_api_x", {})
        via_short = await runtime.call_tool("fred_data", {})
    assert names == {"fred_data", "fred_api_x"}
    assert via_long.data == "fred-api-srv:x"
    assert via_short.data == "fred-srv:data"


async def test_final_name_collision_raises_configuration_error() -> None:
    fred = _spec(_named_server("fred-srv", "api_x"), namespace="fred")
    fred_api = _spec(_named_server("fred-api-srv", "x"), namespace="fred_api")
    async with MCPRuntime([fred, fred_api]) as runtime:
        with pytest.raises(ConfigurationError, match=r"collision.*fred_api_x"):
            await runtime.list_tools()


async def test_missing_tool_on_routed_server_raises_tool_error() -> None:
    spec = _spec(_named_server("calc-tools", "ping"), namespace="calc")
    async with MCPRuntime([spec]) as runtime:
        with pytest.raises(ToolError, match="Unknown tool"):
            await runtime.call_tool("calc_nope", {})


async def test_single_unprefixed_handle_routes_without_inventory_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With exactly one unprefixed handle, unprefixed names route directly
    — no per-call list_tools membership check on the hot path."""
    async with MCPRuntime(local_server=_local_server()) as runtime:
        await runtime.call_tool("add", {"a": 1, "b": 1})

        async def boom(self: _ServerHandle) -> Any:
            raise AssertionError("fast path must not list tools")

        monkeypatch.setattr(_ServerHandle, "tools", boom)
        result = await runtime.call_tool("add", {"a": 2, "b": 3})
    assert result.data == 5


async def test_value_identical_specs_fold_into_one_handle() -> None:
    """Registering the same spec twice (servers= plus annotation) was
    harmless on the composite (mount idempotence) — parity here."""
    server = _named_server("calc-tools", "ping")
    spec_a = _spec(server, namespace="calc")
    spec_b = _spec(server, namespace="calc")
    async with MCPRuntime([spec_a, spec_b]) as runtime:
        assert len(runtime._handles) == 1
        names = [tool.name for tool in await runtime.list_tools()]
    assert names == ["calc_ping"]


# --- Server attribution (observability, no I/O) ----------------------------


def test_server_for_attributes_prefixed_name() -> None:
    spec = _spec(_named_server("calc-tools", "ping"), namespace="calc")
    runtime = MCPRuntime([spec])
    assert runtime.server_for("calc_ping") == "calc"


def test_server_for_unprefixed_mount_returns_none() -> None:
    spec = _spec(_named_server("solo", "lookup"), namespace="solo", prefix=False)
    runtime = MCPRuntime([spec])
    assert runtime.server_for("lookup") is None


def test_server_for_local_tool_returns_none() -> None:
    runtime = MCPRuntime(local_server=_local_server())
    assert runtime.server_for("add") is None


def test_server_for_longest_namespace_wins() -> None:
    fred = _spec(_named_server("fred-srv", "data"), namespace="fred")
    fred_api = _spec(_named_server("fred-api-srv", "x"), namespace="fred_api")
    runtime = MCPRuntime([fred, fred_api])
    assert runtime.server_for("fred_api_x") == "fred_api"
    assert runtime.server_for("fred_data") == "fred"


def test_server_for_unknown_name_returns_none() -> None:
    spec = _spec(_named_server("calc-tools", "ping"), namespace="calc")
    runtime = MCPRuntime([spec])
    assert runtime.server_for("totally_unknown") is None


# --- Remote transport builder ----------------------------------------------


class _ServiceAuth(httpx.Auth):
    """Inject a service-auth header per request (glow only needs *an*
    ``httpx.Auth``)."""

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["X-Service-Auth-Token"] = self._token
        yield request


def test_remote_transport_bare_url_when_unauthed() -> None:
    spec = MCPServer(url="https://x/mcp", namespace="x")._to_spec()
    assert _remote_transport(spec) == "https://x/mcp"


def test_remote_transport_builds_http_transport_with_headers_and_auth() -> None:
    auth = _ServiceAuth("svc-tok")
    spec = MCPServer(url="https://x/mcp", headers={"X-A": "1"}, auth=auth)._to_spec()
    target = _remote_transport(spec)
    assert isinstance(target, StreamableHttpTransport)
    assert target.headers == {"X-A": "1"}
    assert target.auth is auth


def test_remote_transport_builds_sse_transport() -> None:
    spec = MCPServer(
        url="https://x/sse", transport="sse", headers={"X-A": "1"}
    )._to_spec()
    assert isinstance(_remote_transport(spec), SSETransport)


def test_remote_transport_rejects_authed_non_http_transport() -> None:
    spec = MCPServer(
        url="stdio://x", transport="stdio", headers={"X-A": "1"}
    )._to_spec()
    with pytest.raises(ConfigurationError, match="require transport 'http' or 'sse'"):
        _remote_transport(spec)


# --- Remote handles over real HTTP (ephemeral uvicorn) ----------------------


@asynccontextmanager
async def _ephemeral_http_server(
    server: FastMCP[Any],
    wrap_asgi: Callable[[Any], Any] | None = None,
) -> AsyncIterator[str]:
    """Serve ``server`` on an OS-assigned port; yield the /mcp URL."""
    uvicorn = pytest.importorskip("uvicorn")

    app: Any = server.http_app(path="/mcp")
    if wrap_asgi is not None:
        app = wrap_asgi(app)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", lifespan="on"
    )
    http = uvicorn.Server(config)
    serve_task = asyncio.create_task(http.serve())
    try:
        while not http.started:
            await asyncio.sleep(0.02)
        port = http.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        http.should_exit = True
        await serve_task


def _elicit_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("remote-elicit")

    @server.tool
    async def order(ctx: Context) -> str:
        """Ask for approval before acting."""
        result = await ctx.elicit("Place the order?", response_type=None)
        return "placed" if result.action == "accept" else "declined"

    return server


async def test_remote_spec_lists_and_calls_over_http() -> None:
    async with _ephemeral_http_server(_named_server("rsrv", "hello")) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            names = {tool.name for tool in await runtime.list_tools()}
            result = await runtime.call_tool("svc_hello", {})
    assert names == {"svc_hello"}
    assert result.data == "rsrv:hello"


async def test_remote_elicitation_reaches_app_scoped_handler() -> None:
    """A remote server's ``ctx.elicit`` must reach the runtime's handler
    through the direct client — the proxy's back-forwarding is gone."""
    async with _ephemeral_http_server(_elicit_server()) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime(
            [spec], elicitation_handler=auto_approve_handler
        ) as runtime:
            result = await runtime.call_tool("svc_order", {})
    assert result.data == "placed"


# --- Inventory cache: TTL, invalidation, refresh ----------------------------


class _StubClient:
    """Counts list_tools round trips; stands in for a remote client in
    cache-arithmetic units (the framework boundary is covered by the
    real-HTTP tests)."""

    def __init__(self) -> None:
        self.list_calls = 0

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def list_tools(self) -> list[mcp.types.Tool]:
        self.list_calls += 1
        return [
            mcp.types.Tool(name=f"call{self.list_calls}", inputSchema={"type": "object"})
        ]


class _Clock:
    def __init__(self) -> None:
        self.value = 1000.0

    def monotonic(self) -> float:
        return self.value


def _stub_handle(stub: _StubClient, *, live_inventory: bool = False) -> _ServerHandle:
    return _ServerHandle(
        "stub",
        namespace=None,
        prefix=False,
        live_inventory=live_inventory,
        client_factory=lambda _invalidate: stub,  # type: ignore[arg-type,return-value]
    )


async def test_remote_inventory_cached_within_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr("t0_agents.mcp.runtime.time", clock)
    stub = _StubClient()
    handle = _stub_handle(stub)
    first = await handle.tools()
    clock.value += 59.0
    second = await handle.tools()
    assert stub.list_calls == 1
    assert first is second


async def test_remote_inventory_refetched_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr("t0_agents.mcp.runtime.time", clock)
    stub = _StubClient()
    handle = _stub_handle(stub)
    await handle.tools()
    clock.value += 61.0
    await handle.tools()
    assert stub.list_calls == 2


async def test_invalidate_forces_refetch_before_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr("t0_agents.mcp.runtime.time", clock)
    stub = _StubClient()
    handle = _stub_handle(stub)
    await handle.tools()
    handle.invalidate()
    await handle.tools()
    assert stub.list_calls == 2


async def test_live_inventory_handles_never_cache() -> None:
    stub = _StubClient()
    handle = _stub_handle(stub, live_inventory=True)
    await handle.tools()
    await handle.tools()
    assert stub.list_calls == 2


async def test_local_tool_registered_after_boot_is_visible() -> None:
    server = _named_server("local-live", "first")
    async with MCPRuntime(local_server=server) as runtime:
        await runtime.call_tool("first", {})

        @server.tool
        async def second() -> str:
            """Late registration."""
            return "second"

        names = {tool.name for tool in await runtime.list_tools()}
        result = await runtime.call_tool("second", {})
    assert {"first", "second"} <= names
    assert result.data == "second"


async def test_message_handler_invalidates_on_list_changed_and_keeps_chain() -> None:
    """The handler must extend ``TaskNotificationHandler`` (fastmcp's
    default) — a bare callable would silently disable task routing."""
    invalidated: list[bool] = []
    handler = _InventoryMessageHandler(lambda: invalidated.append(True))
    assert isinstance(handler, TaskNotificationHandler)

    class _FakeClient:
        pass

    sentinel = _FakeClient()
    handler.bind(sentinel)  # type: ignore[arg-type]
    assert handler._client_ref() is sentinel

    notification = mcp.types.ServerNotification(
        mcp.types.ToolListChangedNotification(
            method="notifications/tools/list_changed"
        )
    )
    await handler(notification)
    assert invalidated == [True]


async def test_message_handler_replaces_client_on_rebuild() -> None:
    """Each factory call must return a freshly-wired client so the rebuilt
    session keeps invalidation + elicitation bindings."""
    async with _ephemeral_http_server(_named_server("rsrv", "hello")) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            (handle,) = runtime._handles
            await runtime.call_tool("svc_hello", {})
            original = handle.client
            await handle.rebuild_if(handle._generation)
            rebuilt = await runtime.call_tool("svc_hello", {})
    assert handle.client is not original
    assert rebuilt.data == "rsrv:hello"


# --- Session-terminated taxonomy --------------------------------------------


class _SessionKiller:
    """ASGI wrapper that 404s the next N POSTs — the SDK maps that to the
    session-terminated JSON-RPC error (code 32600)."""

    def __init__(self) -> None:
        self.kill_next_posts = 0

    def wrap(self, app: Any) -> Any:
        async def wrapper(scope: Any, receive: Any, send: Any) -> None:
            if (
                scope["type"] == "http"
                and scope["method"] == "POST"
                and self.kill_next_posts > 0
            ):
                self.kill_next_posts -= 1
                await send(
                    {"type": "http.response.start", "status": 404, "headers": []}
                )
                await send({"type": "http.response.body", "body": b""})
                return
            await app(scope, receive, send)

        return wrapper


def _counting_server(executions: list[str]) -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("flaky")

    @server.tool
    async def work() -> str:
        """Record one execution."""
        executions.append("work")
        return "done"

    @server.tool
    async def slow() -> str:
        """Record one execution, slowly."""
        executions.append("slow")
        await asyncio.sleep(1.5)
        return "finally"

    return server


async def test_session_terminated_rebuilds_and_resends_exactly_once() -> None:
    executions: list[str] = []
    killer = _SessionKiller()
    async with _ephemeral_http_server(
        _counting_server(executions), wrap_asgi=killer.wrap
    ) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            warm = await runtime.call_tool("svc_work", {})
            (handle,) = runtime._handles
            original_client = handle.client

            killer.kill_next_posts = 1
            recovered = await runtime.call_tool("svc_work", {})

    assert warm.data == "done"
    assert recovered.data == "done"
    assert executions == ["work", "work"]
    assert handle.client is not original_client


async def test_non_session_errors_never_resend() -> None:
    """A per-call timeout fires AFTER the request dispatched — resending
    could double-execute an order-write. It must propagate untouched."""
    from mcp.shared.exceptions import McpError

    executions: list[str] = []
    async with _ephemeral_http_server(_counting_server(executions)) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            await runtime.call_tool("svc_work", {})
            (handle,) = runtime._handles
            client_before = handle.client

            with pytest.raises(McpError):
                await runtime.call_tool("svc_slow", {}, timeout=0.3)
            await asyncio.sleep(1.5)

    assert executions == ["work", "slow"]
    assert handle.client is client_before


async def test_refresh_surfaces_server_side_tool_additions() -> None:
    server = _named_server("growing", "first")
    async with _ephemeral_http_server(server) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            before = {tool.name for tool in await runtime.list_tools()}

            @server.tool
            async def second() -> str:
                """Late registration."""
                return "second"

            cached = {tool.name for tool in await runtime.list_tools()}
            refreshed = {tool.name for tool in await runtime.refresh()}
    assert before == {"svc_first"}
    assert cached == {"svc_first"}
    assert refreshed == {"svc_first", "svc_second"}


# --- Concurrency, teardown, and efficiency battery ---------------------------


def _echo_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("echo-srv")

    @server.tool
    async def echo(value: str) -> str:
        """Echo the value back."""
        return value

    return server


async def test_concurrent_calls_on_one_warm_client_do_not_mix_results() -> None:
    """25 tasks x call_tool over ONE warm client — the SDK's request-id
    multiplexing is the contract the runtime's whole design rests on."""
    async with _ephemeral_http_server(_echo_server()) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            await runtime.list_tools()

            async def one(index: int) -> tuple[int, Any]:
                result = await runtime.call_tool("svc_echo", {"value": f"v{index}"})
                return index, result.data

            results = await asyncio.gather(*(one(index) for index in range(25)))
    assert sorted(results) == [(index, f"v{index}") for index in range(25)]


async def test_parallel_executor_runs_over_the_runtime() -> None:
    """`parallel_tools=True` reaches the runtime through the real loop
    executor — provider-protocol integration, not a direct-call shortcut."""
    from tensorzero import ToolCall

    from t0_agents.mcp.loop import execute_tool_calls_parallel

    async with _ephemeral_http_server(_echo_server()) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            blocks = [
                ToolCall(
                    id=f"c{index}",
                    raw_arguments="{}",
                    raw_name="svc_echo",
                    arguments={"value": f"v{index}"},
                    name="svc_echo",
                )
                for index in range(10)
            ]
            results = await execute_tool_calls_parallel(runtime, blocks, timeout=30)
    assert [result.result for result in results] == [f"v{index}" for index in range(10)]
    assert [result.id for result in results] == [f"c{index}" for index in range(10)]


async def test_teardown_leaves_no_leaked_tasks() -> None:
    """Warm in-memory handles spawn the same client session machinery as
    remotes; after aclose() none of it may survive.  (The remote flavor is
    implicitly gated by every uvicorn test's graceful shutdown — a leaked
    client connection would hang `await serve_task`.)"""
    before = set(asyncio.all_tasks())
    runtime = MCPRuntime(
        [_spec(_named_server("mem", "probe"), namespace="mem")],
        local_server=_local_server(),
    )
    await runtime.list_tools()
    await runtime.call_tool("mem_probe", {})
    assert set(asyncio.all_tasks()) - before
    await runtime.aclose()
    await asyncio.sleep(0.05)
    leaked = set(asyncio.all_tasks()) - before
    assert leaked == set()


class _PostCounter:
    def __init__(self) -> None:
        self.posts = 0

    def wrap(self, app: Any) -> Any:
        async def wrapper(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http" and scope["method"] == "POST":
                self.posts += 1
            await app(scope, receive, send)

        return wrapper


async def test_warm_call_costs_exactly_one_post() -> None:
    """The proxy path paid ~5 HTTP requests per call (fresh client each
    time); a warm runtime call is exactly one POST."""
    counter = _PostCounter()
    async with _ephemeral_http_server(_echo_server(), wrap_asgi=counter.wrap) as url:
        spec = MCPServer(url=url, namespace="svc")._to_spec()
        async with MCPRuntime([spec]) as runtime:
            await runtime.list_tools()
            await runtime.call_tool("svc_echo", {"value": "warmup"})
            before = counter.posts
            await runtime.call_tool("svc_echo", {"value": "warm"})
            after = counter.posts
    assert after - before == 1


# ---------------------------------------------------------------------------
# Identity pool — ServerCredentials, overlays, pool mechanics
# ---------------------------------------------------------------------------


class _StaticAuth(httpx.Auth):
    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request) -> Any:
        request.headers["X-Token"] = self._token
        yield request


class TestServerCredentials:
    def test_same_material_same_key(self) -> None:
        auth = _StaticAuth("t")
        a = ServerCredentials(headers={"X-User": "u1"}, auth=auth)
        b = ServerCredentials(headers={"X-User": "u1"}, auth=auth)
        assert a.identity_key() == b.identity_key()

    def test_header_value_changes_the_key(self) -> None:
        a = ServerCredentials(headers={"X-User": "u1"})
        b = ServerCredentials(headers={"X-User": "u2"})
        assert a.identity_key() != b.identity_key()

    def test_auth_instance_identity_changes_the_key(self) -> None:
        a = ServerCredentials(auth=_StaticAuth("t"))
        b = ServerCredentials(auth=_StaticAuth("t"))
        assert a.identity_key() != b.identity_key()

    def test_explicit_key_wins_over_fingerprint(self) -> None:
        a = ServerCredentials(headers={"X-User": "u1"}, key="tenant-1")
        b = ServerCredentials(headers={"X-User": "other"}, key="tenant-1")
        assert a.identity_key() == b.identity_key() == "key:tenant-1"

    def test_key_never_contains_credential_material(self) -> None:
        cred = ServerCredentials(headers={"X-Secret": "hunter2"})
        assert "hunter2" not in cred.identity_key()


class TestOverlaidSpec:
    def test_headers_merge_overlay_wins(self) -> None:
        base = MCPServer(
            url="https://example.com/mcp",
            namespace="svc",
            headers={"X-Static": "s", "X-User": "base"},
            timeout=17.0,
            elicits=True,
        )._to_spec()
        overlaid = _OverlaidSpec(
            base, ServerCredentials(headers={"X-User": "u1"})
        )
        assert overlaid.headers == {"X-Static": "s", "X-User": "u1"}
        assert overlaid.url == base.url
        assert overlaid.transport == base.transport
        assert overlaid.namespace == "svc"
        assert overlaid.prefix is True
        assert overlaid.timeout == 17.0
        assert overlaid.elicits is True

    def test_overlay_auth_replaces_base_auth(self) -> None:
        base_auth = _StaticAuth("base")
        overlay_auth = _StaticAuth("overlay")
        base = MCPServer(
            url="https://example.com/mcp", namespace="svc", auth=base_auth
        )._to_spec()
        assert _OverlaidSpec(base, ServerCredentials()).auth is base_auth
        assert (
            _OverlaidSpec(base, ServerCredentials(auth=overlay_auth)).auth
            is overlay_auth
        )


class TestIdentityPool:
    async def test_acquire_mints_once_per_identity(self) -> None:
        runtime = MCPRuntime([_spec(_named_server("svc", "ping"), namespace="svc")])
        base = runtime._handles[0]
        cred_a = ServerCredentials(key="a")
        cred_b = ServerCredentials(key="b")

        first = await runtime._acquire_entry(
            base, cred_a.identity_key(), overlay=None, elicitation_handler=None
        )
        second = await runtime._acquire_entry(
            base, cred_a.identity_key(), overlay=None, elicitation_handler=None
        )
        other = await runtime._acquire_entry(
            base, cred_b.identity_key(), overlay=None, elicitation_handler=None
        )
        try:
            assert first is second
            assert other is not first
            counts = runtime.client_counts()
            assert counts.static == 1
            assert counts.identity == 2
            assert counts.conversation == 0
        finally:
            for entry in (first, second, other):
                runtime._release_entry(entry)
            await runtime.aclose()

    async def test_conversation_scoped_entries_counted_separately(self) -> None:
        runtime = MCPRuntime([_spec(_named_server("svc", "ping"), namespace="svc")])
        base = runtime._handles[0]
        entry = await runtime._acquire_entry(
            base,
            "conv:c-1",
            overlay=None,
            elicitation_handler=None,
            conversation_scoped=True,
        )
        try:
            counts = runtime.client_counts()
            assert counts.identity == 0
            assert counts.conversation == 1
        finally:
            runtime._release_entry(entry)
            await runtime.aclose()

    async def test_idle_ttl_evicts_on_next_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "t0_agents.mcp.runtime.time.monotonic", lambda: clock["now"]
        )
        runtime = MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")],
            pool_idle_ttl=10.0,
        )
        base = runtime._handles[0]
        entry = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        runtime._release_entry(entry)
        assert runtime.client_counts().identity == 1

        clock["now"] += 11.0
        fresh = await runtime._acquire_entry(
            base, "key:b", overlay=None, elicitation_handler=None
        )
        try:
            assert runtime.client_counts().identity == 1
            assert ("svc", "key:a") not in runtime._pool
        finally:
            runtime._release_entry(fresh)
            await runtime.aclose()

    async def test_active_entry_survives_ttl_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "t0_agents.mcp.runtime.time.monotonic", lambda: clock["now"]
        )
        runtime = MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")],
            pool_idle_ttl=10.0,
        )
        base = runtime._handles[0]
        held = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        clock["now"] += 11.0
        other = await runtime._acquire_entry(
            base, "key:b", overlay=None, elicitation_handler=None
        )
        try:
            assert ("svc", "key:a") in runtime._pool
        finally:
            runtime._release_entry(held)
            runtime._release_entry(other)
            await runtime.aclose()

    async def test_max_size_evicts_oldest_idle(self) -> None:
        runtime = MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")],
            pool_max_size=1,
        )
        base = runtime._handles[0]
        first = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        runtime._release_entry(first)
        second = await runtime._acquire_entry(
            base, "key:b", overlay=None, elicitation_handler=None
        )
        try:
            assert ("svc", "key:a") not in runtime._pool
            assert ("svc", "key:b") in runtime._pool
        finally:
            runtime._release_entry(second)
            await runtime.aclose()

    async def test_pool_of_active_entries_fails_loud(self) -> None:
        runtime = MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")],
            pool_max_size=1,
        )
        base = runtime._handles[0]
        held = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        try:
            with pytest.raises(ConfigurationError, match="pool_max_size"):
                await runtime._acquire_entry(
                    base, "key:b", overlay=None, elicitation_handler=None
                )
        finally:
            runtime._release_entry(held)
            await runtime.aclose()

    async def test_discard_entry_only_removes_same_object(self) -> None:
        runtime = MCPRuntime([_spec(_named_server("svc", "ping"), namespace="svc")])
        base = runtime._handles[0]
        entry = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        runtime._release_entry(entry)
        runtime._discard_entry(("svc", "key:a"), entry)
        assert ("svc", "key:a") not in runtime._pool

        fresh = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        runtime._release_entry(fresh)
        runtime._discard_entry(("svc", "key:a"), entry)  # stale object
        try:
            assert runtime._pool[("svc", "key:a")] is fresh
        finally:
            await runtime.aclose()

    async def test_overlay_on_in_memory_mount_fails_loud(self) -> None:
        runtime = MCPRuntime([_spec(_named_server("svc", "ping"), namespace="svc")])
        base = runtime._handles[0]
        with pytest.raises(ConfigurationError, match="in-memory"):
            await runtime._acquire_entry(
                base,
                "key:a",
                overlay=ServerCredentials(headers={"X-User": "u"}),
                elicitation_handler=None,
            )
        await runtime.aclose()

    async def test_pooled_calls_work_and_close_without_task_leaks(self) -> None:
        before = set(asyncio.all_tasks())
        runtime = MCPRuntime([_spec(_named_server("svc", "ping"), namespace="svc")])
        base = runtime._handles[0]
        entry = await runtime._acquire_entry(
            base, "key:a", overlay=None, elicitation_handler=None
        )
        try:
            result = await entry.handle.call(
                "ping", {}, timeout=None, progress_handler=None
            )
            assert result.data == "svc:ping"
        finally:
            runtime._release_entry(entry)
            await runtime.aclose()
        leaked = set(asyncio.all_tasks()) - before
        assert leaked == set()

    def test_pool_knob_validation(self) -> None:
        with pytest.raises(ConfigurationError, match="pool_idle_ttl"):
            MCPRuntime([], pool_idle_ttl=0.0)
        with pytest.raises(ConfigurationError, match="pool_max_size"):
            MCPRuntime([], pool_max_size=0)
        with pytest.raises(ConfigurationError, match="tool_list_timeout"):
            MCPRuntime([], tool_list_timeout=0.0)


# ---------------------------------------------------------------------------
# for_conversation views — scope routing, trampoline, fail-loud
# ---------------------------------------------------------------------------


def _approve_handler(name: str, log: list[str]) -> Any:
    async def handler(
        message: str, response_type: Any, params: Any, context: Any
    ) -> Any:
        log.append(name)
        from fastmcp.client.elicitation import ElicitResult

        return ElicitResult(action="accept")

    return handler


class TestConversationView:
    async def test_static_server_rides_the_base_handle(self) -> None:
        async with MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")]
        ) as runtime:
            view = runtime.for_conversation("c-1")
            result = await view.call_tool("svc_ping", {})
            assert result.data == "svc:ping"
            counts = runtime.client_counts()
            assert (counts.identity, counts.conversation) == (0, 0)

    async def test_list_tools_serves_the_shared_inventory(self) -> None:
        async with MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")]
        ) as runtime:
            view = runtime.for_conversation("c-1")
            assert {tool.name for tool in await view.list_tools()} == {
                "svc_ping"
            }
            assert runtime.client_counts().conversation == 0

    async def test_server_for_delegates_to_the_runtime(self) -> None:
        async with MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")]
        ) as runtime:
            view = runtime.for_conversation("c-1")
            assert view.server_for("svc_ping") == "svc"
            assert view.server_for("nope") is None

    async def test_elicit_server_gets_private_per_conversation_clients(
        self,
    ) -> None:
        """Two conversations on one elicits=True server: each elicit lands
        on its own conversation's handler, never the other's."""
        spec = _spec(_elicit_server(), namespace="svc", elicits=True)
        async with MCPRuntime([spec]) as runtime:
            log_a: list[str] = []
            log_b: list[str] = []
            view_a = runtime.for_conversation(
                "conv-a", elicitation_factory=lambda: _approve_handler("a", log_a)
            )
            view_b = runtime.for_conversation(
                "conv-b", elicitation_factory=lambda: _approve_handler("b", log_b)
            )

            results = await asyncio.gather(
                view_a.call_tool("svc_order", {}),
                view_b.call_tool("svc_order", {}),
            )
            assert [r.data for r in results] == ["placed", "placed"]
            assert log_a == ["a"]
            assert log_b == ["b"]
            counts = runtime.client_counts()
            assert counts.conversation == 2
            assert counts.identity == 0

    async def test_same_conversation_reuses_its_elicit_client(self) -> None:
        spec = _spec(_elicit_server(), namespace="svc", elicits=True)
        async with MCPRuntime([spec]) as runtime:
            log: list[str] = []
            view = runtime.for_conversation(
                "conv-a", elicitation_factory=lambda: _approve_handler("a", log)
            )
            await view.call_tool("svc_order", {})
            await view.call_tool("svc_order", {})
            assert runtime.client_counts().conversation == 1
            assert log == ["a", "a"]

    async def test_trampoline_resolves_the_current_handler_per_elicit(
        self,
    ) -> None:
        """The factory is consulted at elicit time — a handler swapped in
        between calls takes effect (never a frozen closure)."""
        spec = _spec(_elicit_server(), namespace="svc", elicits=True)
        async with MCPRuntime([spec]) as runtime:
            log: list[str] = []
            current: dict[str, Any] = {
                "handler": _approve_handler("first", log)
            }
            view = runtime.for_conversation(
                "conv-a", elicitation_factory=lambda: current["handler"]
            )
            await view.call_tool("svc_order", {})
            current["handler"] = _approve_handler("second", log)
            await view.call_tool("svc_order", {})
            assert log == ["first", "second"]

    async def test_elicit_without_handler_fails_loud(self) -> None:
        spec = _spec(_elicit_server(), namespace="svc", elicits=True)
        async with MCPRuntime([spec]) as runtime:
            view = runtime.for_conversation("conv-a")
            with pytest.raises(ToolError):
                await view.call_tool("svc_order", {})

    async def test_surprise_elicit_on_handlerless_pooled_client_fails_loud(
        self,
    ) -> None:
        """An identity-pooled clone carries no elicitation handler, so a
        server that elicits without declaring ``elicits=True`` fails loud
        (the client never advertises the capability) instead of
        misrouting the prompt to another conversation's handler."""
        spec = _spec(_elicit_server(), namespace="svc")  # NOT elicits=True
        async with MCPRuntime([spec]) as runtime:
            base = runtime._handles[0]
            entry = await runtime._acquire_entry(
                base, "key:a", overlay=None, elicitation_handler=None
            )
            try:
                with pytest.raises(ToolError):
                    await entry.handle.call(
                        "order", {}, timeout=None, progress_handler=None
                    )
            finally:
                runtime._release_entry(entry)

    async def test_unknown_overlay_namespace_fails_loud(self) -> None:
        async with MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")]
        ) as runtime:
            with pytest.raises(ConfigurationError, match="unknown server"):
                runtime.for_conversation(
                    "c-1",
                    overlays={"nope": ServerCredentials(headers={"X": "y"})},
                )

    async def test_unknown_tool_error_contract_preserved(self) -> None:
        async with MCPRuntime(
            [_spec(_named_server("svc", "ping"), namespace="svc")]
        ) as runtime:
            view = runtime.for_conversation("c-1")
            with pytest.raises(ToolError, match="Unknown tool: 'nope'"):
                await view.call_tool("nope", {})

    async def test_overlay_headers_reach_the_wire_per_identity(self) -> None:
        """Two conversations with distinct credentials on one mounted
        server: each call carries its own identity header, from two
        distinct pooled clients."""
        seen: list[str] = []

        def capture(app: Any) -> Any:
            async def wrapper(scope: Any, receive: Any, send: Any) -> None:
                if scope["type"] == "http" and scope["method"] == "POST":
                    headers = dict(scope.get("headers") or [])
                    value = headers.get(b"x-user")
                    if value is not None:
                        seen.append(value.decode())
                await app(scope, receive, send)

            return wrapper

        async with _ephemeral_http_server(
            _named_server("rsrv", "hello"), wrap_asgi=capture
        ) as url:
            spec = MCPServer(url=url, namespace="svc")._to_spec()
            async with MCPRuntime([spec]) as runtime:
                view_a = runtime.for_conversation(
                    "c-a",
                    overlays={
                        "svc": ServerCredentials(headers={"X-User": "u-a"})
                    },
                )
                view_b = runtime.for_conversation(
                    "c-b",
                    overlays={
                        "svc": ServerCredentials(headers={"X-User": "u-b"})
                    },
                )
                result_a = await view_a.call_tool("svc_hello", {})
                result_b = await view_b.call_tool("svc_hello", {})
                assert result_a.data == "rsrv:hello"
                assert result_b.data == "rsrv:hello"
                counts = runtime.client_counts()
                assert counts.identity == 2
                assert counts.conversation == 0
        assert "u-a" in seen and "u-b" in seen

    async def test_same_credentials_share_one_pooled_client(self) -> None:
        async with _ephemeral_http_server(_named_server("rsrv", "hello")) as url:
            spec = MCPServer(url=url, namespace="svc")._to_spec()
            async with MCPRuntime([spec]) as runtime:
                cred = ServerCredentials(headers={"X-User": "u-1"})
                view_a = runtime.for_conversation("c-a", overlays={"svc": cred})
                view_b = runtime.for_conversation("c-b", overlays={"svc": cred})
                await view_a.call_tool("svc_hello", {})
                await view_b.call_tool("svc_hello", {})
                assert runtime.client_counts().identity == 1

    async def test_connect_failure_discards_the_pool_entry(self) -> None:
        spec = MCPServer(
            url="http://127.0.0.1:9/mcp", namespace="svc"
        )._to_spec()
        async with MCPRuntime([spec]) as runtime:
            view = runtime.for_conversation(
                "c-a",
                overlays={"svc": ServerCredentials(headers={"X-User": "u"})},
            )
            with pytest.raises(Exception):
                await view.call_tool("svc_hello", {})
            assert runtime.client_counts().identity == 0
