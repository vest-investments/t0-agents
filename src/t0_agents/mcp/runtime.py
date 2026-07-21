"""Native multi-server MCP router — :class:`MCPRuntime`.

Replaces glow's composite-``FastMCP`` + ``create_proxy`` mount path with
direct, long-lived fastmcp Clients — one warm client per server — behind
the :class:`~t0_agents.core.types.ToolProvider` protocol.  The proxy path
paid a full reconnect (~5 HTTP requests + fresh TCP/TLS) on every remote
tool call because each proxied call ran ``client.new()`` against a
disconnected target; here the handshake happens once per server on first
use and every later call rides the same warm session.

Design notes, verified against installed fastmcp 3.2.3 / mcp SDK 1.27.0:

* fastmcp ``Client`` context managers are reentrant and refcounted, and the
  SDK session multiplexes concurrent requests by request id — one warm
  client per server serves any number of concurrent conversations.
* The server side may reap idle sessions: ``StreamableHTTPSessionManager``
  accepts ``session_idle_timeout`` (seconds; default ``None`` = never,
  incompatible with stateless mode).  A reaped/rolled session surfaces on
  the next call as ``McpError`` with ``error.code == 32600`` ("Session
  terminated") — the SDK maps the transport's HTTP 404 to that error
  *before dispatch*, so the request provably never executed.
* ``tools/list_changed`` notifications ride the server's GET stream, which
  the SDK abandons after 2 failed reconnects; a TTL on any inventory cache
  is the backstop for silently-lost notifications, not a luxury.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx
from fastmcp import Client, FastMCP
from fastmcp.client.tasks import TaskNotificationHandler
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from fastmcp.exceptions import ToolError
from mcp.shared._httpx_utils import McpHttpClientFactory
from mcp.shared.exceptions import McpError

from t0_agents.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from types import TracebackType

    import mcp.types
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core.types import ToolProgressHandler

__all__ = ["MCPRuntime"]

logger = logging.getLogger(__name__)

_TOOL_LIST_TTL_SECONDS = 60.0

# The SDK maps a POST rejected with HTTP 404 (dead/rolled session) to a
# JSON-RPC error with this positive literal code — mcp 1.27.0
# streamable_http.py `_send_session_terminated_error`.
_SESSION_TERMINATED_CODE = 32600


def _is_session_terminated(error: McpError) -> bool:
    return error.error.code == _SESSION_TERMINATED_CODE


class _InventoryMessageHandler(TaskNotificationHandler):
    """Invalidate a handle's inventory cache on ``tools/list_changed``.

    Subclasses ``TaskNotificationHandler`` because fastmcp installs that
    as the *default* message handler (``message_handler or
    TaskNotificationHandler(self)``) — passing a plain handler would
    silently disable task-status routing.  The parent's ``__init__``
    wants the not-yet-constructed client, so :meth:`bind` runs it right
    after ``Client(...)`` and before any traffic can flow.
    """

    def __init__(self, invalidate: Callable[[], None]) -> None:
        self._invalidate = invalidate

    def bind(self, client: Client[Any]) -> None:
        super().__init__(client)

    async def on_tool_list_changed(
        self, message: mcp.types.ToolListChangedNotification
    ) -> None:
        self._invalidate()


class MCPServerSpecLike(Protocol):
    """Structural view of glow's ``MCPServerSpec``.

    ``mcp/`` sits below ``glow/`` in the dependency graph, so the runtime
    reads the spec shape structurally instead of importing it.  Exactly
    one of ``url``/``server`` is set (xor-validated where specs are
    built); ``prefix=True`` applies ``namespace`` as the tool-name prefix.
    """

    @property
    def url(self) -> str | None: ...

    @property
    def server(self) -> FastMCP[Any] | None: ...

    @property
    def transport(self) -> str: ...

    @property
    def namespace(self) -> str | None: ...

    @property
    def headers(self) -> dict[str, str] | None: ...

    @property
    def auth(self) -> httpx.Auth | None: ...

    @property
    def prefix(self) -> bool: ...

    @property
    def timeout(self) -> float | None: ...


def _read_timeout_client_factory(read_timeout: float) -> McpHttpClientFactory:
    """Build an httpx-client factory pinning the read timeout to ``read_timeout``.

    fastmcp calls the factory once per connection and closes the returned
    client itself, so every invocation must mint a fresh
    ``httpx.AsyncClient`` — sharing one would hand fastmcp an
    already-closed client on the next connect.  The 30 s
    connect/write/pool budget mirrors the MCP SDK default; only the read
    gap is overridden.  The ``timeout`` the caller passes (derived from
    the session read timeout, when one is set) is deliberately ignored —
    the spec's per-server value wins.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        kwargs.setdefault("follow_redirects", True)
        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(30.0, read=read_timeout),
            **kwargs,
        )

    return cast(McpHttpClientFactory, factory)


def _remote_transport(
    spec: MCPServerSpecLike,
) -> str | StreamableHttpTransport | SSETransport:
    """Resolve a ``url`` spec to what the direct client should mount.

    A bare URL lets ``Client`` infer the transport (the default path).
    ``headers``/``auth``/``timeout`` cannot ride an inferred transport,
    so when any is set build the explicit transport named by
    ``spec.transport`` (``"http"`` / ``"sse"``) carrying them;
    ``timeout`` travels as an httpx-client factory that pins the read
    window on every connection the client opens.
    """
    url = spec.url
    if url is None:
        raise ConfigurationError("a remote handle requires url=")
    if spec.headers is None and spec.auth is None and spec.timeout is None:
        return url
    factory = (
        _read_timeout_client_factory(spec.timeout)
        if spec.timeout is not None
        else None
    )
    if spec.transport == "sse":
        return SSETransport(
            url, headers=spec.headers, auth=spec.auth, httpx_client_factory=factory
        )
    if spec.transport == "http":
        return StreamableHttpTransport(
            url, headers=spec.headers, auth=spec.auth, httpx_client_factory=factory
        )
    raise ConfigurationError(
        f"MCPServer(url={url!r}): headers=/auth=/timeout= require transport "
        f"'http' or 'sse', got {spec.transport!r}."
    )


class _ServerHandle:
    """One mounted server: a warm client, its inventory, its locks.

    ``client_factory`` receives this handle's cache-invalidation callback
    and returns a fresh, *disconnected* client — called at construction and
    again whenever an eviction rebuilds the client.
    """

    def __init__(
        self,
        key: str,
        *,
        namespace: str | None,
        prefix: bool,
        live_inventory: bool,
        client_factory: Callable[[Callable[[], None]], Client[Any]],
    ) -> None:
        self.key = key
        self.namespace = namespace
        self.prefix = prefix
        self.live_inventory = live_inventory
        self._client_factory = client_factory
        self.client: Client[Any] = client_factory(self.invalidate)
        self._lock = asyncio.Lock()
        self._connected = False
        self._generation = 0
        self._stack = AsyncExitStack()
        self._cache: tuple[list[mcp.types.Tool], float] | None = None

    def invalidate(self) -> None:
        self._cache = None

    def public_name(self, tool_name: str) -> str:
        """The name inference sees — mount-parity with ``namespace=``."""
        if self.prefix and self.namespace is not None:
            return f"{self.namespace}_{tool_name}"
        return tool_name

    async def connect(self) -> None:
        """Enter the client on first use; concurrent callers share one enter."""
        if self._connected:
            return
        async with self._lock:
            if self._connected:
                return
            await self._stack.enter_async_context(self.client)
            self._connected = True

    async def aclose(self) -> None:
        """Async-close the client.  Idempotent; sync close would deadlock."""
        if not self._connected:
            return
        self._connected = False
        await self._stack.aclose()
        self._stack = AsyncExitStack()

    async def tools(self) -> list[mcp.types.Tool]:
        """This handle's inventory — live for in-memory servers, TTL-cached
        for remotes.

        In-memory servers stay live-linked (a ``@server.tool`` registered
        after boot is visible immediately, as it was on the composite).
        Remote inventories cache for ``_TOOL_LIST_TTL_SECONDS``;
        ``tools/list_changed`` invalidates early, and the TTL is the
        backstop for notifications lost to the GET stream's 2-attempt
        reconnect cap.
        """
        await self.connect()
        if self.live_inventory:
            return await self.client.list_tools()
        cached = self._cache
        if cached is not None and time.monotonic() - cached[1] < _TOOL_LIST_TTL_SECONDS:
            return cached[0]
        async with self._lock:
            cached = self._cache
            if (
                cached is not None
                and time.monotonic() - cached[1] < _TOOL_LIST_TTL_SECONDS
            ):
                return cached[0]
            tools = await self.client.list_tools()
            self._cache = (tools, time.monotonic())
            return tools

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None,
        progress_handler: ToolProgressHandler | None,
    ) -> Any:
        """Execute one call, resending it at most once on session loss.

        A session-terminated error means the server rejected the POST at
        the session layer — the request provably never dispatched, so one
        re-send on a rebuilt client is safe even for non-idempotent
        (order-write) tools.  Any other failure, including timeouts and
        mid-call stream deaths, propagates: the call MAY have executed.
        """
        await self.connect()
        generation = self._generation
        try:
            return await self.client.call_tool(
                name, arguments, timeout=timeout, progress_handler=progress_handler
            )
        except McpError as error:
            if not _is_session_terminated(error):
                raise
            logger.warning(
                "MCP session for %r terminated; rebuilding client and "
                "resending %r once",
                self.key,
                name,
            )
            await self.rebuild_if(generation)
            await self.connect()
            return await self.client.call_tool(
                name, arguments, timeout=timeout, progress_handler=progress_handler
            )

    async def rebuild_if(self, generation: int) -> None:
        """Replace the client after a session-terminated error.

        Generation-guarded: N concurrent calls hitting the same dead
        session trigger ONE rebuild — later arrivals see the bumped
        generation and reuse the already-fresh client.
        """
        async with self._lock:
            if self._generation != generation:
                return
            self._generation += 1
            self._cache = None
            was_connected = self._connected
            self._connected = False
            stale_stack = self._stack
            self._stack = AsyncExitStack()
            self.client = self._client_factory(self.invalidate)
        # The stale stack is independent of the rebuilt state — close it
        # outside the lock so a slow dead-session teardown (bounded by the
        # client disconnect timeout) never blocks connects to the fresh
        # client.
        if was_connected:
            try:
                await stale_stack.aclose()
            except Exception:
                logger.debug(
                    "closing the terminated session for %r failed",
                    self.key,
                    exc_info=True,
                )


class MCPRuntime:
    """Multi-server tool router implementing the ``ToolProvider`` protocol.

    Hosts the App-local ``@app.tool`` server as a routed handle; every
    handle's client is constructed with the same App-scoped
    ``elicitation_handler`` (per-conversation scoping is a later
    milestone).  Enter the runtime as an async context manager (or call
    :meth:`aclose`) so every warm client is closed asynchronously on
    teardown.
    """

    def __init__(
        self,
        specs: Sequence[MCPServerSpecLike] = (),
        *,
        local_server: FastMCP[Any] | None = None,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> None:
        self._elicitation_handler = elicitation_handler
        self._handles: list[_ServerHandle] = []
        if local_server is not None:
            self._handles.append(
                self._in_memory_handle(
                    "@app.tool", local_server, namespace=None, prefix=False
                )
            )
        # Value-identical specs fold into one handle — the composite path
        # tolerated re-registration because ``FastMCP.mount`` is idempotent
        # on identical mounts (e.g. ``servers=[GithubMCP]`` plus the same
        # class appearing as a parameter annotation).
        unique_specs: list[MCPServerSpecLike] = []
        for spec in specs:
            if any(spec == seen for seen in unique_specs):
                continue
            unique_specs.append(spec)
        for spec in unique_specs:
            self._handles.append(self._handle_for_spec(spec))
        # Longest namespace first, so a server whose name prefixes another
        # (``fred`` vs ``fred_api``) never steals the longer match — the
        # same rule as ``t0_agents.stream.status.split_tool_identity``.
        self._prefix_routes: list[tuple[str, _ServerHandle]] = sorted(
            (
                (handle.namespace, handle)
                for handle in self._handles
                if handle.prefix and handle.namespace is not None
            ),
            key=lambda route: len(route[0]),
            reverse=True,
        )
        self._unprefixed: list[_ServerHandle] = [
            handle
            for handle in self._handles
            if not (handle.prefix and handle.namespace is not None)
        ]

    async def __aenter__(self) -> MCPRuntime:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close every connected handle (LIFO), aggregating teardown errors."""
        async with AsyncExitStack() as stack:
            for handle in reversed(self._handles):
                stack.push_async_callback(handle.aclose)

    async def list_tools(self) -> list[mcp.types.Tool]:
        """Merged inventory with namespace prefixes applied runtime-side.

        Naming is identical to the composite path's ``mount(namespace=…)``:
        ``{namespace}_{tool}`` when the spec keeps ``prefix=True`` and has a
        namespace, verbatim otherwise.  A final name provided by two
        handles raises :class:`ConfigurationError` — the composite silently
        forwarded duplicates to inference.
        """
        per_handle = await asyncio.gather(
            *(handle.tools() for handle in self._handles)
        )
        merged: list[mcp.types.Tool] = []
        owners: dict[str, str] = {}
        for handle, tools in zip(self._handles, per_handle, strict=True):
            for tool in tools:
                final = handle.public_name(tool.name)
                other = owners.get(final)
                if other is not None:
                    raise ConfigurationError(
                        f"Tool name collision: {final!r} is provided by "
                        f"both {other!r} and {handle.key!r}. Give one of "
                        f"them a distinct namespace= (or rename the tool)."
                    )
                owners[final] = handle.key
                merged.append(
                    tool
                    if final == tool.name
                    else tool.model_copy(update={"name": final})
                )
        return merged

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        progress_handler: ToolProgressHandler | None = None,
    ) -> Any:
        """Route ``name`` to its handle and execute.

        Prefixed names route by longest-namespace decomposition (no
        inventory lookup — the owning server stays the authority on
        whether the bare tool exists); unprefixed names go straight to
        the only unprefixed handle when there is exactly one, and route
        by live inventory membership otherwise.  An unroutable name
        raises :class:`~fastmcp.exceptions.ToolError` with the same
        message shape the composite produced, keeping the agent loop's
        error contract intact.
        """
        routed = await self._route(name)
        if routed is None:
            raise ToolError(f"Unknown tool: {name!r}")
        handle, bare_name = routed
        return await handle.call(
            bare_name, arguments, timeout=timeout, progress_handler=progress_handler
        )

    async def refresh(self) -> list[mcp.types.Tool]:
        """Drop every cached inventory and return the freshly merged list."""
        for handle in self._handles:
            handle.invalidate()
        return await self.list_tools()

    def server_for(self, name: str) -> str | None:
        """Namespace of the server owning ``name`` for observability, or ``None``.

        Longest-prefix match over the prefixed handles (the same rule as
        :meth:`_route`'s prefix arm), returning the owning namespace.
        Unprefixed names — a ``prefix=False`` mount or the local
        ``@app.tool`` server — have no namespace to report and return
        ``None``.  A pure name lookup with no inventory I/O, so it is safe
        to call synchronously while building an event.
        """
        for namespace, _handle in self._prefix_routes:
            if name.startswith(f"{namespace}_"):
                return namespace
        return None

    async def _route(self, name: str) -> tuple[_ServerHandle, str] | None:
        for namespace, handle in self._prefix_routes:
            if name.startswith(f"{namespace}_"):
                return handle, name[len(namespace) + 1 :]
        # A single unprefixed handle (the common case: just the local
        # @app.tool server) receives unprefixed names directly — no
        # per-call inventory lookup; the owning server stays the authority
        # on existence and rejects unknown names with the same ToolError.
        if len(self._unprefixed) == 1:
            return self._unprefixed[0], name
        for handle in self._unprefixed:
            if any(tool.name == name for tool in await handle.tools()):
                return handle, name
        return None

    def _handle_for_spec(self, spec: MCPServerSpecLike) -> _ServerHandle:
        if spec.server is not None:
            key = spec.namespace or f"server:{spec.server.name}"
            return self._in_memory_handle(
                key, spec.server, namespace=spec.namespace, prefix=spec.prefix
            )
        return self._remote_handle(spec.namespace or spec.url or "remote", spec)

    def _remote_handle(self, key: str, spec: MCPServerSpecLike) -> _ServerHandle:
        elicitation_handler = self._elicitation_handler

        def factory(invalidate: Callable[[], None]) -> Client[Any]:
            handler = _InventoryMessageHandler(invalidate)
            client = Client(
                _remote_transport(spec),
                elicitation_handler=elicitation_handler,
                message_handler=handler,
            )
            handler.bind(client)
            return client

        return _ServerHandle(
            key,
            namespace=spec.namespace,
            prefix=spec.prefix,
            live_inventory=False,
            client_factory=factory,
        )

    def _in_memory_handle(
        self,
        key: str,
        server: FastMCP[Any],
        *,
        namespace: str | None,
        prefix: bool,
    ) -> _ServerHandle:
        elicitation_handler = self._elicitation_handler

        def factory(_invalidate: Callable[[], None]) -> Client[Any]:
            return Client(server, elicitation_handler=elicitation_handler)

        return _ServerHandle(
            key,
            namespace=namespace,
            prefix=prefix,
            live_inventory=True,
            client_factory=factory,
        )
