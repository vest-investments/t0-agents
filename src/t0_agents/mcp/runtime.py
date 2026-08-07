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
import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
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

__all__ = ["ClientCounts", "MCPRuntime", "ServerCredentials"]

logger = logging.getLogger(__name__)

_TOOL_LIST_TTL_SECONDS = 60.0

# Per-handle deadline on the list_tools() fan-out.  Deliberately NOT
# derived from ``spec.timeout`` — that is the per-server httpx read
# window, sized for slow tool calls (2400 s where an elicit wait must
# clear), which as an inventory deadline would be no deadline at all.
_TOOL_LIST_TIMEOUT_SECONDS = 10.0

_POOL_IDLE_TTL_SECONDS = 900.0
_POOL_MAX_SIZE = 1024

# The SDK maps a POST rejected with HTTP 404 (dead/rolled session) to a
# JSON-RPC error with this positive literal code — mcp 1.27.0
# streamable_http.py `_send_session_terminated_error`.
_SESSION_TERMINATED_CODE = 32600

# A server whose SDK predates that mapping rejects an unknown session id
# with a bare 400 instead.  Not an SDK contract but an observed server
# behaviour (mcp-customers, staging and prod), and it never reaches the
# JSON-RPC layer — the transport raises for status first, so this shape
# arrives as `httpx.HTTPStatusError`.
_SESSION_REJECTED_STATUS = 400


def _is_session_terminated(error: McpError) -> bool:
    return error.error.code == _SESSION_TERMINATED_CODE


def _is_session_rejected(error: httpx.HTTPStatusError) -> bool:
    return error.response.status_code == _SESSION_REJECTED_STATUS


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

    @property
    def elicits(self) -> bool: ...


@dataclass(frozen=True)
class ServerCredentials:
    """Per-identity credential overlay for one mounted server.

    ``headers`` merge over the spec's static headers (overlay keys win),
    ``auth`` replaces the spec's ``httpx.Auth`` when set.  ``key`` is an
    optional explicit pool key; when omitted the pool fingerprints the
    resolved credential material instead, so two conversations carrying
    the same headers and the same ``auth`` *instance* share one warm
    client, and a rotated token mints a fresh pool entry while the stale
    one idles out.  ``httpx.Auth`` objects are additionally fresh per
    request by construction (``auth_flow`` runs on every request), so an
    auth object that reads a mutable token needs no pool churn at all.
    """

    headers: Mapping[str, str] | None = None
    auth: httpx.Auth | None = None
    key: str | None = None

    def identity_key(self) -> str:
        """Stable pool-key component for this overlay.

        The fingerprint digests header items and the ``auth`` object's
        identity — never returned to logs; the hex digest itself does not
        reveal the credential material.
        """
        if self.key is not None:
            return f"key:{self.key}"
        digest = hashlib.sha256()
        for name, value in sorted((self.headers or {}).items()):
            digest.update(f"{name}\x00{value}\x00".encode())
        if self.auth is not None:
            digest.update(f"auth:{id(self.auth)}".encode())
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class ClientCounts:
    """Snapshot of the runtime's client population, for tests and ops.

    ``static`` counts the base handles (the local ``@app.tool`` server
    plus one per mounted spec); ``identity`` counts pooled per-identity
    entries; ``conversation`` counts per-conversation entries (elicit
    scoping).  The pool-model invariant: ``identity`` grows with
    *identities x overlaid servers*, ``conversation`` only with
    *conversations that touched an ``elicits=True`` server* — never with
    ``conversations x servers``.
    """

    static: int
    identity: int
    conversation: int


class _OverlaidSpec:
    """A spec view merging an identity overlay onto a base spec.

    Presents the :class:`MCPServerSpecLike` shape with ``headers`` merged
    (overlay keys win) and ``auth`` replaced when the overlay carries one;
    everything else — url, transport, namespace, prefix, timeout — comes
    from the base, so per-server transport behaviour (M1's read-timeout
    factory included) is identical across identities.
    """

    def __init__(
        self, base: MCPServerSpecLike, overlay: ServerCredentials
    ) -> None:
        self._base = base
        merged = {**(base.headers or {}), **dict(overlay.headers or {})}
        self._headers: dict[str, str] | None = merged or None
        self._auth = overlay.auth if overlay.auth is not None else base.auth

    @property
    def url(self) -> str | None:
        return self._base.url

    @property
    def server(self) -> FastMCP[Any] | None:
        return self._base.server

    @property
    def transport(self) -> str:
        return self._base.transport

    @property
    def namespace(self) -> str | None:
        return self._base.namespace

    @property
    def headers(self) -> dict[str, str] | None:
        return self._headers

    @property
    def auth(self) -> httpx.Auth | None:
        return self._auth

    @property
    def prefix(self) -> bool:
        return self._base.prefix

    @property
    def timeout(self) -> float | None:
        return self._base.timeout

    @property
    def elicits(self) -> bool:
        return self._base.elicits


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
        spec: MCPServerSpecLike | None = None,
        server: FastMCP[Any] | None = None,
    ) -> None:
        self.key = key
        self.namespace = namespace
        self.prefix = prefix
        self.live_inventory = live_inventory
        # The mount this handle was built from — ``spec`` for MCPServer
        # mounts, ``server`` for the local ``@app.tool`` FastMCP.  The
        # identity pool reads them to clone per-identity /
        # per-conversation siblings of this handle.
        self.spec = spec
        self.server = server
        self._client_factory = client_factory
        self.client: Client[Any] = client_factory(self.invalidate)
        self._lock = asyncio.Lock()
        self._connected = False
        self._generation = 0
        self._stack = AsyncExitStack()
        self._cache: tuple[list[mcp.types.Tool], float] | None = None
        # Most recent successful inventory, deliberately surviving
        # invalidate() — refresh() invalidates every handle before
        # re-listing, which must not destroy the fallback consulted
        # when that re-list fails.
        self._last_known: list[mcp.types.Tool] | None = None

    def invalidate(self) -> None:
        self._cache = None

    def public_name(self, tool_name: str) -> str:
        """The name inference sees — mount-parity with ``namespace=``."""
        if self.prefix and self.namespace is not None:
            return f"{self.namespace}_{tool_name}"
        return tool_name

    async def connect(self) -> None:
        """Enter the client on first use; concurrent callers share one enter.

        An enter that fails — including one *cancelled* mid-handshake by
        a caller's deadline — leaves the client half-entered, and fastmcp
        clients must never be re-entered from that state (the next enter
        wedges, turning a transient outage into a permanent one).  The
        except arm re-mints while still holding the lock, so no queued
        caller can ever observe the tainted client.
        """
        if self._connected:
            return
        async with self._lock:
            if self._connected:
                return
            try:
                await self._stack.enter_async_context(self.client)
            except BaseException:
                self._remint_locked()
                raise
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

        Every successful fetch also lands in ``_last_known``, the
        TTL-exempt fallback :meth:`MCPRuntime.list_tools` serves when
        this handle cannot answer within its deadline.
        """
        return await self._tools_once()

    async def _tools_once(self) -> list[mcp.types.Tool]:
        """Single fetch attempt: connect, then live or TTL-cached inventory."""
        await self.connect()
        if self.live_inventory:
            tools = await self.client.list_tools()
            self._last_known = tools
            return tools
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
            self._last_known = tools
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

        A session-*terminated* error is synthesised per rejected POST and
        carries this request's own id, so the request provably never
        dispatched and one re-send on a rebuilt client is safe even for
        non-idempotent (order-write) tools.

        A session-*rejection* carries no such attribution: the bare 400
        tears the whole transport down, so every call in flight on this
        handle raises it — including one the server already dispatched
        and ran.  The two are indistinguishable from here, so the handle
        is rebuilt (without it the dead client is reused and every later
        call fails too) but the call is never resent.

        Any other failure, including timeouts and mid-call stream deaths,
        propagates: the call MAY have executed.
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
        except httpx.HTTPStatusError as error:
            if not _is_session_rejected(error):
                raise
            logger.warning(
                "MCP session for %r rejected with HTTP %d; rebuilding "
                "client without resending %r",
                self.key,
                error.response.status_code,
                name,
            )
            await self.rebuild_if(generation)
            raise

    def _remint_locked(self) -> tuple[bool, AsyncExitStack]:
        """Swap in a factory-fresh client and stack; caller holds ``_lock``.

        Returns ``(was_connected, stale_stack)`` so a caller that
        replaced a live session can close the old stack outside the lock.
        """
        self._generation += 1
        self._cache = None
        was_connected = self._connected
        self._connected = False
        stale_stack = self._stack
        self._stack = AsyncExitStack()
        self.client = self._client_factory(self.invalidate)
        return was_connected, stale_stack

    async def rebuild_if(self, generation: int) -> None:
        """Replace the client after a session-terminated error.

        Generation-guarded: N concurrent calls hitting the same dead
        session trigger ONE rebuild — later arrivals see the bumped
        generation and reuse the already-fresh client.
        """
        async with self._lock:
            if self._generation != generation:
                return
            was_connected, stale_stack = self._remint_locked()
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


@dataclass
class _PoolEntry:
    """One pooled client: a cloned handle plus its lease bookkeeping.

    ``active`` counts in-flight calls (an entry is only evictable at
    zero — eviction must never close a client under a live call, which
    on the HITL path is a human reading a confirmation prompt);
    ``last_used`` drives idle-TTL eviction; ``conversation_scoped``
    separates elicit-scoping entries from identity entries in
    :meth:`MCPRuntime.client_counts`.
    """

    handle: _ServerHandle
    last_used: float = field(default_factory=time.monotonic)
    active: int = 0
    conversation_scoped: bool = False


class MCPRuntime:
    """Multi-server tool router implementing the ``ToolProvider`` protocol.

    Hosts the App-local ``@app.tool`` server as a routed handle; every
    base handle's client is constructed with the same App-scoped
    ``elicitation_handler``.  Conversation-scoped identity rides a
    side pool of cloned handles keyed ``(server, identity)`` — see
    :meth:`for_conversation`.  Enter the runtime as an async context
    manager (or call :meth:`aclose`) so every warm client is closed
    asynchronously on teardown.

    ``pool_idle_ttl`` / ``pool_max_size`` bound the identity pool:
    entries idle longer than the TTL are evicted lazily on pool access
    (async-closed off the lock), and at the size cap the oldest idle
    entry is evicted to make room — if every entry has a call in
    flight, acquisition fails loud rather than tearing a live client
    down.

    ``tool_list_timeout`` bounds each handle's share of the
    :meth:`list_tools` fan-out; see that method for the degradation
    order when a server cannot answer within it.
    """

    def __init__(
        self,
        specs: Sequence[MCPServerSpecLike] = (),
        *,
        local_server: FastMCP[Any] | None = None,
        elicitation_handler: ElicitationHandler | None = None,
        pool_idle_ttl: float = _POOL_IDLE_TTL_SECONDS,
        pool_max_size: int = _POOL_MAX_SIZE,
        tool_list_timeout: float = _TOOL_LIST_TIMEOUT_SECONDS,
    ) -> None:
        if pool_idle_ttl <= 0:
            raise ConfigurationError(
                f"MCPRuntime(pool_idle_ttl=...) must be positive, "
                f"got {pool_idle_ttl}."
            )
        if pool_max_size < 1:
            raise ConfigurationError(
                f"MCPRuntime(pool_max_size=...) must be >= 1, "
                f"got {pool_max_size}."
            )
        if tool_list_timeout <= 0:
            raise ConfigurationError(
                f"MCPRuntime(tool_list_timeout=...) must be positive, "
                f"got {tool_list_timeout}."
            )
        self._tool_list_timeout = tool_list_timeout
        self._elicitation_handler = elicitation_handler
        self._pool_idle_ttl = pool_idle_ttl
        self._pool_max_size = pool_max_size
        self._pool: dict[tuple[str, str], _PoolEntry] = {}
        self._pool_lock = asyncio.Lock()
        # Evicted handles close in short-lived tracked tasks so a pool
        # access never pays another identity's disconnect latency;
        # :meth:`aclose` awaits any still in flight (the teardown-leak
        # gate contract).
        self._teardown_tasks: set[asyncio.Task[None]] = set()
        self._handles: list[_ServerHandle] = []
        if local_server is not None:
            self._handles.append(
                self._in_memory_handle(
                    "@app.tool",
                    local_server,
                    namespace=None,
                    prefix=False,
                    elicitation_handler=elicitation_handler,
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
        """Close every connected handle (LIFO), aggregating teardown errors.

        Pool entries close before the base handles they were cloned
        from, and any in-flight eviction teardown tasks are awaited so
        no client session task survives the runtime (the teardown-leak
        gate contract).
        """
        pool_entries = list(self._pool.values())
        self._pool.clear()
        async with AsyncExitStack() as stack:
            for handle in reversed(self._handles):
                stack.push_async_callback(handle.aclose)
            for entry in reversed(pool_entries):
                stack.push_async_callback(entry.handle.aclose)
        if self._teardown_tasks:
            await asyncio.gather(
                *self._teardown_tasks, return_exceptions=True
            )

    async def list_tools(self) -> list[mcp.types.Tool]:
        """Merged inventory with namespace prefixes applied runtime-side.

        Naming is identical to the composite path's ``mount(namespace=…)``:
        ``{namespace}_{tool}`` when the spec keeps ``prefix=True`` and has a
        namespace, verbatim otherwise.  A final name provided by two
        handles raises :class:`ConfigurationError` — the composite silently
        forwarded duplicates to inference.

        Each handle gets ``tool_list_timeout`` seconds of the fan-out.
        Handles are process-shared, so one server that cannot answer
        would otherwise hang every conversation's turn; instead it
        degrades: its last-known inventory is served (ignoring the TTL),
        and a handle that has never answered is excluded from this
        merge.  The accepted cost is staleness — a tool added to a
        server while it is unreachable stays invisible until it answers
        again.  An unreachable server never raises; a genuine name
        collision still does.
        """
        per_handle = await asyncio.gather(
            *(
                asyncio.wait_for(handle.tools(), self._tool_list_timeout)
                for handle in self._handles
            ),
            return_exceptions=True,
        )
        merged: list[mcp.types.Tool] = []
        owners: dict[str, str] = {}
        for handle, result in zip(self._handles, per_handle, strict=True):
            if isinstance(result, BaseException):
                fallback = handle._last_known
                if fallback is None:
                    logger.warning(
                        "MCP server %r did not answer list_tools (%r) and "
                        "has no prior inventory; excluding it from this "
                        "merge",
                        handle.key,
                        result,
                    )
                    continue
                logger.warning(
                    "MCP server %r did not answer list_tools (%r); serving "
                    "its last-known inventory (%d tools)",
                    handle.key,
                    result,
                    len(fallback),
                )
                tools = fallback
            else:
                tools = result
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
                key,
                spec.server,
                namespace=spec.namespace,
                prefix=spec.prefix,
                elicitation_handler=self._elicitation_handler,
                spec=spec,
            )
        return self._remote_handle(
            spec.namespace or spec.url or "remote",
            spec,
            elicitation_handler=self._elicitation_handler,
        )

    def _remote_handle(
        self,
        key: str,
        spec: MCPServerSpecLike,
        *,
        elicitation_handler: ElicitationHandler | None,
    ) -> _ServerHandle:
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
            spec=spec,
        )

    def _in_memory_handle(
        self,
        key: str,
        server: FastMCP[Any],
        *,
        namespace: str | None,
        prefix: bool,
        elicitation_handler: ElicitationHandler | None,
        spec: MCPServerSpecLike | None = None,
    ) -> _ServerHandle:
        def factory(_invalidate: Callable[[], None]) -> Client[Any]:
            return Client(server, elicitation_handler=elicitation_handler)

        return _ServerHandle(
            key,
            namespace=namespace,
            prefix=prefix,
            live_inventory=True,
            client_factory=factory,
            spec=spec,
            server=server,
        )

    # ------------------------------------------------------------------
    # Identity pool
    # ------------------------------------------------------------------

    def _clone_handle(
        self,
        base: _ServerHandle,
        *,
        overlay: ServerCredentials | None,
        elicitation_handler: ElicitationHandler | None,
        pool_suffix: str,
    ) -> _ServerHandle:
        """Build a sibling of ``base`` carrying an overlay and/or handler.

        Remote handles ride an :class:`_OverlaidSpec` (headers merged,
        auth swapped) through the same transport construction as the
        base; in-memory handles reconnect to the same ``FastMCP``
        instance.  An overlay on an in-memory mount is a caller bug —
        headers/auth have no transport to ride — and fails loud.
        """
        key = f"{base.key}#{pool_suffix}"
        if base.server is not None:
            if overlay is not None:
                raise ConfigurationError(
                    f"ServerCredentials for {base.key!r}: an in-memory "
                    f"server= mount has no HTTP transport to carry "
                    f"headers/auth."
                )
            return self._in_memory_handle(
                key,
                base.server,
                namespace=base.namespace,
                prefix=base.prefix,
                elicitation_handler=elicitation_handler,
                spec=base.spec,
            )
        assert base.spec is not None, "remote handles always carry their spec"
        spec = (
            _OverlaidSpec(base.spec, overlay) if overlay is not None else base.spec
        )
        return self._remote_handle(
            key, spec, elicitation_handler=elicitation_handler
        )

    async def _acquire_entry(
        self,
        base: _ServerHandle,
        identity_key: str,
        *,
        overlay: ServerCredentials | None,
        elicitation_handler: ElicitationHandler | None,
        conversation_scoped: bool = False,
    ) -> _PoolEntry:
        """Lease the pool entry for ``(base, identity_key)``, minting it
        on first use.

        Pairs with :meth:`_release_entry` (call in a ``finally``).  Every
        acquisition sweeps expired idle entries and enforces the size
        cap; evicted handles close in tracked background tasks so no
        caller pays a stranger's disconnect latency.
        """
        now = time.monotonic()
        key = (base.key, identity_key)
        async with self._pool_lock:
            self._sweep_locked(now)
            entry = self._pool.get(key)
            if entry is None:
                if len(self._pool) >= self._pool_max_size:
                    self._evict_oldest_idle_locked()
                handle = self._clone_handle(
                    base,
                    overlay=overlay,
                    elicitation_handler=elicitation_handler,
                    pool_suffix=identity_key,
                )
                entry = _PoolEntry(
                    handle=handle, conversation_scoped=conversation_scoped
                )
                self._pool[key] = entry
            entry.active += 1
            entry.last_used = now
        return entry

    def _release_entry(self, entry: _PoolEntry) -> None:
        entry.active -= 1
        entry.last_used = time.monotonic()

    def _discard_entry(self, key: tuple[str, str], entry: _PoolEntry) -> None:
        """Drop ``entry`` from the pool (connect-failure eviction).

        Only removes the mapping when it still points at this entry —
        a concurrent re-mint under the same key must not be clobbered.
        The dead handle closes off-path like every eviction.
        """
        if self._pool.get(key) is entry:
            del self._pool[key]
        self._schedule_close(entry.handle)

    def _sweep_locked(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._pool.items()
            if entry.active == 0 and now - entry.last_used > self._pool_idle_ttl
        ]
        for key in expired:
            self._schedule_close(self._pool.pop(key).handle)

    def _evict_oldest_idle_locked(self) -> None:
        idle = [
            (entry.last_used, key)
            for key, entry in self._pool.items()
            if entry.active == 0
        ]
        if not idle:
            raise ConfigurationError(
                f"MCP identity pool is full ({self._pool_max_size} clients, "
                f"all with calls in flight).  Raise "
                f"MCPRuntime(pool_max_size=...) or reduce concurrent "
                f"identities."
            )
        _, oldest = min(idle)
        self._schedule_close(self._pool.pop(oldest).handle)

    def _schedule_close(self, handle: _ServerHandle) -> None:
        """Close an evicted handle in a tracked short-lived task."""

        async def close() -> None:
            try:
                await handle.aclose()
            except Exception:
                logger.debug(
                    "closing evicted pool client %r failed",
                    handle.key,
                    exc_info=True,
                )

        task = asyncio.create_task(close())
        self._teardown_tasks.add(task)
        task.add_done_callback(self._teardown_tasks.discard)

    def client_counts(self) -> ClientCounts:
        """Current client population — the pool-model assertion surface."""
        conversation = sum(
            1 for entry in self._pool.values() if entry.conversation_scoped
        )
        return ClientCounts(
            static=len(self._handles),
            identity=len(self._pool) - conversation,
            conversation=conversation,
        )

    def for_conversation(
        self,
        conversation_key: str,
        *,
        overlays: Mapping[str, ServerCredentials] | None = None,
        elicitation_factory: Callable[[], ElicitationHandler | None]
        | None = None,
    ) -> _ConversationView:
        """A :class:`~t0_agents.core.types.ToolProvider` view binding one
        conversation's identity onto this runtime.

        ``overlays`` maps server **namespaces** to per-conversation
        credentials; a key naming no mounted namespace fails loud here —
        credentials silently not applied would be a security bug, not an
        inconvenience.  ``elicitation_factory`` resolves the
        conversation's *current* elicitation handler at elicit time
        (never a frozen closure), so a consumer can rewire the handler
        between turns.

        Calls route exactly like :meth:`call_tool`, then pick their
        client by scope: an ``elicits=True`` server gets this
        conversation's private client (the only scoping that keeps HITL
        prompts on the right conversation — the SDK discards the
        elicit↔call correlation id); an overlaid server shares the
        identity-pool entry for those credentials, handler-less so a
        surprise elicit fails loud instead of misrouting; everything
        else rides the shared base handle, byte-identical to the static
        path.  ``list_tools`` serves the shared inventory — never
        re-listed per conversation.
        """
        resolved = dict(overlays or {})
        if resolved:
            known = {
                handle.namespace
                for handle in self._handles
                if handle.namespace is not None
            }
            unknown = sorted(set(resolved) - known)
            if unknown:
                raise ConfigurationError(
                    f"ServerCredentials for unknown server namespace(s) "
                    f"{unknown!r} — mounted namespaces: {sorted(known)!r}. "
                    f"Credentials that silently apply to nothing are a "
                    f"security bug, so this fails loud."
                )
        return _ConversationView(
            self,
            conversation_key,
            overlays=resolved,
            elicitation_factory=elicitation_factory,
        )



class _ConversationView:
    """One conversation's :class:`~t0_agents.core.types.ToolProvider` facade.

    Built by :meth:`MCPRuntime.for_conversation`.  Holds no clients of
    its own — every call leases from the runtime's pool (or rides the
    shared base handle), so an abandoned view costs nothing and its
    conversation's clients idle out via the pool TTL.  Implements
    ``SupportsServerAttribution`` by delegation so per-call
    observability keeps working through the view (the M7 contract).
    """

    def __init__(
        self,
        runtime: MCPRuntime,
        conversation_key: str,
        *,
        overlays: dict[str, ServerCredentials],
        elicitation_factory: Callable[[], ElicitationHandler | None] | None,
    ) -> None:
        self._runtime = runtime
        self._conversation_key = conversation_key
        self._overlays = overlays
        self._elicitation_factory = elicitation_factory

    async def list_tools(self) -> list[mcp.types.Tool]:
        return await self._runtime.list_tools()

    def server_for(self, name: str) -> str | None:
        return self._runtime.server_for(name)

    async def _trampoline(
        self,
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> Any:
        """Resolve the conversation's CURRENT handler and delegate.

        Installed on this conversation's private elicit clients.  The
        factory is consulted per elicitation — never captured — so
        rewiring the handler between turns takes effect immediately.  No
        resolvable handler is a loud error: silently declining (or worse,
        falling back to some other conversation's handler) would misroute
        a human-in-the-loop confirmation.
        """
        factory = self._elicitation_factory
        handler = factory() if factory is not None else None
        if handler is None:
            raise RuntimeError(
                f"A tool elicited in conversation "
                f"{self._conversation_key!r} but the conversation has no "
                f"elicitation handler.  Pass elicitation_handler= when "
                f"opening the conversation (or set an App-level handler)."
            )
        return await handler(message, response_type, params, context)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        progress_handler: ToolProgressHandler | None = None,
    ) -> Any:
        routed = await self._runtime._route(name)
        if routed is None:
            raise ToolError(f"Unknown tool: {name!r}")
        handle, bare_name = routed
        overlay = (
            self._overlays.get(handle.namespace)
            if handle.namespace is not None
            else None
        )
        elicits = handle.spec.elicits if handle.spec is not None else False
        if not elicits and overlay is None:
            return await handle.call(
                bare_name,
                arguments,
                timeout=timeout,
                progress_handler=progress_handler,
            )
        if elicits:
            identity_key = f"conv:{self._conversation_key}"
            handler: ElicitationHandler | None = cast(
                "ElicitationHandler", self._trampoline
            )
            conversation_scoped = True
        else:
            assert overlay is not None  # the double-None case returned above
            identity_key = overlay.identity_key()
            handler = None
            conversation_scoped = False
        entry = await self._runtime._acquire_entry(
            handle,
            identity_key,
            overlay=overlay,
            elicitation_handler=handler,
            conversation_scoped=conversation_scoped,
        )
        try:
            try:
                await entry.handle.connect()
            except Exception:
                # A client that cannot even connect (bad credentials, dead
                # endpoint) must not wedge its pool slot — drop the entry
                # so the next call re-resolves and re-mints.
                self._runtime._discard_entry(
                    (handle.key, identity_key), entry
                )
                raise
            return await entry.handle.call(
                bare_name,
                arguments,
                timeout=timeout,
                progress_handler=progress_handler,
            )
        finally:
            self._runtime._release_entry(entry)

