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
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from t0_agents.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from types import TracebackType

    import httpx
    import mcp.types
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core.types import ToolProgressHandler

__all__ = ["MCPRuntime"]


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
        await self.connect()
        return await self.client.list_tools()

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None,
        progress_handler: ToolProgressHandler | None,
    ) -> Any:
        await self.connect()
        return await self.client.call_tool(
            name, arguments, timeout=timeout, progress_handler=progress_handler
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
        for spec in specs:
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
        whether the bare tool exists); unprefixed names route by live
        inventory membership.  An unroutable name raises
        :class:`~fastmcp.exceptions.ToolError` with the same message shape
        the composite produced, keeping the agent loop's error contract
        intact.
        """
        routed = await self._route(name)
        if routed is None:
            raise ToolError(f"Unknown tool: {name!r}")
        handle, bare_name = routed
        return await handle.call(
            bare_name, arguments, timeout=timeout, progress_handler=progress_handler
        )

    async def _route(self, name: str) -> tuple[_ServerHandle, str] | None:
        for namespace, handle in self._prefix_routes:
            if name.startswith(f"{namespace}_"):
                return handle, name[len(namespace) + 1 :]
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
        raise ConfigurationError(
            f"MCPRuntime cannot route url={spec.url!r} yet — remote handles "
            f"land in the next slice."
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
