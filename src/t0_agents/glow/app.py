"""``App`` — the registry + lifecycle for ``@app.agent`` decorators.

Pattern lift from FastAPI's ``app = FastAPI()`` followed by
``@app.get(...)``.  ``App`` owns:

- The :class:`AsyncTensorZeroGateway` reference threaded into every
  underlying :class:`~t0_agents.core.agent.Agent`.  Optional: an App
  whose every agent is overridden via :attr:`agent_overrides` needs no
  gateway and may be constructed with ``t0=None``.
- The agent registry (``name -> _AgentSpec``).
- The :attr:`dependency_overrides` mapping consulted on every
  ``Depends(...)`` resolution by :class:`~t0_agents.glow.di.DependencyResolver`.
- The :attr:`agent_overrides` mapping consulted by :meth:`_lookup_or_build`
  before constructing any agent — ``app.agent_overrides["name"] =
  ScriptedAgent(...)`` swaps an agent by name (agent-level mirror of
  ``dependency_overrides``).
- A lazy in-memory ``fastmcp.FastMCP`` server (``_mcp_server``) and
  matching ``fastmcp.Client`` (``_owned_client``).  Only allocated when
  the App needs a real MCP surface: at least one ``@app.tool``
  registration or one :class:`MCPServer` spec.  Apps that only register
  sub-agent orchestrators pay zero FastMCP cost — every glow agent is
  constructed with ``tool_provider=None`` and the v1.0 core auto-allocates
  a :class:`SubAgentToolProvider` for any agent with ``sub_agents=[...]``.
- The lazy build path: each ``_AgentSpec`` materialises its underlying
  :class:`Agent` on first ``await wrapper(msg)`` *or* eagerly via
  ``await app.build()``.  ``app.lifespan()`` is the recommended entry
  point — it enters the owned ``Client`` once and exits it cleanly on
  teardown.

``App`` does **not** modify any v1.0 primitive — every
``@app.agent`` decorator compiles to a normal :class:`Agent` that
exposes itself on ``wrapper._agent`` for escape-hatch use.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

from fastmcp import Client, FastMCP
from fastmcp.client.transports import SSETransport, StreamableHttpTransport
from fastmcp.server import create_proxy
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from t0_agents.glow._spec import _AgentSpec
from t0_agents.glow.di import DependencyResolver
from t0_agents.glow.handle import FluentRun
from t0_agents.glow.mcp_typing import MCPServer, MCPServerSpec
from t0_agents.memory.tensorzero import TensorZeroMemory
from t0_agents.tooling import FilteredToolProvider

if TYPE_CHECKING:
    from types import TracebackType

    from fastmcp.client.elicitation import ElicitationHandler
    from tensorzero import ToolChoice

    from t0_agents.core.conversation import ConversationIndex, ConversationSummary
    from t0_agents.core.middleware.base import Middleware
    from t0_agents.core.types import AgentResult, MemoryProvider
    from t0_agents.glow.decorators import _AgentWrapper

__all__ = ["App"]

logger = logging.getLogger(__name__)

# Sentinel telling ``@app.agent`` "caller omitted this knob" apart from an
# explicit ``None`` — which is a meaningful value for e.g. ``tool_choice`` /
# ``tags`` / ``middleware``.  Only knobs the caller actually passes are
# forwarded into ``AgentConfig``; the rest keep their declarative defaults.
_UNSET: Any = object()

# Method-local generics used by ``App.agent`` so static checkers can
# unify the decorated function's parameters / return type with the
# returned ``_AgentWrapper[P, R]`` — the FastAPI pattern.
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _describe_tool_entry(entry: Any) -> str:
    """Short label for a ``tools=`` entry used in warnings and errors."""
    if isinstance(entry, type):
        return entry.__name__
    if isinstance(entry, MCPServer):
        return f"{type(entry).__name__} instance"
    return getattr(entry, "__name__", repr(entry))


def _mcp_server_namespace(agent_name: str, entry: Any) -> str | None:
    """Prefix namespace of an :class:`MCPServer` ``tools=`` entry, else ``None``.

    Raises :class:`ConfigurationError` when the entry is a real ``MCPServer``
    whose mounted tools carry no prefix to filter on — either a
    ``prefix=False`` mount (unprefixed by choice) or a ``namespace=None``
    mount.  Such tools are selected by explicit name / glob string in
    ``tools=`` rather than by passing the server.
    """
    if isinstance(entry, MCPServer):
        spec: MCPServerSpec | None = entry._to_spec()
    elif isinstance(entry, type) and issubclass(entry, MCPServer):
        spec = getattr(entry, "_mcp_spec", None)
    else:
        return None
    if spec is None:  # the bare ``MCPServer`` base — not a real server
        return None
    if not spec.prefix:
        raise ConfigurationError(
            f"@app.agent({agent_name!r}) tools= includes MCPServer "
            f"{_describe_tool_entry(entry)} mounted with prefix=False — its "
            f"tool names are unprefixed, so select them by explicit name or "
            f"glob string in tools=, not by passing the server."
        )
    if spec.namespace is None:
        raise ConfigurationError(
            f"@app.agent({agent_name!r}) tools= includes MCPServer "
            f"{_describe_tool_entry(entry)} without namespace= — a namespace "
            f"is required to filter its tools (its mounted tool names are "
            f"prefixed with the namespace)."
        )
    return spec.namespace


def _match_tool_entry(
    agent_name: str, entry: Any, live_names: set[str]
) -> set[str]:
    """Resolve one ``tools=`` entry to the live tool names it selects."""
    namespace = _mcp_server_namespace(agent_name, entry)
    if namespace is not None:
        prefix = f"{namespace}_"
        return {name for name in live_names if name.startswith(prefix)}
    if isinstance(entry, str):
        return {name for name in live_names if fnmatch.fnmatchcase(name, entry)}
    if callable(entry) and not isinstance(entry, type):
        name = getattr(entry, "__name__", None)
        return {name} & live_names if name is not None else set()
    raise ConfigurationError(
        f"@app.agent({agent_name!r}) tools= entry {entry!r} is not a "
        f"supported spec — pass an @app.tool callable, an MCPServer "
        f"class/instance, or a glob string."
    )


def _resolve_tool_filter(
    agent_name: str, tools: list[Any], live_names: set[str]
) -> set[str]:
    """Resolve a ``@app.agent(tools=[...])`` spec to a set of live tool names.

    Each entry is matched against ``live_names``: an ``@app.tool`` callable by
    its ``__name__``; an :class:`MCPServer` by its ``namespace`` prefix
    (required); a glob string by :func:`fnmatch.fnmatchcase`.  An entry
    matching nothing emits a :class:`UserWarning` naming the available tools
    (the dead-pattern diagnostic).  An empty ``tools`` yields an empty set —
    an intentionally toolless agent, with no warning.
    """
    allow: set[str] = set()
    for entry in tools:
        matched = _match_tool_entry(agent_name, entry, live_names)
        if not matched:
            warnings.warn(
                f"@app.agent({agent_name!r}) tools= entry "
                f"{_describe_tool_entry(entry)!r} matched no available tool; "
                f"available: {sorted(live_names)!r}",
                stacklevel=2,
            )
        allow |= matched
    return allow


def _remote_target(spec: MCPServerSpec) -> str | StreamableHttpTransport | SSETransport:
    """Resolve a ``url`` spec to what :func:`create_proxy` should wrap.

    A bare URL lets ``create_proxy`` infer the transport (the default, and
    the no-auth path).  ``headers``/``auth`` cannot ride an inferred
    transport, so when either is set build the explicit transport named by
    ``spec.transport`` (``"http"`` / ``"sse"``) carrying them.
    """
    url = cast(str, spec.url)  # xor-validated: non-None whenever server is None
    if spec.headers is None and spec.auth is None:
        return url
    if spec.transport == "sse":
        return SSETransport(url, headers=spec.headers, auth=spec.auth)
    if spec.transport == "http":
        return StreamableHttpTransport(url, headers=spec.headers, auth=spec.auth)
    raise ConfigurationError(
        f"MCPServer(url={url!r}): headers=/auth= require transport 'http' "
        f"or 'sse', got {spec.transport!r}."
    )


class App:
    """Glow application — registry + lifecycle for ``@app.agent`` decorators.

    Construct once near the entry point (typically inside ``async def
    main()`` after the :class:`AsyncTensorZeroGateway` is created) and
    use ``@app.agent("name")`` to register agents declaratively::

        async def main() -> None:
            async with await build_t0() as t0:
                app = App(t0=t0)

                @app.agent("tutor")
                async def tutor(q: str) -> str:
                    \"\"\"An English tutor.\"\"\"

                async with app.lifespan():
                    print(await tutor("Say hi formally in Japanese."))

    Agents are built lazily on first invocation, but ``app.lifespan()``
    (or ``await app.build()`` / ``await app.shutdown()``) is required
    in any deployment that uses ``@app.tool`` or talks to a real
    TensorZero gateway — those paths call into the App-owned FastMCP
    ``Client``, which is connected by :meth:`build` and disconnected by
    :meth:`shutdown`.

    App-level ``tags`` / ``middleware`` / ``namespace`` set defaults for
    every registered agent; per-agent ``@app.agent(...)`` knobs override
    them (``tags`` dict-merge, agent-level keys winning).

    ``elicitation_handler`` binds a human-in-the-loop resolver onto the
    App-owned ``Client``, so a mounted tool's ``ctx.elicit(...)`` reaches a
    real resolver instead of auto-declining.  A ``fastmcp.Client`` binds
    exactly one handler, and the App owns exactly one Client — so this is one
    resolver per App, which fits the per-tenant "one App per session/drive"
    model; multi-tenant HITL uses one App (hence one handler) per tenant.
    """

    def __init__(
        self,
        *,
        t0: AsyncTensorZeroGateway | None = None,
        gateway: str | None = None,
        embedded: str | None = None,
        servers: list[type[MCPServer] | MCPServer] | None = None,
        tags: dict[str, str] | None = None,
        middleware: list[Middleware] | None = None,
        namespace: str | None = None,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> None:
        # Gateway sources — at most one of the three constructor forms:
        #   t0=        an already-built AsyncTensorZeroGateway (caller owns it)
        #   gateway=   an HTTP gateway URL     -> build_http, App-owned
        #   embedded=  a tensorzero.toml path  -> build_embedded, App-owned
        # None of the three defers resolution to the first build():
        # TENSORZERO_URL -> TENSORZERO_CONFIG -> clear error (but only when
        # some agent actually needs a gateway — fully-overridden Apps never
        # resolve).  See :meth:`_ensure_t0`.
        provided = [
            name
            for name, value in (("t0", t0), ("gateway", gateway), ("embedded", embedded))
            if value is not None
        ]
        if len(provided) > 1:
            raise ConfigurationError(
                f"App(...): pass at most one of t0=, gateway=, embedded= — "
                f"got {' and '.join(provided)}."
            )
        self._t0: AsyncTensorZeroGateway | None = t0
        self._gateway_source: tuple[Literal["http", "embedded"], str] | None = (
            ("http", gateway)
            if gateway is not None
            else ("embedded", embedded)
            if embedded is not None
            else None
        )
        # True when the App resolved (and therefore owns) the gateway —
        # owned gateways ride ``_exit_stack`` and are closed by
        # :meth:`shutdown`; a caller-supplied ``t0=`` is never entered
        # or closed by the App.
        self._owns_t0: bool = False
        # App-level ``AgentConfig`` defaults, layered *under* per-agent
        # ``@app.agent(...)`` knobs by :meth:`_resolve_config_overrides`.
        # ``tags`` dict-merge with agent-level keys winning; ``namespace``
        # and ``middleware`` are taken agent-level-first, else the default.
        self._default_tags: dict[str, str] | None = tags
        self._default_middleware: list[Middleware] | None = middleware
        self._default_namespace: str | None = namespace
        self._agent_registry: dict[str, _AgentSpec] = {}
        # Public mapping consulted by ``DependencyResolver`` on every
        # ``Depends(...)`` resolution.  Lives here so users can install
        # overrides without raising ``AttributeError`` at any point in
        # the lifecycle.
        self.dependency_overrides: dict[Callable[..., Any], Callable[..., Any]] = {}
        # Agent-level mirror of ``dependency_overrides``: swap an agent by
        # name before it is built.  ``_lookup_or_build`` consults this first,
        # so a fully-overridden App never touches the gateway (``t0=None`` ok).
        self.agent_overrides: dict[str, Agent] = {}

        # App-owned FastMCP server + matching Client, lazily allocated.
        # ``@app.tool`` and :class:`MCPServer` mounts go through
        # :meth:`_ensure_mcp_server`, which creates the pair on first
        # use.  Apps without ``@app.tool`` or ``MCPServer`` specs (the
        # sub-agents-only orchestration case) leave both ``None`` —
        # ``_lookup_or_build`` then passes ``tool_provider=None`` to
        # :class:`Agent`, and the v1.0 core auto-allocates a
        # :class:`SubAgentToolProvider` for the delegation surface.
        self._mcp_server: FastMCP[Any] | None = None
        self._owned_client: Client[Any] | None = None
        # Bound onto ``_owned_client`` at allocation (:meth:`_ensure_mcp_server`)
        # so a mounted tool's ``ctx.elicit(...)`` round-trips to a real
        # resolver instead of auto-declining.  ``None`` keeps the auto-decline
        # default.  One handler per Client — see the class docstring.
        self._elicitation_handler: ElicitationHandler | None = elicitation_handler
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_entered: bool = False
        # Auto-build latch: True after a successful :meth:`build`, reset
        # by :meth:`shutdown`.  Consumption goes through
        # :meth:`_ensure_ready_for`, which triggers one full build (with
        # an INFO log) when the App was never explicitly booted.
        self._started: bool = False
        # Serializes the boot sequence (gateway resolution, MCP mounts,
        # client entry) against concurrent first consumptions — e.g.
        # ``asyncio.gather`` over several agents of an unstarted App.
        # ``asyncio.Lock()`` is loop-agnostic at construction on
        # Python ≥ 3.10, so module-level Apps are safe.
        self._boot_lock: asyncio.Lock = asyncio.Lock()

        # The :class:`DependencyResolver` instance lives here so it can
        # consult ``self.dependency_overrides`` directly during
        # ``@app.tool`` injections.  ``DependencyResolver`` is stateless
        # beyond the per-call cache it threads through recursive
        # resolution, so a single instance per ``App`` is safe.
        self._resolver: DependencyResolver = DependencyResolver()

        # Typed-MCP integration.  Specs discovered via the ``servers=``
        # kwarg are seeded first; ``@app.agent`` registration appends more
        # specs harvested from parameter annotations (``Annotated`` and
        # subclass forms).  Each spec is mounted onto ``_mcp_server`` in
        # :meth:`build` BEFORE entering ``_owned_client``, so mounted
        # tools appear on the first ``list_tools()`` call.  The list is
        # not deduplicated here because the same :class:`MCPServerSpec`
        # instance could legitimately appear from multiple sources
        # (e.g. ``servers=[GithubMCP]`` plus a decorator-side reference
        # to the same class) — ``FastMCP.mount`` is idempotent on
        # identical mounts and namespacing keeps tool names collision-free.
        self._mcp_specs: list[MCPServerSpec] = []
        for entry in servers or []:
            if isinstance(entry, MCPServer):
                # An instance carries per-mount config (e.g. url + headers/
                # auth) without a module-level subclass — the shape the
                # authed-remote-mount use case needs.
                self._mcp_specs.append(entry._to_spec())
            elif isinstance(entry, type) and issubclass(entry, MCPServer):
                # ``_mcp_spec`` is set by ``__init_subclass__``; the bare
                # ``MCPServer`` base has none — reject it with a clear error
                # instead of an AttributeError.
                spec: MCPServerSpec | None = getattr(entry, "_mcp_spec", None)
                if spec is None:
                    raise ConfigurationError(
                        "App(servers=...): the bare MCPServer base is not a "
                        "server; pass a subclass or an instance."
                    )
                self._mcp_specs.append(spec)
            else:
                raise ConfigurationError(
                    f"App(servers=...): every entry must be an MCPServer "
                    f"subclass or instance; got {entry!r}."
                )

    # ------------------------------------------------------------------
    # Decorator surface — delegates to ``glow.decorators._make_decorator``
    # ------------------------------------------------------------------

    def agent(
        self,
        name: str,
        *,
        function_name: str | None = None,
        model: str | None = _UNSET,
        variant: str | None = _UNSET,
        params: dict[str, Any] | None = _UNSET,
        cache: dict[str, Any] | None = _UNSET,
        max_turns: int = _UNSET,
        tool_timeout: float | None = _UNSET,
        parallel_tools: bool = _UNSET,
        dryrun: bool = _UNSET,
        tool_choice: ToolChoice | None = _UNSET,
        tags: dict[str, str] | None = _UNSET,
        namespace: str | None = _UNSET,
        memory: MemoryProvider | None = _UNSET,
        memory_strategy: Literal[
            "auto", "system_template", "prepended_message", "messages"
        ] = _UNSET,
        middleware: list[Middleware] | None = _UNSET,
        tools: list[Any] | None = None,
        replace: bool = False,
    ) -> Callable[[Callable[_P, Awaitable[_R]]], _AgentWrapper[_P, _R]]:
        """Return the ``@app.agent("name")`` decorator factory.

        Preserves the decorated function's parameters via ``ParamSpec``
        and its return annotation via ``TypeVar`` so static checkers
        see ``tutor("hi")`` typed as ``FluentRun[str]`` (and
        ``await tutor("hi")`` typed as ``str``) when the user writes
        ``async def tutor(q: str) -> str``.  Construction logic lives
        in :mod:`t0_agents.glow.decorators`; this method is the typed
        thin shim that delegates to ``_make_decorator``.

        Every keyword beyond ``name``/``function_name``/``tools`` maps
        1:1 onto the corresponding
        :class:`~t0_agents.core.config.AgentConfig` field (``model``,
        ``variant``, ``params``, ``cache``, ``max_turns``,
        ``tool_timeout``, ``parallel_tools``, ``dryrun``,
        ``tool_choice``, ``tags``, ``namespace``, ``memory``,
        ``memory_strategy``, ``middleware``).  Omitted knobs keep their
        ``AgentConfig`` defaults; supplied knobs are layered over any
        App-level defaults (agent-level wins — see :meth:`__init__`).

        ``model=`` declares the agent with zero tensorzero.toml —
        inference routes through the gateway's built-in passthrough and
        the docstring rides along as the system prompt.
        ``AgentConfig`` enforces the routing rules at build time:
        ``model`` XOR ``function_name`` (neither → ``"agent"``),
        ``variant`` only on the function path, and JSON-function output
        modes (a structured return annotation) are rejected on the
        ``model=`` path with a ready-to-paste TOML stanza.

        ``tools`` filters which of the App's composed tools this agent
        sees: a list of ``@app.tool`` callables, :class:`MCPServer`
        classes/instances (matched by their ``namespace`` prefix), and
        glob strings (e.g. ``"customers_get_*"``).  Omitted (``None``)
        keeps the back-compat behaviour — the agent sees every tool;
        ``tools=[]`` is explicitly toolless.  Patterns matching no live
        tool warn at :meth:`build`.  Sub-agent ``<name>_ask`` tools are
        never filtered.

        ``replace=True`` re-registers over an existing name instead of
        raising the duplicate-name error — the REPL/notebook
        re-decoration affordance (pairs with :meth:`remove_agent`).
        """
        # Imported lazily to break the import cycle between
        # ``glow.app`` and ``glow.decorators`` (the latter imports
        # :class:`App` for type purposes).
        from t0_agents.glow.decorators import _make_decorator

        knobs: dict[str, Any] = {
            "model": model,
            "variant": variant,
            "params": params,
            "cache": cache,
            "max_turns": max_turns,
            "tool_timeout": tool_timeout,
            "parallel_tools": parallel_tools,
            "dryrun": dryrun,
            "tool_choice": tool_choice,
            "tags": tags,
            "namespace": namespace,
            "memory": memory,
            "memory_strategy": memory_strategy,
            "middleware": middleware,
        }
        config_overrides = {k: v for k, v in knobs.items() if v is not _UNSET}

        return _make_decorator(
            self,
            name,
            function_name=function_name,
            config_overrides=config_overrides,
            tools=tools,
            replace=replace,
        )

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register ``fn`` as a FastMCP tool on the App-owned server.

        Parameters whose default is a :func:`~t0_agents.glow.di.Depends`
        sentinel are stripped from the tool's auto-generated JSON
        Schema and re-injected at call time via the App's
        :class:`~t0_agents.glow.di.DependencyResolver`.  The tool's
        docstring becomes the MCP tool description (FastMCP convention).

        Returns the original function unchanged so callers can still
        invoke it directly outside the MCP loop — though direct calls
        will not resolve ``Depends`` parameters; the resolution happens
        only through the MCP transport.

        Construction logic lives in :mod:`t0_agents.glow.decorators`;
        this method is the typed thin shim that delegates.
        """
        from t0_agents.glow.decorators import _make_tool_decorator

        return _make_tool_decorator(self)(fn)

    def remove_agent(self, name: str) -> None:
        """Drop the agent registered under ``name`` — the REPL affordance.

        Frees the name for a fresh ``@app.agent(name)`` registration
        (pairs with ``@app.agent(name, replace=True)``).  Wrapper
        objects from the removed registration keep working — they hold
        their own spec; only the registry entry is detached.  Raises
        :class:`ConfigurationError` when the name is not registered.
        """
        if name not in self._agent_registry:
            raise ConfigurationError(
                f"App.remove_agent({name!r}): no agent is registered under "
                f"this name.  Registered: {sorted(self._agent_registry)!r}."
            )
        del self._agent_registry[name]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_mcp_server(self) -> FastMCP[Any]:
        """Lazily allocate the App-owned FastMCP server + Client.

        Called from :meth:`tool` (the ``@app.tool`` decorator) at
        decoration time and from :meth:`build` when there are
        :class:`MCPServer` specs to mount.  Idempotent — subsequent
        calls reuse the already-allocated pair.
        """
        if self._mcp_server is None:
            self._mcp_server = FastMCP("glow-app")
            self._owned_client = Client(
                self._mcp_server, elicitation_handler=self._elicitation_handler
            )
        return self._mcp_server

    def _needs_gateway(self) -> bool:
        """True when some registered agent will require a real gateway.

        Mirrors :meth:`_lookup_or_build`'s short-circuits: an agent whose
        name has an ``agent_overrides`` entry, or whose spec already
        carries a built :class:`Agent`, never touches the gateway.  Used
        by :meth:`build` so a deferred-source App (bare ``App()``) only
        resolves — and only *errors* on unresolvable — when a gateway is
        genuinely needed.
        """
        return any(
            spec.agent is None and spec.name not in self.agent_overrides
            for spec in self._agent_registry.values()
        )

    async def _ensure_t0(self) -> AsyncTensorZeroGateway:
        """Return the gateway, resolving and taking ownership on first need.

        Resolution order: an explicit ``t0=`` (never entered or closed by
        the App) → the constructor source (``gateway=`` → ``build_http``,
        ``embedded=`` → ``build_embedded``) → the environment
        (``TENSORZERO_URL`` → ``TENSORZERO_CONFIG``) → a
        :class:`ConfigurationError` naming every option.  Resolved
        gateways are entered on the App's exit stack so
        :meth:`shutdown` closes them; ``self._t0`` is reset afterwards so
        a later :meth:`build` re-resolves cleanly.

        The SDK's builders return ``Union[Awaitable[gateway], gateway]``
        (keyed off ``async_setup``, default True) — the cast below is the
        single place glow absorbs that union.
        """
        if self._t0 is not None:
            return self._t0
        source: tuple[Literal["http", "embedded"], str]
        if self._gateway_source is not None:
            source = self._gateway_source
        else:
            url = os.environ.get("TENSORZERO_URL")
            config = os.environ.get("TENSORZERO_CONFIG")
            if url:
                source = ("http", url)
            elif config:
                source = ("embedded", config)
            else:
                raise ConfigurationError(
                    "App has no gateway to build agents with: pass t0= (an "
                    "AsyncTensorZeroGateway), gateway= (an HTTP gateway URL), "
                    "or embedded= (a tensorzero.toml path) to App(...), or "
                    "set TENSORZERO_URL / TENSORZERO_CONFIG in the "
                    "environment."
                )
        kind, value = source
        if kind == "http":
            awaitable = cast(
                "Awaitable[AsyncTensorZeroGateway]",
                AsyncTensorZeroGateway.build_http(gateway_url=value),
            )
        else:
            awaitable = cast(
                "Awaitable[AsyncTensorZeroGateway]",
                AsyncTensorZeroGateway.build_embedded(config_file=value),
            )
        gw = await awaitable
        await self._exit_stack.enter_async_context(gw)
        self._t0 = gw
        self._owns_t0 = True
        return gw

    def _resolve_config_overrides(self, spec: _AgentSpec) -> dict[str, Any]:
        """Layer App-level defaults under the agent-level ``config_overrides``.

        ``namespace`` and ``middleware`` take the agent-level value when the
        agent supplied one, else the App default.  ``tags`` dict-merge so
        ``App(tags={"env": "prod"})`` + ``@app.agent(tags={"team": "cs"})``
        yields both, with agent-level keys winning on collision (mirrors the
        core ``Agent._merge_tags`` precedence).  An explicit agent-level
        ``tags=None`` opts out of the App default (distinct from omission).
        """
        overrides: dict[str, Any] = dict(spec.config_overrides)
        if "namespace" not in overrides and self._default_namespace is not None:
            overrides["namespace"] = self._default_namespace
        if "middleware" not in overrides and self._default_middleware is not None:
            overrides["middleware"] = self._default_middleware
        # ``tags`` dict-merge (agent keys win), but an explicit agent-level
        # ``tags=None`` opts out of the App default entirely — an explicit
        # None stays distinct from omission (the ``_UNSET`` contract).
        if "tags" not in overrides:
            if self._default_tags is not None:
                overrides["tags"] = dict(self._default_tags)
        elif overrides["tags"] is not None and self._default_tags is not None:
            overrides["tags"] = {**self._default_tags, **overrides["tags"]}
        return overrides

    def _lookup_or_build(self, spec: _AgentSpec) -> Agent:
        """Return the underlying :class:`Agent`, constructing it if needed.

        Sync — :class:`Agent.__init__` is sync.  ``async`` is reserved
        for :meth:`build` so the MCP-mount machinery can interleave
        ``await FastMCP.mount(...)`` cleanly.

        When ``spec.sub_agent_specs`` is non-empty, depth-first
        constructs each sub-agent first so the parent's
        :class:`AgentConfig` can be assembled with the fully-built
        :class:`Agent` references.  The recursion is finite because
        the decorator-time check enforces sub-agents register BEFORE
        their parents.
        """
        override = self.agent_overrides.get(spec.name)
        if override is not None:
            return override
        if spec.agent is not None:
            return spec.agent
        # Fail fast before building sub-agents: a non-overridden agent needs
        # a real gateway.  (An override or a fully-overridden App returns above.)
        t0 = self._t0
        if t0 is None:
            raise ConfigurationError(
                f"App cannot build agent {spec.name!r}: no gateway is "
                f"available.  Pass t0=, gateway=, or embedded= to App(...), "
                f"set TENSORZERO_URL / TENSORZERO_CONFIG in the environment, "
                f"or register an override for every agent via "
                f"app.agent_overrides."
            )
        sub_agents: list[Agent] = [
            self._lookup_or_build(sub_spec) for sub_spec in spec.sub_agent_specs
        ]
        config_kwargs: dict[str, Any] = {
            "name": spec.name,
            "function_name": spec.function_name,
            "output": spec.output_mode,
        }
        if sub_agents:
            config_kwargs["sub_agents"] = sub_agents
        if spec.guardrails:
            # Feed harvested ``Annotated[T, Guardrail(...)]`` metadata
            # into the canonical ``AgentConfig.guardrails`` field so
            # ``GuardrailsMiddleware`` picks them up at runtime without
            # further glow plumbing.
            config_kwargs["guardrails"] = spec.guardrails
        # Remaining ``AgentConfig`` knobs passed to ``@app.agent`` (never
        # ``name``/``function_name``/``output``/``sub_agents``/``guardrails``,
        # which are managed above), layered over App-level defaults, flow
        # through verbatim.  ``AgentConfig`` validates them in
        # ``__post_init__``.
        config_kwargs.update(self._resolve_config_overrides(spec))
        # tool_provider is the App-owned client when at least one tool
        # or MCP server is configured; otherwise None lets Agent.__init__
        # auto-allocate a SubAgentToolProvider for the sub-agents-only
        # case (or run toolless for the chat-only case).
        tool_provider: ToolProvider | None = (
            cast(ToolProvider, self._owned_client)
            if self._owned_client is not None
            else None
        )
        # Per-agent ``tools=`` filter (resolved against the live tool list in
        # :meth:`build`).  Wrap the real-tool provider so only allow-listed
        # names reach inference.  ``Agent.__init__`` then layers a
        # ``SubAgentToolProvider`` on top, so sub-agent ``<name>_ask`` tools
        # are appended *after* the filter and remain visible.
        if spec.resolved_tools is not None and tool_provider is not None:
            tool_provider = FilteredToolProvider(tool_provider, spec.resolved_tools)
        agent = Agent(
            AgentConfig(**config_kwargs),
            t0=t0,
            tool_provider=tool_provider,
        )
        spec.agent = agent
        return agent

    def _agent_invoke(
        self,
        agent: Agent,
        message: str,
        mode: Any,
        call_kwargs: dict[str, Any],
        *,
        system_prompt: str | None,
        spec_name: str,
    ) -> Any:
        """Resolve the templating triangle, then fire ``Agent.invoke``.

        The shared body of every glow build closure: ``system_arguments``
        (server-side templates), a raw ``system`` prompt, and ``prompt_vars``
        (client-side ``str.format`` on the effective system string) are
        mutually constrained in the TensorZero gateway, so exactly one path
        through them is legal.  Extracted from :meth:`_invoke` so the plain
        call path and the conversation send path (:meth:`_invoke_in_conversation`)
        apply identical semantics.  ``call_kwargs`` is mutated (``prompt_vars``
        and ``system`` are popped) — the caller passes the closure's own dict.
        """
        prompt_vars = call_kwargs.pop("prompt_vars", None)
        if call_kwargs.get("system_arguments") is not None:
            if call_kwargs.get("system") is not None:
                raise ConfigurationError(
                    "FluentRun.with_(): pass either system_arguments= "
                    "(server-side template variables) or system= (a raw "
                    "system prompt), not both — they are mutually "
                    "exclusive in the TensorZero gateway."
                )
            if prompt_vars is not None:
                raise ConfigurationError(
                    "FluentRun.with_(): prompt_vars= (client-side "
                    "str.format on the docstring/system prompt) and "
                    "system_arguments= (TZ server-side templates) are "
                    "mutually exclusive — pick one templating side."
                )
            return agent.invoke(message, output=mode, eager=False, **call_kwargs)
        system = call_kwargs.pop("system", system_prompt)
        if prompt_vars is not None:
            if system is None:
                raise ConfigurationError(
                    f"FluentRun.with_(prompt_vars=...): agent "
                    f"{spec_name!r} has no system prompt to format — "
                    f"give the decorated function a docstring or pass "
                    f"system= alongside."
                )
            try:
                system = system.format(**prompt_vars)
            except (IndexError, KeyError, ValueError) as exc:
                raise ConfigurationError(
                    f"FluentRun.with_(prompt_vars=...): the system "
                    f"prompt of agent {spec_name!r} could not be "
                    f"formatted: {exc!r}. Placeholders use str.format "
                    f"syntax — every {{name}} needs a matching "
                    f"prompt_vars key; literal braces need doubling."
                ) from exc
        return agent.invoke(
            message,
            output=mode,
            eager=False,
            system=system,
            **call_kwargs,
        )

    def _invoke(self, spec: _AgentSpec, message: str) -> FluentRun[Any]:
        """Build a :class:`FluentRun` over the underlying agent.

        Everything is deferred to first consumption: the ``prepare``
        hook (glow's auto-build, :meth:`_ensure_ready_for`) readies the
        App, then the build closure resolves the underlying
        :class:`Agent` and fires :meth:`Agent.invoke`.  Calling the
        wrapper is therefore free of side effects — a deferred-source
        ``App()`` can hand out handles before any gateway exists, and
        nothing (agent, tool filter) is cached before the App is built.
        Per-call overrides (.text / .json / .as_) swap the
        :class:`OutputMode` *before* the closure is consumed.
        """
        system_prompt = spec.system_prompt

        async def prepare() -> None:
            await self._ensure_ready_for(spec)

        def build(mode: Any, **call_kwargs: Any) -> Any:
            agent = self._lookup_or_build(spec)
            return self._agent_invoke(
                agent,
                message,
                mode,
                call_kwargs,
                system_prompt=system_prompt,
                spec_name=spec.name,
            )

        return FluentRun(
            build,
            spec.output_mode,
            value_transform=spec.value_transform,
            known_servers=tuple(
                s.namespace for s in self._mcp_specs if s.namespace is not None
            ),
            prepare=prepare,
        )

    def _invoke_in_conversation(
        self,
        spec: _AgentSpec,
        message: str,
        *,
        episode_id: str,
        tags: dict[str, str] | None,
        prior_messages: Callable[[], Sequence[Mapping[str, Any]] | None] | None,
        on_result: Callable[[AgentResult[Any]], None],
        extra_prepare: Callable[[], Awaitable[None]] | None = None,
    ) -> FluentRun[Any]:
        """Build a :class:`FluentRun` for one turn of a :class:`Conversation`.

        A sibling of :meth:`_invoke` that seeds the per-call invoke surface a
        conversation owns — ``episode_id`` (pinned to ``conv.id``), ``tags``,
        and an optional ``prior_messages`` snapshot resolved lazily at
        consumption (so it reflects any hydration and prior recorded turns) —
        then attaches the ``on_result`` recorder and folds an ``extra_prepare``
        step (hydration) into the auto-build ``prepare`` hook.  Seeds ride
        ``setdefault`` so an explicit ``.with_(episode_id=...)`` still wins, and
        agent resolution stays deferred to first consumption (the M7 rule that
        a config error surfaces at consumption, never at call).
        """
        system_prompt = spec.system_prompt

        async def prepare() -> None:
            await self._ensure_ready_for(spec)
            if extra_prepare is not None:
                await extra_prepare()

        def build(mode: Any, **call_kwargs: Any) -> Any:
            agent = self._lookup_or_build(spec)
            call_kwargs.setdefault("episode_id", episode_id)
            if tags:
                call_kwargs.setdefault("tags", dict(tags))
            if prior_messages is not None:
                seeded = prior_messages()
                if seeded:
                    call_kwargs.setdefault("prior_messages", seeded)
            return self._agent_invoke(
                agent,
                message,
                mode,
                call_kwargs,
                system_prompt=system_prompt,
                spec_name=spec.name,
            )

        return FluentRun(
            build,
            spec.output_mode,
            value_transform=spec.value_transform,
            known_servers=tuple(
                s.namespace for s in self._mcp_specs if s.namespace is not None
            ),
            prepare=prepare,
            on_result=on_result,
        )

    async def conversations(
        self,
        agent: _AgentWrapper[Any, Any],
        *,
        tags: Mapping[str, str] | None = None,
        limit: int = 20,
        index: ConversationIndex | None = None,
    ) -> list[ConversationSummary]:
        """List ``agent``'s recent conversations, newest first.

        **Dev-grade by default**: builds a
        :class:`~t0_agents.memory.tensorzero.TensorZeroMemory` over the App
        gateway and groups a bounded window of recent inferences client-side —
        see :meth:`TensorZeroMemory.list_conversations` for the exact horizon
        and ``turn_count`` guarantees.  Pass ``index=`` — any
        :class:`~t0_agents.core.conversation.ConversationIndex` — to delegate
        to a production inbox instead; that is the seam a real store plugs into
        with no further glow change.  ``tags`` scopes the listing (e.g.
        ``tags={"customer_id": cid}``), aligning read filters with the write
        tags a :class:`~t0_agents.glow.conversation.Conversation` stamps.
        """
        if index is not None:
            return list(await index.list_conversations(tags=tags, limit=limit))
        t0 = await self._ensure_t0()
        memory = TensorZeroMemory(t0, function_name=agent._spec.function_name)
        return list(await memory.list_conversations(tags=tags, limit=limit))

    async def build(self) -> None:
        """Eagerly construct every registered agent and connect the Client.

        Idempotent — calling :meth:`build` repeatedly enters the
        owned ``Client`` once (latched on ``_client_entered``) and
        re-uses the cached :class:`Agent` per spec.  Recommended in
        ``main()`` or inside :meth:`lifespan` so configuration errors
        (forward-ref resolution, invalid ``function_name``, MCP
        connection failures) surface before any user invocation kicks
        off network traffic.  (Auto-build via consumption boots the
        same infrastructure but constructs agents lazily per spec —
        see :meth:`_ensure_ready_for`.)
        """
        async with self._boot_lock:
            # Resolve the gateway first so it sits UNDER the client on the
            # exit stack (LIFO close: client, then gateway).  A constructor
            # source (gateway=/embedded=) always resolves; the deferred form
            # (bare App()) resolves only when some agent will need it — a
            # fully-overridden App keeps working with no gateway and no env.
            if self._t0 is None and (
                self._gateway_source is not None or self._needs_gateway()
            ):
                await self._ensure_t0()
            await self._start()
            for agent_spec in self._agent_registry.values():
                self._lookup_or_build(agent_spec)

    async def _start(self) -> None:
        """Boot the App infrastructure — everything except agent construction.

        Mounts MCP specs, enters the owned Client, and resolves every
        registered ``tools=`` filter against the live tool list.  Sets
        the ``_started`` latch.  Shared by the eager :meth:`build` and
        the consumption-time auto-build, which constructs only the
        agents actually consumed.
        """
        if not self._client_entered:
            # Materialise the FastMCP pair only when there's at least
            # one MCP spec to mount.  ``@app.tool`` registrations already
            # ensured allocation at decoration time; this branch covers
            # the ``MCPServer``-only case.
            if self._mcp_specs:
                server = self._ensure_mcp_server()
                # ``FastMCP.mount`` is sync and immediate.  ``url``-based
                # specs are wrapped via ``create_proxy`` (FastMCP's
                # canonical bridge from a URL / Client / Transport into a
                # mountable ``FastMCP`` instance); in-memory ``server``
                # specs are mounted directly (the ``examples/14_multi_server``
                # pattern).
                for mcp_spec in self._mcp_specs:
                    backend: FastMCP[Any]
                    if mcp_spec.server is not None:
                        backend = mcp_spec.server
                    else:
                        # ``mcp_spec.url`` is guaranteed non-None when
                        # ``mcp_spec.server`` is None (xor-enforced).
                        # ``_remote_target`` returns a bare URL, or an
                        # explicit transport carrying headers=/auth=.
                        backend = create_proxy(_remote_target(mcp_spec))
                    # ``prefix=False`` opts out of the namespace prefix:
                    # mount with ``namespace=None`` so tool names stay
                    # unprefixed (fastmcp presents them verbatim).
                    mount_namespace = mcp_spec.namespace if mcp_spec.prefix else None
                    server.mount(backend, namespace=mount_namespace)
            # Enter the owned Client only if it exists — sub-agents-only
            # apps that never registered a tool or an MCPServer spec
            # leave it ``None`` and skip the FastMCP transport entirely.
            if self._owned_client is not None:
                await self._exit_stack.enter_async_context(self._owned_client)
            self._client_entered = True
        # Resolve each agent's ``tools=`` filter against the now-connected
        # client's live tool list and warn on patterns matching nothing — a
        # build-time diagnostic that turns prefix-guessing bugs into a visible
        # warning.  A separate pass *before* the build loop so a parent
        # agent's recursion into its sub-agent specs sees their resolved
        # filters already set.
        live_names: set[str] = set()
        if self._owned_client is not None:
            live_names = {
                tool.name for tool in await self._owned_client.list_tools()
            }
        for agent_spec in self._agent_registry.values():
            if agent_spec.tools is not None:
                agent_spec.resolved_tools = _resolve_tool_filter(
                    agent_spec.name, agent_spec.tools, live_names
                )
        self._started = True

    async def shutdown(self) -> None:
        """Disconnect the owned Client and close any owned gateway.  Idempotent.

        Closes the internal :class:`AsyncExitStack`, which exits the
        ``Client.__aexit__`` registered by :meth:`build` and the owned
        gateway entered by :meth:`_ensure_t0` (LIFO — client first, then
        gateway).  A caller-supplied ``t0=`` is never closed.  Safe to
        call multiple times — the stack is replaced after each close so
        a subsequent :meth:`build` re-enters (and, for owned gateways,
        re-resolves) cleanly.  Required after any non-:meth:`lifespan`
        :meth:`build` to avoid leaking the in-process FastMCP transport
        task across test boundaries.
        """
        if self._client_entered or self._owns_t0:
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            self._client_entered = False
            if self._owns_t0:
                self._t0 = None
                self._owns_t0 = False
        self._started = False

    def _spec_needs_construction(self, spec: _AgentSpec) -> bool:
        """True when consuming ``spec`` will construct a real Agent."""
        return spec.agent is None and spec.name not in self.agent_overrides

    async def _ensure_ready_for(self, spec: _AgentSpec) -> None:
        """Ready the App for consuming ``spec`` — glow's auto-build.

        The first consumption boots the App infrastructure
        (INFO-logged): gateway resolution *when this spec needs
        construction*, MCP mounts, client connection, and ``tools=``
        filter resolution — so nothing is ever cached half-booted (the
        old silently-toolless / disconnected-client traps).  Unlike the
        explicit boot bracket (``async with app:`` / :meth:`build`),
        auto-build constructs agents lazily per consumed spec: an App
        whose consumed agents are all overridden keeps running with no
        gateway at all, and a misconfigured agent only fails its own
        consumption.  On an already-started App, a spec registered
        later (REPL ``replace=True``) gets its gateway and ``tools=``
        filter resolved late, before it builds.
        """
        # Hot-path fast exit: once started, only a spec with an
        # unresolved ``tools=`` filter has boot work left.  ``_started``
        # flips only under the boot lock, so this pre-check is safe.
        if self._started and not (
            spec.tools is not None and spec.resolved_tools is None
        ):
            return
        async with self._boot_lock:
            if not self._started:
                logger.info(
                    "glow App auto-building on first consumption (%d agent(s) "
                    "registered); use `async with app:` for explicit lifecycle "
                    "control.",
                    len(self._agent_registry),
                )
            if self._t0 is None and (
                (not self._started and self._gateway_source is not None)
                or self._spec_needs_construction(spec)
            ):
                await self._ensure_t0()
            if not self._started:
                await self._start()
                return
            if (
                spec.tools is not None
                and spec.resolved_tools is None
                and self._owned_client is not None
                and self._client_entered
            ):
                live_names = {
                    tool.name for tool in await self._owned_client.list_tools()
                }
                spec.resolved_tools = _resolve_tool_filter(
                    spec.name, spec.tools, live_names
                )

    async def __aenter__(self) -> App:
        """``async with app:`` — build on entry, shut down on exit.

        The boot bracket for scripts and services; byte-equivalent to
        ``async with app.lifespan():`` with one less call.  Auto-build
        (:meth:`_ensure_ready_for`) makes even this optional for
        notebooks and REPLs — the bracket buys deterministic teardown.
        """
        await self.build()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Async context manager that builds every agent and yields.

        Pattern lifted from FastAPI: ``async with app.lifespan(): ...``
        enters the owned Client, eagerly constructs each registered
        agent, yields control to the caller, then unconditionally
        disconnects on exit (even if the body raises).  ``async with
        app:`` is the equivalent shorthand; both make the Client (and
        any App-owned gateway) lifecycle bullet-proof.
        """
        await self.build()
        try:
            yield
        finally:
            await self.shutdown()
