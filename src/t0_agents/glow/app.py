"""``App`` — the registry + lifecycle for ``@app.agent`` decorators.

Pattern lift from FastAPI's ``app = FastAPI()`` followed by
``@app.get(...)``.  ``App`` owns:

- The :class:`AsyncTensorZeroGateway` reference threaded into every
  underlying :class:`~t0_agents.core.agent.Agent`.
- The agent registry (``name -> _AgentSpec``).
- The :attr:`dependency_overrides` mapping consulted on every
  ``Depends(...)`` resolution by :class:`~t0_agents.glow.di.DependencyResolver`.
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

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

from fastmcp import Client, FastMCP
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

if TYPE_CHECKING:
    from tensorzero import ToolChoice

    from t0_agents.core.middleware.base import Middleware
    from t0_agents.core.types import MemoryProvider
    from t0_agents.glow.decorators import _AgentWrapper

__all__ = ["App"]

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
    """

    def __init__(
        self,
        *,
        t0: AsyncTensorZeroGateway,
        servers: list[type[MCPServer]] | None = None,
        tags: dict[str, str] | None = None,
        middleware: list[Middleware] | None = None,
        namespace: str | None = None,
    ) -> None:
        self._t0: AsyncTensorZeroGateway = t0
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
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_entered: bool = False

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
        for server_cls in servers or []:
            if not (isinstance(server_cls, type) and issubclass(server_cls, MCPServer)):
                raise ConfigurationError(
                    f"App(servers=...): every entry must be a subclass of "
                    f"MCPServer; got {server_cls!r}."
                )
            self._mcp_specs.append(server_cls._mcp_spec)

    # ------------------------------------------------------------------
    # Decorator surface — delegates to ``glow.decorators._make_decorator``
    # ------------------------------------------------------------------

    def agent(
        self,
        name: str,
        *,
        function_name: str = "agent",
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
    ) -> Callable[[Callable[_P, Awaitable[_R]]], _AgentWrapper[_P, _R]]:
        """Return the ``@app.agent("name")`` decorator factory.

        Preserves the decorated function's parameters via ``ParamSpec``
        and its return annotation via ``TypeVar`` so static checkers
        see ``tutor("hi")`` typed as ``FluentRun[str]`` (and
        ``await tutor("hi")`` typed as ``str``) when the user writes
        ``async def tutor(q: str) -> str``.  Construction logic lives
        in :mod:`t0_agents.glow.decorators`; this method is the typed
        thin shim that delegates to ``_make_decorator``.

        Every keyword beyond ``name``/``function_name`` maps 1:1 onto the
        corresponding :class:`~t0_agents.core.config.AgentConfig` field
        (``max_turns``, ``tool_timeout``, ``parallel_tools``, ``dryrun``,
        ``tool_choice``, ``tags``, ``namespace``, ``memory``,
        ``memory_strategy``, ``middleware``).  Omitted knobs keep their
        ``AgentConfig`` defaults; supplied knobs are layered over any
        App-level defaults (agent-level wins — see :meth:`__init__`).
        """
        # Imported lazily to break the import cycle between
        # ``glow.app`` and ``glow.decorators`` (the latter imports
        # :class:`App` for type purposes).
        from t0_agents.glow.decorators import _make_decorator

        knobs: dict[str, Any] = {
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
            self, name, function_name=function_name, config_overrides=config_overrides
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
            self._owned_client = Client(self._mcp_server)
        return self._mcp_server

    def _resolve_config_overrides(self, spec: _AgentSpec) -> dict[str, Any]:
        """Layer App-level defaults under the agent-level ``config_overrides``.

        ``namespace`` and ``middleware`` take the agent-level value when the
        agent supplied one, else the App default.  ``tags`` dict-merge so
        ``App(tags={"env": "prod"})`` + ``@app.agent(tags={"team": "cs"})``
        yields both, with agent-level keys winning on collision (mirrors the
        core ``Agent._merge_tags`` precedence).
        """
        overrides: dict[str, Any] = dict(spec.config_overrides)
        if "namespace" not in overrides and self._default_namespace is not None:
            overrides["namespace"] = self._default_namespace
        if "middleware" not in overrides and self._default_middleware is not None:
            overrides["middleware"] = self._default_middleware
        if self._default_tags is not None:
            overrides["tags"] = {**self._default_tags, **(overrides.get("tags") or {})}
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
        if spec.agent is not None:
            return spec.agent
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
        agent = Agent(
            AgentConfig(**config_kwargs),
            t0=self._t0,
            tool_provider=tool_provider,
        )
        spec.agent = agent
        return agent

    def _invoke(self, spec: _AgentSpec, message: str) -> FluentRun[Any]:
        """Build a :class:`FluentRun` over the underlying agent.

        The build closure is captured here so :class:`FluentRun` can
        defer the :meth:`Agent.invoke` call until either ``__await__``,
        ``.result()``, or ``.events(...)`` fires.  Per-call overrides
        (.text / .json / .as_) swap the :class:`OutputMode` *before*
        the closure is consumed.
        """
        agent = self._lookup_or_build(spec)
        system_prompt = spec.system_prompt

        def build(mode: Any) -> Any:
            return agent.invoke(
                message,
                output=mode,
                eager=False,
                system=system_prompt,
            )

        return FluentRun(
            build,
            spec.output_mode,
            value_transform=spec.value_transform,
        )

    async def build(self) -> None:
        """Eagerly construct every registered agent and connect the Client.

        Idempotent — calling :meth:`build` repeatedly enters the
        owned ``Client`` once (latched on ``_client_entered``) and
        re-uses the cached :class:`Agent` per spec.  Recommended in
        ``main()`` or inside :meth:`lifespan` so configuration errors
        (forward-ref resolution, invalid ``function_name``, MCP
        connection failures) surface before any user invocation kicks
        off network traffic.
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
                        # ``mcp_spec.server`` is None — MCPServer's
                        # ``__init__`` / ``__init_subclass__`` enforces
                        # the xor.
                        backend = create_proxy(mcp_spec.url)  # type: ignore[arg-type]
                    server.mount(backend, namespace=mcp_spec.namespace)
            # Enter the owned Client only if it exists — sub-agents-only
            # apps that never registered a tool or an MCPServer spec
            # leave it ``None`` and skip the FastMCP transport entirely.
            if self._owned_client is not None:
                await self._exit_stack.enter_async_context(self._owned_client)
            self._client_entered = True
        for agent_spec in self._agent_registry.values():
            self._lookup_or_build(agent_spec)

    async def shutdown(self) -> None:
        """Disconnect the owned Client.  Idempotent.

        Closes the internal :class:`AsyncExitStack`, which exits the
        ``Client.__aexit__`` registered by :meth:`build`.  Safe to call
        multiple times — the stack is replaced after each close so a
        subsequent :meth:`build` re-enters cleanly.  Required after
        any non-:meth:`lifespan` :meth:`build` to avoid leaking the
        in-process FastMCP transport task across test boundaries.
        """
        if self._client_entered:
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            self._client_entered = False

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Async context manager that builds every agent and yields.

        Pattern lifted from FastAPI: ``async with app.lifespan(): ...``
        enters the owned Client, eagerly constructs each registered
        agent, yields control to the caller, then unconditionally
        disconnects on exit (even if the body raises).  This is the
        recommended way to use :class:`App` because it makes the
        Client lifecycle bullet-proof.
        """
        await self.build()
        try:
            yield
        finally:
            await self.shutdown()
