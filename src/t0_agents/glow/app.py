"""``App`` — the registry + lifecycle for ``@app.agent`` decorators.

Pattern lift from FastAPI's ``app = FastAPI()`` followed by
``@app.get(...)``.  ``App`` owns:

- The :class:`AsyncTensorZeroGateway` reference threaded into every
  underlying :class:`~t0_agents.core.agent.Agent`.
- The agent registry (``name -> _AgentSpec``).
- The :attr:`dependency_overrides` mapping consulted on every
  ``Depends(...)`` resolution by :class:`~t0_agents.glow.di.DependencyResolver`.
- A long-lived in-memory ``fastmcp.FastMCP`` server (``_mcp_server``)
  and the matching ``fastmcp.Client`` (``_owned_client``).  ``@app.tool``
  registers user tools on the server; every glow agent is constructed
  with ``tool_provider=self._owned_client`` so the v1.0 toolless-chat
  fallback is never reached for glow agents (M2 §4 carry-forward
  option (b)).
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
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

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
    from t0_agents.glow.decorators import _AgentWrapper

__all__ = ["App"]

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
    """

    def __init__(
        self,
        *,
        t0: AsyncTensorZeroGateway,
        servers: list[type[MCPServer]] | None = None,
    ) -> None:
        self._t0: AsyncTensorZeroGateway = t0
        self._agent_registry: dict[str, _AgentSpec] = {}
        # M3 wires the resolution path; the attribute lives here so
        # users can install overrides without raising ``AttributeError``
        # at any point in the lifecycle.
        self.dependency_overrides: dict[Callable[..., Any], Callable[..., Any]] = {}

        # App-owned FastMCP server + matching Client.  The server starts
        # empty; ``@app.tool`` registers tools onto it.  Every glow agent
        # is constructed with ``tool_provider=self._owned_client``, so
        # the v1.0 toolless-chat fallback at
        # ``core/agent.py:_resolve_chat_tool_provider`` is never reached
        # for glow agents (M2 §4 carry-forward option (b)).  The Client
        # is connected by :meth:`build` (idempotent); :meth:`shutdown`
        # exits the matching ``__aexit__``.
        self._mcp_server: FastMCP[Any] = FastMCP("glow-app")
        self._owned_client: Client[Any] = Client(self._mcp_server)
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_entered: bool = False

        # The :class:`DependencyResolver` instance lives here so it can
        # consult ``self.dependency_overrides`` directly during
        # ``@app.tool`` injections.  ``DependencyResolver`` is stateless
        # beyond the per-call cache it threads through recursive
        # resolution, so a single instance per ``App`` is safe.
        self._resolver: DependencyResolver = DependencyResolver()

        # M4: typed-MCP integration.  Specs discovered via the ``servers=``
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
    ) -> Callable[[Callable[_P, Awaitable[_R]]], _AgentWrapper[_P, _R]]:
        """Return the ``@app.agent("name")`` decorator factory.

        Preserves the decorated function's parameters via ``ParamSpec``
        and its return annotation via ``TypeVar`` so static checkers
        see ``tutor("hi")`` typed as ``FluentRun[str]`` (and
        ``await tutor("hi")`` typed as ``str``) when the user writes
        ``async def tutor(q: str) -> str``.  Construction logic lives
        in :mod:`t0_agents.glow.decorators`; this method is the typed
        thin shim that delegates to ``_make_decorator``.
        """
        # Imported lazily to break the import cycle between
        # ``glow.app`` and ``glow.decorators`` (the latter imports
        # :class:`App` for type purposes).
        from t0_agents.glow.decorators import _make_decorator

        return _make_decorator(self, name, function_name=function_name)

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

    def _lookup_or_build(self, spec: _AgentSpec) -> Agent:
        """Return the underlying :class:`Agent`, constructing it if needed.

        Sync — :class:`Agent.__init__` is sync.  ``async`` is reserved
        for :meth:`build` so M4's MCP-mount machinery can interleave
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
            # M4: feed harvested ``Annotated[T, Guardrail(...)]`` metadata
            # into the v1.0 ``AgentConfig.guardrails`` field so the
            # existing ``GuardrailsMiddleware`` picks them up at runtime
            # without further glow plumbing.
            config_kwargs["guardrails"] = spec.guardrails
        agent = Agent(
            AgentConfig(**config_kwargs),
            t0=self._t0,
            tool_provider=cast(ToolProvider, self._owned_client),
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
            # M4: mount every typed MCP server BEFORE entering the
            # client so the mounted tools surface on the first
            # ``list_tools()`` call.  ``FastMCP.mount`` is sync and
            # immediate — no client re-entry required.  ``url``-based
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
                self._mcp_server.mount(backend, namespace=mcp_spec.namespace)
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
