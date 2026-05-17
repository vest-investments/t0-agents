"""``App`` — the registry + lifecycle for ``@app.agent`` decorators.

Pattern lift from FastAPI's ``app = FastAPI()`` followed by
``@app.get(...)``.  ``App`` owns:

- The :class:`AsyncTensorZeroGateway` reference threaded into every
  underlying :class:`~t0_agents.core.agent.Agent`.
- The agent registry (``name -> _AgentSpec``).
- The :attr:`dependency_overrides` mapping used by M3's
  ``Depends(...)`` machinery (in M2 it is an empty dict that lives
  here for forward-compat — ``app.dependency_overrides[provider] = mock``
  works without raising ``AttributeError`` even though resolution
  itself lands in M3).
- The lazy build path: each ``_AgentSpec`` materialises its underlying
  :class:`Agent` on first ``await wrapper(msg)`` *or* eagerly via
  ``await app.build()`` (recommended in ``main()`` / ``lifespan()`` so
  configuration errors surface fast).

``App`` does **not** modify any v1.0 primitive — every
``@app.agent`` decorator compiles to a normal :class:`Agent` that
exposes itself on ``wrapper._agent`` for escape-hatch use.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.glow._spec import _AgentSpec
from t0_agents.glow.handle import FluentRun

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

                print(await tutor("Say hi formally in Japanese."))

    Agents are built lazily on first invocation.  Use
    ``await app.build()`` (or wrap in ``async with app.lifespan():``)
    to force-construct every registered agent — useful in scenarios
    where configuration errors should surface eagerly.
    """

    def __init__(self, *, t0: AsyncTensorZeroGateway) -> None:
        self._t0: AsyncTensorZeroGateway = t0
        self._agent_registry: dict[str, _AgentSpec] = {}
        # M3 wires the resolution path; the attribute lives here so
        # users can install overrides in M2 code without raising
        # ``AttributeError`` ("app.dependency_overrides[get_db] = mock"
        # is a no-op in M2 because there are no providers yet).
        self.dependency_overrides: dict[Callable[..., Any], Callable[..., Any]] = {}

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _lookup_or_build(self, spec: _AgentSpec) -> Agent:
        """Return the underlying :class:`Agent`, constructing it if needed.

        Sync — :class:`Agent.__init__` is sync.  ``async`` is reserved
        for :meth:`build` so M4's MCP-mount machinery can interleave
        ``await FastMCP.mount(...)`` cleanly.
        """
        if spec.agent is not None:
            return spec.agent
        agent = Agent(
            AgentConfig(
                name=spec.name,
                function_name=spec.function_name,
                output=spec.output_mode,
            ),
            t0=self._t0,
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
        """Eagerly construct every registered agent.

        Recommended in ``main()`` or inside :meth:`lifespan` so
        configuration errors (forward-ref resolution, invalid
        ``function_name``, etc.) surface before any user invocation
        kicks off network traffic.
        """
        for spec in self._agent_registry.values():
            self._lookup_or_build(spec)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Async context manager that builds every agent and yields.

        Pattern lifted from FastAPI: ``async with app.lifespan(): ...``
        is the natural place to surface eager construction errors.
        M4 extends teardown with MCP server unmount; M2 has nothing to
        clean up on exit.
        """
        await self.build()
        yield
