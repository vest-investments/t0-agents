"""``Depends`` sentinel + :class:`DependencyResolver` for glow.

The injection model is lifted from FastAPI: a function parameter whose
default is a :class:`Depends` instance is resolved at call time by
walking the provider's own signature, recursively resolving its own
``Depends`` defaults, and threading the resolved value in as a keyword
argument.  Two integration points exist in glow:

* :func:`~t0_agents.glow.app.App.tool` — strips ``Depends`` parameters
  from the FastMCP-introspected schema and re-injects them at call
  time inside the tool wrapper.
* :func:`~t0_agents.glow.app.App.agent` — only accepts
  ``Depends(glow_wrapper)`` (the sub-agent special case); every other
  ``Depends`` shape is rejected at decoration time with a pointer to
  ``@app.tool``.

The resolver is intentionally a small class (not free functions) so
the per-call cache lives on the resolver between recursive calls
without leaking across invocations.  An ``App`` owns one
:class:`DependencyResolver` instance; the cache is reset implicitly
by passing a fresh ``call_cache`` dict from the entry point.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from t0_agents.glow.app import App

__all__ = ["DependencyResolver", "Depends"]


@dataclass(frozen=True, slots=True)
class _Depends:
    """Internal sentinel default backing the public :func:`Depends`.

    ``provider`` is the callable that produces the value.  Sync and
    async providers are both supported; the resolver dispatches on
    :func:`inspect.iscoroutinefunction` plus :func:`inspect.isawaitable`
    so factories returning awaitables also work.

    When ``use_cache`` is ``True`` (the default), the same instance
    resolved twice in one call returns the cached value.  The cache is
    keyed by :func:`id` of the sentinel instance, so two distinct
    ``Depends(get_db)`` literals are cached separately; callers wanting
    cross-parameter reuse must share a single module-level instance
    (the FastAPI idiom).

    ``app.dependency_overrides[provider] = mock_provider`` replaces
    the resolved provider at call time (not at registration), so
    tests can swap dependencies without re-instantiating the decorator.

    End users construct sentinel instances via :func:`Depends`; mypy
    sees that callable as returning :class:`Any`, so a typed parameter
    default ``db: DB = Depends(get_db)`` does not trip the
    ``[assignment]`` check.  ``isinstance(x, _Depends)`` is the runtime
    discriminator used by the resolver and the decorator-side signature
    walks; the bare name ``_Depends`` stays internal so the public
    surface presents :func:`Depends` as a value-constructor, not a
    class.
    """

    provider: Callable[..., Any]
    use_cache: bool = True


if TYPE_CHECKING:
    # Static-check view: ``Depends`` is a callable returning ``Any`` so
    # a typed parameter default — ``db: DB = Depends(get_db)`` — is
    # accepted by mypy strict.  FastAPI uses the same trick.
    def Depends(  # noqa: N802 — public name mirrors FastAPI convention.
        provider: Callable[..., Any],
        *,
        use_cache: bool = True,
    ) -> Any: ...
else:
    # Runtime view: ``Depends`` IS the dataclass class, so
    # ``Depends(get_db)`` constructs a sentinel instance and
    # ``isinstance(x, Depends)`` works for external callers that
    # introspect (we still prefer ``isinstance(x, _Depends)`` inside
    # this package — both are the same object at runtime).
    Depends = _Depends


class DependencyResolver:
    """Walks function signatures and resolves :class:`Depends` defaults.

    Stateless beyond the per-call cache passed in.  One instance is
    owned by each :class:`~t0_agents.glow.app.App`; callers don't
    interact with it directly — the ``@app.tool`` decorator and the
    agent-side ``Depends`` plumbing call ``resolve`` from inside their
    own wrappers.
    """

    async def resolve(
        self,
        fn: Callable[..., Any],
        app: App,
        call_cache: dict[int, Any] | None = None,
    ) -> dict[str, Any]:
        """Return ``{param_name: resolved_value}`` for every ``Depends`` default.

        Parameters without a ``Depends`` default are skipped — the
        caller is expected to populate them from elsewhere (the MCP
        tool call's arguments, or the agent's user message).

        ``call_cache`` should be ``None`` at the entry point; the
        resolver allocates a fresh dict and threads it through every
        recursive call so a single ``Depends`` instance reused across
        the resolution tree is computed once.
        """
        if call_cache is None:
            call_cache = {}

        sig = inspect.signature(fn)
        resolved: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            default = param.default
            if not isinstance(default, _Depends):
                continue
            resolved[name] = await self._resolve_one(default, app, call_cache)
        return resolved

    async def _resolve_one(
        self,
        dep: _Depends,
        app: App,
        call_cache: dict[int, Any],
    ) -> Any:
        """Resolve a single ``Depends`` instance, honoring cache + overrides.

        ``app.dependency_overrides`` is consulted *every* call (not
        snapshotted at registration) so test-time replacement works
        even after the decorator has fired.  The provider's own
        ``Depends`` defaults are resolved recursively; the same
        ``call_cache`` is threaded all the way down so cycle-of-one
        deduplication holds across the whole tree.
        """
        if dep.use_cache and id(dep) in call_cache:
            return call_cache[id(dep)]

        provider: Callable[..., Any] = app.dependency_overrides.get(
            dep.provider, dep.provider
        )

        provider_kwargs = await self.resolve(provider, app, call_cache)

        value: Any
        if inspect.iscoroutinefunction(provider):
            value = await provider(**provider_kwargs)
        else:
            value = provider(**provider_kwargs)
            if inspect.isawaitable(value):
                value = await value

        if dep.use_cache:
            call_cache[id(dep)] = value
        return value
