"""Unit tests for :class:`t0_agents.glow.di.DependencyResolver`.

Seven cases (no TZ gateway):

* Single ``Depends(get_db)`` resolves and the resolved value is
  threaded into the call kwargs.
* Nested ``Depends`` — the provider's own ``Depends`` defaults are
  resolved recursively.
* ``use_cache=True`` (default): the same ``Depends`` instance
  resolved twice in one call returns the same value.
* ``use_cache=False``: each resolution invokes the provider freshly.
* ``app.dependency_overrides[get_db] = mock_db`` replaces the
  resolved value at call time (not snapshotted at registration).
* Sync providers are called inline; async providers (and sync
  providers returning awaitables) are awaited.
* Parameters without a ``Depends`` default are left untouched.
"""

from __future__ import annotations

from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, Depends
from t0_agents.glow.di import DependencyResolver


def _fake_t0() -> AsyncTensorZeroGateway:
    """Return an opaque sentinel cast to :class:`AsyncTensorZeroGateway`.

    :class:`App.__init__` stores the reference but does not touch it
    during resolution-only tests.
    """
    return cast(AsyncTensorZeroGateway, object())


async def test_single_depends_resolves_and_threads_value() -> None:
    """A single ``Depends(get_db)`` produces a ``{db: <value>}`` map."""
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    def get_db() -> dict[str, str]:
        return {"name": "real_db"}

    async def use_db(db: dict[str, str] = Depends(get_db)) -> None: ...

    resolved = await resolver.resolve(use_db, app)
    assert resolved == {"db": {"name": "real_db"}}


async def test_nested_depends_resolves_recursively() -> None:
    """A provider that itself has a ``Depends`` default is resolved depth-first."""
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    def get_settings() -> dict[str, str]:
        return {"db_url": "sqlite:///:memory:"}

    def get_db(settings: dict[str, str] = Depends(get_settings)) -> dict[str, str]:
        return {"url": settings["db_url"]}

    async def use_db(db: dict[str, str] = Depends(get_db)) -> None: ...

    resolved = await resolver.resolve(use_db, app)
    assert resolved == {"db": {"url": "sqlite:///:memory:"}}


async def test_use_cache_default_dedupes_within_one_call() -> None:
    """The same ``Depends`` instance reused across params returns one value.

    ``use_cache=True`` (default) keys the cache by ``id(Depends)`` —
    same instance, same result.  The provider must be called exactly
    once per resolution tree.
    """
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    call_count = 0

    def get_db() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"call": call_count}

    shared_dep = Depends(get_db)

    async def use_two_db_params(
        a: dict[str, Any] = shared_dep,
        b: dict[str, Any] = shared_dep,
    ) -> None: ...

    resolved = await resolver.resolve(use_two_db_params, app)
    assert call_count == 1
    assert resolved["a"] is resolved["b"]
    assert resolved["a"] == {"call": 1}


async def test_use_cache_false_invokes_provider_fresh() -> None:
    """``use_cache=False`` bypasses the cache; each resolution re-invokes."""
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    call_count = 0

    def get_db() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"call": call_count}

    shared_dep = Depends(get_db, use_cache=False)

    async def use_two_db_params(
        a: dict[str, Any] = shared_dep,
        b: dict[str, Any] = shared_dep,
    ) -> None: ...

    resolved = await resolver.resolve(use_two_db_params, app)
    assert call_count == 2
    assert resolved["a"] == {"call": 1}
    assert resolved["b"] == {"call": 2}


async def test_dependency_overrides_replaces_provider_at_call_time() -> None:
    """``app.dependency_overrides[get_db] = mock_db`` swaps the resolved value.

    Override is set *after* the function is defined, so the resolver
    cannot have snapshotted the provider at decoration time — every
    resolution must look up the override fresh.
    """
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    def get_db() -> dict[str, str]:
        return {"name": "real_db"}

    async def use_db(db: dict[str, str] = Depends(get_db)) -> None: ...

    # Real path first.
    resolved = await resolver.resolve(use_db, app)
    assert resolved == {"db": {"name": "real_db"}}

    # Install override and re-resolve — the value changes without
    # rewriting `use_db`.
    def mock_db() -> dict[str, str]:
        return {"name": "mock_db"}

    app.dependency_overrides[get_db] = mock_db
    resolved = await resolver.resolve(use_db, app)
    assert resolved == {"db": {"name": "mock_db"}}

    # Remove the override → original provider is used again.
    del app.dependency_overrides[get_db]
    resolved = await resolver.resolve(use_db, app)
    assert resolved == {"db": {"name": "real_db"}}


async def test_async_and_sync_providers_both_work() -> None:
    """Async providers are ``await``-ed; sync providers run inline.

    Also covers the "sync provider returning Awaitable" edge case:
    if a sync function returns a coroutine, the resolver awaits it.
    """
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    def sync_provider() -> str:
        return "sync"

    async def async_provider() -> str:
        return "async"

    def sync_returns_awaitable() -> Any:
        # A sync wrapper that returns the awaitable produced by an
        # async helper.  Common pattern when adapting libraries that
        # return coroutines from sync factories.
        async def _inner() -> str:
            return "sync-returns-awaitable"

        return _inner()

    async def consumer(
        a: str = Depends(sync_provider),
        b: str = Depends(async_provider),
        c: str = Depends(sync_returns_awaitable),
    ) -> None: ...

    resolved = await resolver.resolve(consumer, app)
    assert resolved == {"a": "sync", "b": "async", "c": "sync-returns-awaitable"}


async def test_non_depends_parameters_are_left_untouched() -> None:
    """Parameters without a ``Depends`` default don't appear in the
    resolved map — the caller supplies them from elsewhere.
    """
    app = App(t0=_fake_t0())
    resolver = DependencyResolver()

    def get_db() -> str:
        return "db"

    async def mixed(
        user_message: str,
        order_id: int,
        db: str = Depends(get_db),
    ) -> None: ...

    resolved = await resolver.resolve(mixed, app)
    assert resolved == {"db": "db"}
    assert "user_message" not in resolved
    assert "order_id" not in resolved
