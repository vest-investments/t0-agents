"""App boot forms — gateway sources, deferred env resolution, ownership.

M7 (P4 boot collapse): ``App(gateway=...)``, ``App(embedded=...)``, and
the bare ``App()`` deferred form resolve the TensorZero gateway lazily
at build time; resolved gateways are App-owned (entered on the exit
stack, closed by ``shutdown()``, re-resolved on the next ``build()``).
A caller-supplied ``t0=`` is never entered or closed.  The SDK builders
are faked by swapping the ``AsyncTensorZeroGateway`` symbol inside
``t0_agents.glow.app`` — no network, no docker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import AgentResult
from t0_agents.glow import App
from t0_agents.tooling import FilteredToolProvider


class _FakeGateway:
    """Records async-CM entry/exit — stands in for the real gateway."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeGateway:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boot tests must not inherit gateway env vars from the dev shell."""
    monkeypatch.delenv("TENSORZERO_URL", raising=False)
    monkeypatch.delenv("TENSORZERO_CONFIG", raising=False)


@pytest.fixture()
def fake_builders(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Swap the gateway class seen by ``glow.app`` for recording fakes.

    The real ``AsyncTensorZeroGateway`` is a native (pyo3) class whose
    attributes cannot be monkeypatched — replacing the module-level
    symbol that ``App._ensure_t0`` resolves through is the reliable
    seam.  Each builder returns an awaitable of a fresh ``_FakeGateway``
    (mirroring the SDK's ``async_setup=True`` union shape).
    """
    calls: dict[str, list[Any]] = {"http": [], "embedded": [], "gateways": []}

    class _Builders:
        @staticmethod
        def build_http(**kwargs: Any) -> Any:
            calls["http"].append(kwargs)
            gw = _FakeGateway()
            calls["gateways"].append(gw)

            async def _make() -> _FakeGateway:
                return gw

            return _make()

        @staticmethod
        def build_embedded(**kwargs: Any) -> Any:
            calls["embedded"].append(kwargs)
            gw = _FakeGateway()
            calls["gateways"].append(gw)

            async def _make() -> _FakeGateway:
                return gw

            return _make()

    monkeypatch.setattr("t0_agents.glow.app.AsyncTensorZeroGateway", _Builders)
    return calls


def _register_agent(app: App, name: str = "tutor") -> Any:
    @app.agent(name)
    async def tutor(query: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    return tutor


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_multiple_gateway_sources() -> None:
    sentinel = cast(AsyncTensorZeroGateway, object())
    with pytest.raises(ConfigurationError, match="at most one"):
        App(t0=sentinel, gateway="http://localhost:3000")
    with pytest.raises(ConfigurationError, match="at most one"):
        App(gateway="http://localhost:3000", embedded="config/tensorzero.toml")
    with pytest.raises(ConfigurationError, match="at most one"):
        App(t0=sentinel, embedded="config/tensorzero.toml")
    with pytest.raises(ConfigurationError, match="at most one"):
        App(
            t0=sentinel,
            gateway="http://localhost:3000",
            embedded="config/tensorzero.toml",
        )


# ---------------------------------------------------------------------------
# Owned sources — gateway= / embedded=
# ---------------------------------------------------------------------------


async def test_gateway_url_form_builds_http_and_owns_it(
    fake_builders: dict[str, list[Any]],
) -> None:
    app = App(gateway="http://gw.example:3000")

    await app.build()
    assert fake_builders["http"] == [{"gateway_url": "http://gw.example:3000"}]
    assert fake_builders["embedded"] == []
    gw = fake_builders["gateways"][0]
    assert gw.entered and not gw.exited

    await app.shutdown()
    assert gw.exited

    # Re-entry re-resolves: the App released ownership on shutdown.
    await app.build()
    assert len(fake_builders["http"]) == 2
    await app.shutdown()


async def test_embedded_form_builds_embedded(
    fake_builders: dict[str, list[Any]],
) -> None:
    app = App(embedded="config/tensorzero.toml")

    await app.build()
    assert fake_builders["embedded"] == [{"config_file": "config/tensorzero.toml"}]
    assert fake_builders["http"] == []
    await app.shutdown()
    assert fake_builders["gateways"][0].exited


# ---------------------------------------------------------------------------
# Deferred source — bare App() + environment
# ---------------------------------------------------------------------------


async def test_deferred_resolves_url_env_first(
    fake_builders: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORZERO_URL", "http://from-env:3000")
    monkeypatch.setenv("TENSORZERO_CONFIG", "ignored.toml")
    app = App()
    _register_agent(app)

    await app.build()
    assert fake_builders["http"] == [{"gateway_url": "http://from-env:3000"}]
    assert fake_builders["embedded"] == []
    await app.shutdown()


async def test_deferred_resolves_config_env_when_no_url(
    fake_builders: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORZERO_CONFIG", "config/dev.toml")
    app = App()
    _register_agent(app)

    await app.build()
    assert fake_builders["embedded"] == [{"config_file": "config/dev.toml"}]
    await app.shutdown()


async def test_deferred_without_env_and_needing_gateway_raises(
    fake_builders: dict[str, list[Any]],
) -> None:
    app = App()
    _register_agent(app)

    with pytest.raises(ConfigurationError, match="no gateway"):
        await app.build()
    assert fake_builders["http"] == [] and fake_builders["embedded"] == []


async def test_deferred_fully_overridden_app_never_resolves(
    fake_builders: dict[str, list[Any]],
) -> None:
    """An App whose every agent is overridden needs no gateway and no env."""
    from t0_agents.testing import ScriptedAgent

    app = App()
    _register_agent(app, "forecaster")
    app.agent_overrides["forecaster"] = ScriptedAgent(result="22C")

    await app.build()
    assert fake_builders["http"] == [] and fake_builders["embedded"] == []
    await app.shutdown()


# ---------------------------------------------------------------------------
# Auto-build on first consumption + async with app  (the trap killers)
# ---------------------------------------------------------------------------


class _FakeRun:
    async def result(self) -> AgentResult[Any]:
        return AgentResult(
            value="ok", episode_id="ep_boot", inference_ids=["if_boot"], turns=1
        )


class _FakeAgent:
    """Records construction — glow builds it in place of the real Agent."""

    instances: ClassVar[list[_FakeAgent]] = []

    def __init__(self, config: Any, *, t0: Any, tool_provider: Any = None) -> None:
        self.config = config
        self.t0 = t0
        self.tool_provider = tool_provider
        type(self).instances.append(self)

    def invoke(self, user_message: str, **kwargs: Any) -> Any:
        return _FakeRun()


@pytest.fixture()
def fake_agent_cls(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAgent]:
    _FakeAgent.instances = []
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    return _FakeAgent


async def test_consumption_auto_builds_with_info_and_kills_both_traps(
    fake_agent_cls: type[_FakeAgent],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Consuming without any lifespan must yield a FULLY built agent.

    Pre-M7 this cached a half-booted agent forever: the ``tools=``
    filter stayed unresolved (silently void — trap A) and the owned
    FastMCP client stayed disconnected (opaque error at first
    list_tools — trap B).
    """
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)

    @app.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    @app.agent("tutor", tools=["ping"])
    async def tutor(query: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    with caplog.at_level(logging.INFO, logger="t0_agents.glow.app"):
        assert await tutor("hi") == "ok"

    assert any("auto-building on first consumption" in r.message for r in caplog.records)
    # Trap B: the owned client got connected by the auto-build.
    assert app._client_entered
    # Trap A: the agent was built AFTER filter resolution — wrapped.
    (agent,) = fake_agent_cls.instances
    assert isinstance(agent.tool_provider, FilteredToolProvider)

    caplog.clear()
    # The latch holds: second consumption neither rebuilds nor re-logs.
    assert await tutor("again") == "ok"
    assert len(fake_agent_cls.instances) == 1
    assert not any("auto-building" in r.message for r in caplog.records)
    await app.shutdown()


async def test_concurrent_first_consumptions_boot_once(
    fake_builders: dict[str, list[Any]],
    fake_agent_cls: type[_FakeAgent],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Racing first consumptions must serialize on the boot lock.

    Without it, concurrent ``_ensure_ready_for`` calls interleave at
    the await points and resolve the gateway (and enter the client)
    more than once — the gemini-flagged race on PR #77.
    """
    app = App(gateway="http://gw.example:3000")

    @app.agent("alpha")
    async def alpha(query: str) -> str:  # type: ignore[empty-body]
        """Alpha."""

    @app.agent("beta")
    async def beta(query: str) -> str:  # type: ignore[empty-body]
        """Beta."""

    with caplog.at_level(logging.INFO, logger="t0_agents.glow.app"):
        results = await asyncio.gather(alpha("x"), beta("y"))

    assert results == ["ok", "ok"]
    # Exactly one gateway resolved and one boot logged — not one per racer.
    assert len(fake_builders["gateways"]) == 1
    assert sum("auto-building" in r.message for r in caplog.records) == 1
    await app.shutdown()
    assert fake_builders["gateways"][0].exited


async def test_async_with_app_is_the_lifespan_bracket(
    fake_agent_cls: type[_FakeAgent],
) -> None:
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)
    _register_agent(app)

    async with app as entered:
        assert entered is app
        # Eager build: the agent exists before any consumption.
        assert len(fake_agent_cls.instances) == 1
    # Exit tears down: a later build re-enters cleanly (idempotent path).
    assert not app._client_entered


# ---------------------------------------------------------------------------
# REPL surface — @app.agent(replace=True) + App.remove_agent
# ---------------------------------------------------------------------------


async def test_replace_reregisters_and_rebuilds(
    fake_agent_cls: type[_FakeAgent],
) -> None:
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)

    @app.agent("tutor")
    async def tutor_v1(query: str) -> str:  # type: ignore[empty-body]
        """Version one."""

    assert await tutor_v1("hi") == "ok"
    assert len(fake_agent_cls.instances) == 1

    @app.agent("tutor", replace=True)
    async def tutor_v2(query: str) -> str:  # type: ignore[empty-body]
        """Version two."""

    assert await tutor_v2("hi") == "ok"
    # A fresh Agent was built for the fresh spec; the old wrapper keeps
    # its own (detached) spec and cached agent.
    assert len(fake_agent_cls.instances) == 2
    assert await tutor_v1("still works") == "ok"
    assert len(fake_agent_cls.instances) == 2
    await app.shutdown()


async def test_replace_after_start_late_resolves_tools(
    fake_agent_cls: type[_FakeAgent],
) -> None:
    """A replacement registered on a started App still gets its filter."""
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)

    @app.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    @app.agent("tutor")
    async def tutor_v1(query: str) -> str:  # type: ignore[empty-body]
        """No filter."""

    assert await tutor_v1("boot") == "ok"  # auto-build; App is started

    @app.agent("tutor", replace=True, tools=["ping"])
    async def tutor_v2(query: str) -> str:  # type: ignore[empty-body]
        """Filtered."""

    assert await tutor_v2("hi") == "ok"
    assert isinstance(fake_agent_cls.instances[-1].tool_provider, FilteredToolProvider)
    await app.shutdown()


async def test_remove_agent_frees_the_name(fake_agent_cls: type[_FakeAgent]) -> None:
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)
    _register_agent(app, "tutor")

    app.remove_agent("tutor")
    # The name is free again — plain re-registration, no replace= needed.
    _register_agent(app, "tutor")

    with pytest.raises(ConfigurationError, match="no agent is registered"):
        app.remove_agent("ghost")


# ---------------------------------------------------------------------------
# Explicit t0= — caller-owned, never touched
# ---------------------------------------------------------------------------


async def test_explicit_t0_never_entered_nor_closed(
    fake_builders: dict[str, list[Any]],
) -> None:
    sentinel = cast(AsyncTensorZeroGateway, object())
    app = App(t0=sentinel)
    _register_agent(app)

    await app.build()
    await app.shutdown()
    assert fake_builders["http"] == [] and fake_builders["embedded"] == []
    assert app._t0 is sentinel
