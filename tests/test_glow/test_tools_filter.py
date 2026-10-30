"""Unit tests for per-agent ``@app.agent(tools=[...])`` filtering (M4).

Gateway-free throughout:

* the pure resolver ``_resolve_tool_filter`` is driven with a synthetic live
  tool-name set (each spec kind, dead-pattern warning, ``namespace=`` guard,
  toolless, unsupported entry);
* the end-to-end wiring runs through ``app.lifespan()`` over an in-memory
  ``@app.tool`` surface with the real :class:`Agent` (``t0`` is a bare sentinel
  — :meth:`Agent.__init__` never touches it), so the real
  :class:`SubAgentToolProvider` composition is exercised for the ``_ask``
  passthrough case.
"""

from __future__ import annotations

import socket
import warnings
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError, T0AgentsError
from t0_agents.delegation import SubAgentToolProvider
from t0_agents.glow import App, MCPServer
from t0_agents.glow.app import _resolve_tool_filter
from t0_agents.glow.decorators import _tool_specs
from t0_agents.testing import MockToolProvider
from t0_agents.tooling import FilteredToolProvider


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


def _refused_url() -> str:
    """A URL on a just-released local port — connects are refused fast."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/mcp"


async def _tool_names(wrapper: Any) -> set[str]:
    """Names the wrapper's built agent actually exposes to inference."""
    provider = wrapper.agent._tool_provider
    return {tool.name for tool in await provider.list_tools()}


# ---------------------------------------------------------------------------
# Resolver — pure, no gateway
# ---------------------------------------------------------------------------


def test_resolve_glob_pattern() -> None:
    live = {"get_a", "get_b", "set_c"}
    assert _resolve_tool_filter("ag", ["get_*"], live) == {"get_a", "get_b"}


def test_resolve_exact_string_is_literal_match() -> None:
    live = {"weather", "weather_forecast"}
    assert _resolve_tool_filter("ag", ["weather"], live) == {"weather"}


def test_resolve_callable_by_dunder_name() -> None:
    live = {"lookup_order", "other"}

    async def lookup_order() -> None: ...

    assert _resolve_tool_filter("ag", [lookup_order], live) == {"lookup_order"}


def test_resolve_mcp_server_by_namespace_prefix() -> None:
    live = {"math_add", "math_sub", "weather_now"}

    class Math(MCPServer, url="http://x", namespace="math"):
        pass

    assert _resolve_tool_filter("ag", [Math], live) == {"math_add", "math_sub"}


def test_resolve_empty_tools_is_empty_set_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert _resolve_tool_filter("ag", [], {"weather"}) == set()


def test_dead_pattern_warns_naming_candidates() -> None:
    with pytest.warns(UserWarning, match="matched no available tool"):
        result = _resolve_tool_filter("ag", ["nonexistent_*"], {"weather"})
    assert result == set()


def test_bare_string_tools_raises_instead_of_widening() -> None:
    """`tools="get_*"` splits into characters, and the '*' matches all.

    Left alone, the filter the caller wrote to narrow returns every tool —
    including whichever one they mounted the server for and meant to keep
    out of reach. Refused at the boundary that would do the splitting.
    """
    with pytest.raises(ConfigurationError, match="bare string"):
        _tool_specs("@app.agent('ag')", "get_*")


def test_tool_specs_normalises_any_iterable_to_a_list() -> None:
    entries = ["get_*", "place_order"]
    assert _tool_specs("@app.agent('ag')", entries) == entries
    assert _tool_specs("@app.agent('ag')", tuple(entries)) == entries
    assert _tool_specs("@app.agent('ag')", None) is None
    assert _tool_specs("@app.agent('ag')", []) == []


def test_mcp_server_without_namespace_raises() -> None:
    class NoNs(MCPServer, url="http://x"):
        pass

    with pytest.raises(ConfigurationError, match="namespace="):
        _resolve_tool_filter("ag", [NoNs], {"foo_bar"})


def test_unsupported_entry_raises() -> None:
    with pytest.raises(ConfigurationError, match="not a supported spec"):
        _resolve_tool_filter("ag", [123], {"foo"})


def test_arbitrary_class_is_not_treated_as_callable() -> None:
    # A non-MCPServer class is callable in Python; it must still raise rather
    # than fall through the callable branch to a confusing dead-pattern warning.
    class NotATool:
        pass

    with pytest.raises(ConfigurationError, match="not a supported spec"):
        _resolve_tool_filter("ag", [NotATool], {"foo"})


# ---------------------------------------------------------------------------
# Wiring — through app.lifespan() with a real Agent
# ---------------------------------------------------------------------------


async def test_two_agents_see_disjoint_toolsets_third_sees_all() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def alpha() -> str:
        return "a"

    @app.tool
    async def beta() -> str:
        return "b"

    @app.tool
    async def gamma() -> str:
        return "g"

    @app.agent("x", tools=[alpha, beta])
    async def x(q: str) -> str:  # type: ignore[empty-body]
        """Sees alpha + beta."""

    @app.agent("y", tools=["gamma"])
    async def y(q: str) -> str:  # type: ignore[empty-body]
        """Sees gamma."""

    @app.agent("z")
    async def z(q: str) -> str:  # type: ignore[empty-body]
        """No tools= — sees everything (back-compat)."""

    async with app.lifespan():
        assert await _tool_names(x) == {"alpha", "beta"}
        assert await _tool_names(y) == {"gamma"}
        assert await _tool_names(z) == {"alpha", "beta", "gamma"}


async def test_empty_tools_yields_toolless_agent() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def alpha() -> str:
        return "a"

    @app.agent("mute", tools=[])
    async def mute(q: str) -> str:  # type: ignore[empty-body]
        """Explicitly toolless."""

    async with app.lifespan():
        assert await _tool_names(mute) == set()


async def test_dead_pattern_warns_at_build_time() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def weather() -> str:
        return "w"

    @app.agent("typo", tools=["nonexistant_*"])
    async def typo(q: str) -> str:  # type: ignore[empty-body]
        """A prefix-guessing bug."""

    with pytest.warns(UserWarning, match="matched no available tool"):
        async with app.lifespan():
            assert await _tool_names(typo) == set()


async def test_boot_fails_when_a_filtered_agent_misses_a_server() -> None:
    """The filter resolves once: a server absent from the boot inventory
    would stay filtered out for the App's lifetime."""
    app = App(
        t0=_fake_t0(), servers=[MCPServer(url=_refused_url(), namespace="dark")]
    )

    @app.agent("chat", tools=["dark_probe"])
    async def chat(q: str) -> str:  # type: ignore[empty-body]
        """Needs the dark server."""

    with pytest.raises(T0AgentsError, match="'dark'"):
        async with app.lifespan():
            pass


async def test_a_toolless_agent_boots_beside_a_server_that_never_answered() -> None:
    app = App(
        t0=_fake_t0(), servers=[MCPServer(url=_refused_url(), namespace="dark")]
    )

    @app.agent("mute", tools=[])
    async def mute(q: str) -> str:  # type: ignore[empty-body]
        """Toolless by declaration."""

    async with app.lifespan():
        assert await _tool_names(mute) == set()

        @app.agent("late_mute", tools=[])
        async def late_mute(q: str) -> str:  # type: ignore[empty-body]
            """Toolless, registered after boot."""

        late_spec = app._agent_registry["late_mute"]
        await app._ensure_ready_for(late_spec)
        assert late_spec.resolved_tools == set()


async def test_a_late_filtered_agent_fails_when_a_server_never_answered() -> None:
    app = App(
        t0=_fake_t0(), servers=[MCPServer(url=_refused_url(), namespace="dark")]
    )

    @app.tool
    async def weather() -> str:
        return "w"

    async with app.lifespan():

        @app.agent("late", tools=["dark_probe"])
        async def late(q: str) -> str:  # type: ignore[empty-body]
            """Registered after boot."""

        with pytest.raises(T0AgentsError, match="'dark'"):
            await late("hi")


async def test_boot_without_filters_excludes_a_server_that_never_answered() -> None:
    app = App(
        t0=_fake_t0(), servers=[MCPServer(url=_refused_url(), namespace="dark")]
    )

    @app.tool
    async def weather() -> str:
        return "w"

    @app.agent("chat")
    async def chat(q: str) -> str:  # type: ignore[empty-body]
        """Sees whatever answers."""

    async with app.lifespan():
        assert await _tool_names(chat) == {"weather"}


def test_bare_string_at_the_decorator_fails_on_declaration() -> None:
    """A list that lost its brackets is refused where it is written."""
    app = App(t0=_fake_t0())

    @app.tool
    async def weather() -> str:
        return "w"

    with pytest.raises(ConfigurationError, match="bare string"):
        app.agent("typo", tools="weath*")  # type: ignore[arg-type]


async def test_sub_agent_ask_survives_the_filter() -> None:
    # The filter wraps only the inner real-tool provider; SubAgentToolProvider
    # appends the synthetic ``<name>_ask`` tool *after* it, so delegation is
    # never filtered out even when a real tool (``beta``) is.
    inner = MockToolProvider({"alpha": lambda _a: "a", "beta": lambda _a: "b"})
    helper = Agent(
        AgentConfig(name="helper", function_name="helper_fn"),
        t0=_fake_t0(),
        tool_provider=None,
    )
    provider = SubAgentToolProvider(
        [helper], inner=FilteredToolProvider(inner, {"alpha"})
    )

    names = {tool.name for tool in await provider.list_tools()}
    assert names == {"alpha", "helper_ask"}
