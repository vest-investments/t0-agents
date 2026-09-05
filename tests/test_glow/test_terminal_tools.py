"""Unit tests for ``@app.agent(terminal_tools=[...])``.

Gateway-free: the pure diagnostic ``_warn_unmatched_terminal_tools`` is driven
with a synthetic live tool-name set, and the wiring runs through
``app.lifespan()`` over an in-memory ``@app.tool`` surface with the real
:class:`Agent` (``t0`` is a bare sentinel — :meth:`Agent.__init__` never
touches it), so what is asserted is the config the built agent actually carries.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App
from t0_agents.glow.app import _warn_unmatched_terminal_tools


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


def _terminal_tools(wrapper: Any) -> tuple[str, ...]:
    return cast(tuple[str, ...], wrapper.agent._config.terminal_tools)


# ---------------------------------------------------------------------------
# Diagnostic — pure, no gateway
# ---------------------------------------------------------------------------


def test_live_glob_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unmatched_terminal_tools("ag", ("present_*",), {"present_chart"})


def test_dead_pattern_warns_naming_candidates() -> None:
    with pytest.warns(UserWarning, match="matched no available tool"):
        _warn_unmatched_terminal_tools("ag", ("presnt_*",), {"present_chart"})


def test_each_dead_pattern_warns_once() -> None:
    with pytest.warns(UserWarning) as caught:
        _warn_unmatched_terminal_tools(
            "ag", ("present_*", "typo_a", "typo_b"), {"present_chart"}
        )
    assert len(caught) == 2


def test_no_terminal_tools_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unmatched_terminal_tools("ag", (), {"present_chart"})


# ---------------------------------------------------------------------------
# Wiring — through app.lifespan() with a real Agent
# ---------------------------------------------------------------------------


async def test_declaration_reaches_the_built_agent_config() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def present_chart() -> str:
        return "rendered"

    @app.agent("advisor", terminal_tools=["present_*"])
    async def advisor(q: str) -> str:  # type: ignore[empty-body]
        """Ends its turn on a render."""

    async with app.lifespan():
        assert _terminal_tools(advisor) == ("present_*",)


async def test_a_declared_list_is_normalised_to_a_tuple() -> None:
    """AgentConfig.terminal_tools is a tuple; the decorator accepts any list."""
    app = App(t0=_fake_t0())

    @app.tool
    async def present_chart() -> str:
        return "rendered"

    @app.tool
    async def present_quick_replies() -> str:
        return "pills"

    @app.agent("advisor", terminal_tools=["present_chart", "present_quick_replies"])
    async def advisor(q: str) -> str:  # type: ignore[empty-body]
        """Ends its turn on a render."""

    async with app.lifespan():
        assert _terminal_tools(advisor) == (
            "present_chart",
            "present_quick_replies",
        )


async def test_omitting_the_knob_leaves_the_agent_unchanged() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def present_chart() -> str:
        return "rendered"

    @app.agent("advisor")
    async def advisor(q: str) -> str:  # type: ignore[empty-body]
        """No terminal tools."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        async with app.lifespan():
            assert _terminal_tools(advisor) == ()


async def test_dead_pattern_warns_at_build_time() -> None:
    app = App(t0=_fake_t0())

    @app.tool
    async def present_chart() -> str:
        return "rendered"

    @app.agent("typo", terminal_tools=["presnt_*"])
    async def typo(q: str) -> str:  # type: ignore[empty-body]
        """A prefix-guessing bug."""

    with pytest.warns(UserWarning, match="terminal_tools= entry"):
        async with app.lifespan():
            assert _terminal_tools(typo) == ("presnt_*",)


async def test_a_bare_string_is_refused_at_the_decorator() -> None:
    """`tuple("present_*")` would shatter into characters, and the '*' among
    them matches every tool — silently making the whole surface terminal."""
    app = App(t0=_fake_t0())

    with pytest.raises(ConfigurationError, match="bare string"):

        @app.agent("advisor", terminal_tools="present_*")  # type: ignore[arg-type]
        async def advisor(q: str) -> str:  # type: ignore[empty-body]
            """Ends its turn on a render."""
