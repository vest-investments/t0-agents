"""``App.agent_overrides`` swaps a glow agent by name, with no gateway."""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, RunError, Text, ToolCall
from t0_agents.stream.events import (
    RunErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallEvent,
)
from t0_agents.testing import ScriptedAgent


async def test_override_swaps_agent_by_name_without_a_gateway() -> None:
    app = App()  # no t0 — the only agent is overridden below

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    app.agent_overrides["forecaster"] = ScriptedAgent(result="22C")

    assert await forecaster("weather?") == "22C"


async def test_overridden_agent_events_flow_through_selectors() -> None:
    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    app.agent_overrides["forecaster"] = ScriptedAgent(
        result="sunny",
        events=[
            ToolCallEvent(id="c1", name="get_weather", arguments={}),
            TextDeltaEvent(id="t1", delta="sunny"),
            TextEndEvent(id="t1"),
        ],
    )

    statuses = [
        s
        async for s in forecaster("q").events(
            ToolCall.map(lambda tc: f"Calling {tc.name}")
        )
    ]
    assert statuses == ["Calling get_weather"]

    # v2 retarget: ``Text`` selects the delta events that carry text.
    text_events = [type(e).__name__ async for e in forecaster("q").events(Text)]
    assert text_events == ["TextDeltaEvent"]


async def test_unoverridden_agent_without_a_gateway_fails_loudly() -> None:
    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    with pytest.raises(ConfigurationError, match="no gateway"):
        forecaster("q")


async def test_glow_events_never_raise_mid_stream() -> None:
    """The glow on_error="event" default end-to-end: a failing run ends
    with one terminal RunErrorEvent instead of raising."""
    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    partial = TextDeltaEvent(id="t1", delta="par")
    app.agent_overrides["forecaster"] = ScriptedAgent(
        events=[partial], raises=ValueError("gateway died")
    )

    events = [e async for e in forecaster("q").events()]

    assert events[0] == partial
    terminal = events[-1]
    assert isinstance(terminal, RunErrorEvent)
    assert terminal.error_type == "ValueError"
    assert len([e for e in events if isinstance(e, RunErrorEvent)]) == 1

    # The selector path filters but never raises either.
    errors = [e async for e in forecaster("q").events(RunError)]
    assert len(errors) == 1 and isinstance(errors[0], RunErrorEvent)


async def test_glow_stream_text_covers_the_reasoning_path() -> None:
    from t0_agents.core.types import AgentResult

    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    app.agent_overrides["forecaster"] = ScriptedAgent(
        result=AgentResult(value="It's 22°C.", turns=1, response_source="thought"),
    )

    chunks = [c async for c in forecaster("q").stream_text()]
    assert chunks == ["It's 22°C."]


async def test_glow_statuses_interleaves_hints() -> None:
    from t0_agents.stream.status import Status

    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    app.agent_overrides["forecaster"] = ScriptedAgent(
        result="sunny",
        events=[ToolCallEvent(id="c1", name="get_weather", arguments={})],
    )

    hints = [i for i in [i async for i in forecaster("q").statuses()] if isinstance(i, Status)]
    assert hints == [
        Status(text="Calling get_weather…", server=None, tool="get_weather")
    ]
