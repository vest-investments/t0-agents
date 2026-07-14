"""``App.agent_overrides`` swaps a glow agent by name, with no gateway."""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, Text, ToolCall
from t0_agents.stream.events import TextEndEvent, ToolCallEvent
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

    text_events = [type(e).__name__ async for e in forecaster("q").events(Text)]
    assert text_events == ["TextEndEvent"]


async def test_unoverridden_agent_without_a_gateway_fails_loudly() -> None:
    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""

    with pytest.raises(ConfigurationError, match="no gateway"):
        forecaster("q")
