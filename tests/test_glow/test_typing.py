"""Static-typing assertions for the glow surface.

``typing.assert_type`` is a runtime no-op; the real signal is
``uv run mypy tests/test_glow/test_typing.py``.  The async tests also
exercise the asserted values at runtime — gateway-free, via
``ScriptedAgent`` + ``app.agent_overrides`` — so the file both
type-checks and runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, assert_type

from t0_agents.glow import AgentHandle, App, Depends, FluentRun, Text, llm
from t0_agents.stream.events import AgentEvent, TextEndEvent
from t0_agents.testing import ScriptedAgent


async def test_await_of_a_handle_is_typed_as_the_return_annotation() -> None:
    app = App()

    @app.agent("forecaster")
    async def forecaster(query: str) -> str:
        """Forecast the weather."""
        return llm()

    app.agent_overrides["forecaster"] = ScriptedAgent(result="sunny")

    handle = forecaster("weather?")
    assert_type(handle, FluentRun[str])

    answer = await handle
    assert_type(answer, str)
    assert answer == "sunny"


async def test_orchestrator_with_agent_handle_depends_runs_gateway_free() -> None:
    app = App()

    @app.agent("researcher")
    async def researcher(query: str) -> str:
        """Research a topic."""
        return llm()

    @app.agent("orchestrator")
    async def orchestrator(
        query: str, research: AgentHandle[..., str] = Depends(researcher)
    ) -> str:
        """Coordinate research, then formatting."""
        return llm()

    app.agent_overrides["orchestrator"] = ScriptedAgent(result="coordinated")

    answer = await orchestrator("plan a trip")
    assert_type(answer, str)
    assert answer == "coordinated"


if TYPE_CHECKING:
    # Pure static checks (mypy-only; never executed).  events() returns
    # un-consumed async iterators, so these stay out of the runtime path.
    def _events_overloads_narrow_the_element_type(run: FluentRun[str]) -> None:
        assert_type(run.events(), AsyncIterator[AgentEvent])
        assert_type(run.events(Text), AsyncIterator[TextEndEvent])
        assert_type(run.events(Text.map(lambda te: te.id)), AsyncIterator[str])

    def _agent_handle_is_callable_and_returns_a_fluent_run(
        research: AgentHandle[..., str],
    ) -> None:
        assert_type(research("topic"), FluentRun[str])
