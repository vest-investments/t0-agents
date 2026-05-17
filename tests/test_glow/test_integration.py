"""Integration tests for the M2 glow surface.

Requires the glow gateway up:

    docker compose -f examples/glow/docker-compose.yml --env-file .env up -d

The two tests mirror the two M2 examples
(``examples/glow/01_hello.py`` and ``examples/glow/02_structured.py``)
end-to-end: hello-world chat and structured-output Pydantic return.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import cast

import pytest
from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


class _TripPlan(BaseModel):
    destinations: list[str]
    cost_usd: float


async def test_hello_world_chat_agent_runs_end_to_end() -> None:
    """Chat agent returns a non-empty text response under ``lifespan()``.

    Exercises the M2 + M3 end-to-end path: ``@app.agent`` registers,
    the wrapper compiles to ``Agent(t0=t0, tool_provider=app._owned_client)``,
    ``app.lifespan()`` connects the App-owned FastMCP client (M3),
    and the LLM produces text that flows back through ``FluentRun``.
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("tutor")
        async def tutor(query: str) -> str:  # type: ignore[empty-body]
            """An English tutor that explains grammar like you're 12."""

        async with app.lifespan():
            response = await tutor("Say the word hello.")

    assert isinstance(response, str)
    assert response.strip() != ""


async def test_structured_output_returns_pydantic_instance() -> None:
    """Pydantic return annotation → ``StructuredOutput`` → typed result.

    Uses ``function_name="plan_trip"`` (JSON-strict TZ function
    defined in ``examples/glow/config/tensorzero.toml``).  The schema
    matches :class:`_TripPlan` so the inferred ``StructuredOutput``
    materialises a Pydantic instance the caller can use directly.
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("planner", function_name="plan_trip")
        async def plan(query: str) -> _TripPlan:  # type: ignore[empty-body]
            """Plan trips with realistic budgets."""

        async with app.lifespan():
            trip = await plan("3 days in Japan under $2000")

    assert isinstance(trip, _TripPlan)
    assert len(trip.destinations) > 0
    assert trip.cost_usd > 0
