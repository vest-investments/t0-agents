"""Integration tests for the glow surface (M2 + M3).

Requires the glow gateway up:

    docker compose -f examples/glow/docker-compose.yml --env-file .env up -d

The M2 tests mirror ``examples/glow/01_hello.py`` and
``examples/glow/02_structured.py`` end-to-end.  The M3 test mirrors
``examples/08_multi_agent`` but every node is declared via glow
(``@app.agent`` + ``Depends(<wrapper>)`` for the sub-agent special
case).
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, Depends, compose

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


async def test_orchestrator_with_glow_sub_agents() -> None:
    """Orchestrator with ``Depends(researcher), Depends(writer)`` end-to-end.

    Mirrors ``examples/08_multi_agent`` but declares every node via
    glow.  The orchestrator's ``AgentConfig.sub_agents`` is populated
    at decoration time from the ``Depends(<wrapper>)`` defaults; v1.0
    ``install_sub_agents`` registers each as a ``<name>_ask`` MCP tool
    on the App-owned FastMCP server (because every glow agent shares
    ``tool_provider=app._owned_client``, see commit 2).
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("researcher")
        async def researcher(query: str) -> str:  # type: ignore[empty-body]
            """Summarize the requested topic in a single sentence."""

        @app.agent("writer")
        async def writer(query: str) -> str:  # type: ignore[empty-body]
            """Format the provided summary as a single markdown bullet."""

        @app.agent("orchestrator", function_name="orchestrator")
        async def orchestrator(  # type: ignore[empty-body]
            query: str,
            research: Any = Depends(researcher),
            write: Any = Depends(writer),
        ) -> str:
            """Coordinate research then formatting via the available _ask tools."""

        async with app.lifespan():
            response = await orchestrator(
                "Write a one-bullet markdown summary of photosynthesis. "
                "Use researcher_ask to research first, then writer_ask to "
                "format the result."
            )

    assert isinstance(response, str)
    assert response.strip() != ""

    # Sub-agents were constructed depth-first as part of the lifespan
    # build, so the orchestrator's spec holds them in
    # ``sub_agent_specs`` with populated ``agent`` slots.
    orch_spec = orchestrator._spec
    assert len(orch_spec.sub_agent_specs) == 2
    assert {s.name for s in orch_spec.sub_agent_specs} == {"researcher", "writer"}
    for sub in orch_spec.sub_agent_specs:
        assert sub.agent is not None


async def test_llm_routed_dict_composer_dispatches_to_picked_route() -> None:
    """``compose({"math": ..., "code": ...})`` dispatches via the synthetic
    classifier and runs the picked route end-to-end.

    The synthetic classifier uses ``ToolCallOutput(tool_choice="required")``
    with one no-op FastMCP tool per route key; the LLM picks one, the
    loop terminates (``expects_tool_loop=False``), and the picked route
    receives the original user query.

    Each route prepends a distinct sentinel to its response so we can
    assert which branch ran without inspecting CompositeRun internals.
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("math_route")
        async def math_route(query: str) -> str:  # type: ignore[empty-body]
            """Math expert.  Always begin your reply with the exact token
            'MATH:'."""

        @app.agent("code_route")
        async def code_route(query: str) -> str:  # type: ignore[empty-body]
            """Python coder.  Always begin your reply with the exact token
            'CODE:'."""

        async with app.lifespan():
            router = compose({"math_route": math_route, "code_route": code_route})

            math_run = router.invoke("Solve x squared equals four.")
            math_result = await math_run.result()
            code_run = router.invoke(
                "Write a Python function that prints hello world."
            )
            code_result = await code_run.result()

    math_text = str(math_result.value)
    code_text = str(code_result.value)
    assert "MATH:" in math_text, f"math route did not run; got {math_text!r}"
    assert "CODE:" in code_text, f"code route did not run; got {code_text!r}"


async def test_llm_routed_dict_composer_supports_composer_routes() -> None:
    """Composer-as-route case: one route is itself a compose([...]).

    Proves the v1.0 ``Composable`` protocol carries through the dict
    branch — the picked route can be a nested Composer (here a tiny
    sequential pipeline), not just a leaf Agent.
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("step_one")
        async def step_one(query: str) -> str:  # type: ignore[empty-body]
            """Summarise the input in one short sentence."""

        @app.agent("step_two")
        async def step_two(query: str) -> str:  # type: ignore[empty-body]
            """Take the previous summary and prepend 'PIPELINE:' to it."""

        @app.agent("trivial_route")
        async def trivial_route(query: str) -> str:  # type: ignore[empty-body]
            """Reply with the single word 'SINGLE'."""

        async with app.lifespan():
            pipeline = compose([step_one, step_two])
            router = compose({
                "pipeline_route": pipeline,
                "trivial_route": trivial_route,
            })

            run = router.invoke("Briefly describe photosynthesis.")
            result = await run.result()

    text = str(result.value)
    # Whichever route the LLM picks, the result is a string; the
    # acceptance criterion is that the dispatch succeeded without
    # raising ``ComposerError`` for an unknown route key.
    assert isinstance(text, str)
    assert text.strip() != ""
