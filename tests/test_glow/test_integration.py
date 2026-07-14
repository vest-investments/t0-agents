"""Integration tests for the glow surface (M2 + M3).

Requires the glow gateway up:

    docker compose -f examples/glow/docker-compose.yml --env-file .env up -d

The M2 tests mirror ``examples/glow/01_hello.py`` and
``examples/glow/02_structured.py`` end-to-end.  The M3 test mirrors
``examples/agent/08_multi_agent`` but every node is declared via glow
(``@app.agent`` + ``Depends(<wrapper>)`` for the sub-agent special
case).
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Any, cast

import pytest
from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.glow import App, Depends, regex_guardrail

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


# Module-level guardrail instance so the ``Annotated[str, _NO_SSN]``
# return annotation on the test's inner ``@app.agent``-decorated
# function resolves through ``get_type_hints(fn, ...)``, which walks
# ``fn.__globals__`` only.  An enclosing-test local would be invisible
# to that walk (the same pattern that motivates the module-level MCP
# subclasses in ``test_mcp_typing.py``).
_NO_SSN = regex_guardrail(
    r"\b\d{3}-\d{2}-\d{4}\b",
    "Response contains an SSN-shaped value",
)


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

    Mirrors ``examples/agent/08_multi_agent`` but declares every node via
    glow.  The orchestrator's ``AgentConfig.sub_agents`` is populated
    at decoration time from the ``Depends(<wrapper>)`` defaults; the
    auto-allocated :class:`SubAgentToolProvider` then exposes each as a
    ``<name>_ask`` MCP tool to the orchestrator LLM.
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


async def test_regex_guardrail_on_return_annotation_blocks_response() -> None:
    """``Annotated[str, regex_guardrail(...)]`` on the return type blocks
    the response when the regex matches the LLM's output.

    The system prompt instructs the LLM to include the placeholder
    SSN ``123-45-6789`` verbatim; the guardrail's ``after_run`` hook
    matches the SSN pattern and raises :class:`GuardrailBlockedError`
    via the existing v1.0 ``GuardrailsMiddleware``.
    """
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent("support")
        async def support(query: str) -> Annotated[str, _NO_SSN]:  # type: ignore[empty-body]
            """Return ONLY the literal placeholder SSN ``123-45-6789``
            with no other text."""

        with pytest.raises(GuardrailBlockedError, match="SSN"):
            async with app.lifespan():
                await support("Please return the placeholder SSN now.")


# ---------------------------------------------------------------------------
# M3 (glow v2): model= functionless routing
# ---------------------------------------------------------------------------


async def test_model_agent_runs_functionless_with_tools_episode_and_streaming() -> None:
    """The functionless 4-way combo end-to-end through glow: ``model=``
    (zero TOML) + an ``@app.tool`` + episode continuity + ``events()``
    streaming.  Mirrors the Q19 probe that gated this design.
    """
    from t0_agents.stream.events import DoneEvent, ToolResultEvent

    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.tool
        def get_capital(country: str) -> str:
            """Return the capital city of a country."""
            return {"France": "Paris"}.get(country, "unknown")

        @app.agent("geo", model="groq::openai/gpt-oss-20b")
        async def geo(q: str) -> str:  # type: ignore[empty-body]
            """A geography assistant. Always call the get_capital tool
            before answering questions about capitals."""

        async with app.lifespan():
            first = await geo("Say the word ready.").result()
            assert first.episode_id

            events: list[Any] = []
            follow_up = geo(
                "What is the capital of France? You MUST call the "
                "get_capital tool first."
            ).with_(episode_id=first.episode_id)
            async for event in follow_up.events():
                events.append(event)

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response.strip() != ""
    assert done[0].result.episode_id == first.episode_id
    assert any(isinstance(e, ToolResultEvent) for e in events)


async def test_functionless_inferences_store_under_default_with_auto_tag() -> None:
    """Functionless runs land under ``tensorzero::default`` carrying the
    ``t0_agents.agent`` auto-tag — the per-agent observability handle
    (and the documented ``TensorZeroMemory(tag_filter=...)`` cover for
    the function_name-filter blind spot).
    """
    import asyncio
    from uuid import uuid4

    from tensorzero import InferenceFilterTag, ListInferencesRequest, OrderBy

    agent_name = f"tagcheck-{uuid4().hex[:8]}"
    async with await _gateway_factory() as t0:
        app = App(t0=t0)

        @app.agent(agent_name, model="groq::openai/gpt-oss-20b")
        async def tagged(q: str) -> str:  # type: ignore[empty-body]
            """Reply in one word."""

        async with app.lifespan():
            await tagged("Say ok.")

        await asyncio.sleep(2)  # ClickHouse ingest is asynchronous
        request = ListInferencesRequest(
            function_name="tensorzero::default",
            filters=InferenceFilterTag(
                key="t0_agents.agent",
                value=agent_name,
                comparison_operator="=",
            ),
            order_by=[
                OrderBy(by="timestamp", direction="descending")  # type: ignore[list-item]
            ],
            limit=5,
        )
        stored = await t0.list_inferences(request=request)

    assert len(stored.inferences) >= 1
    assert all(
        inf.function_name == "tensorzero::default" for inf in stored.inferences
    )
