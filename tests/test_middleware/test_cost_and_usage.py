"""``CostTrackingMiddleware`` populates ``Run.cost`` / ``Run.usage``.

v0.12 only tracked cost when guardrails were configured (the tracker
was a closure on ``GuardrailContext``).  v1.0's always-on
:class:`CostTrackingMiddleware` exposes totals on the public
:meth:`Run.cost` and :meth:`Run.usage` accessors regardless of
guardrail configuration.  Tests assert the M1 D-5 carry-forward is
honoured (placeholders now return live data).
"""

from __future__ import annotations

from typing import cast
from uuid import UUID

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import RunUsage, ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


def _build_agent(t0: FakeT0) -> Agent:
    return Agent(
        AgentConfig(name="cost", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


class TestSingleInference:
    async def test_cost_and_usage_populated_after_run(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="hi")],
                    usage=FakeUsage(
                        input_tokens=10, output_tokens=5, cost=0.0012,
                    ),
                ),
            ],
        )
        agent = _build_agent(t0)

        run = agent.invoke("hello")
        await run.result()

        assert run.cost() == 0.0012
        usage = run.usage()
        assert isinstance(usage, RunUsage)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.cost == 0.0012


class TestNoGuardrailsStillTracked:
    """v1.0 enhancement: no-guardrails runs now expose cost / usage."""

    async def test_run_with_no_guardrails_still_has_cost(self) -> None:
        # In v0.12 this would have returned 0.0 because the cost tracker
        # only fired when `_make_cost_tracker(guard_ctx)` was wired.  In
        # v1.0 the always-on ``CostTrackingMiddleware`` populates the
        # accessors regardless of guardrail configuration.
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(
                        input_tokens=7, output_tokens=3, cost=0.0007,
                    ),
                ),
            ],
        )
        agent = _build_agent(t0)

        run = agent.invoke("go")
        await run.result()

        assert run.cost() > 0.0
        assert run.usage() is not None


class TestMissingPricing:
    """Local providers may omit ``cost``; treat ``None`` as zero."""

    async def test_cost_none_treated_as_zero(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="local")],
                    usage=FakeUsage(
                        input_tokens=5, output_tokens=2, cost=None,
                    ),
                ),
            ],
        )
        agent = _build_agent(t0)

        run = agent.invoke("query")
        await run.result()

        assert run.cost() == 0.0
        usage = run.usage()
        assert isinstance(usage, RunUsage)
        assert usage.input_tokens == 5
        assert usage.output_tokens == 2


class TestExplicitNoMiddleware:
    """``middleware=[]`` opts out of cost-tracking — placeholders stay zero."""

    async def test_empty_middleware_skips_cost_tracker(self) -> None:
        # Power-user opt-out: passing ``middleware=[]`` disables every
        # default middleware, including ``CostTrackingMiddleware``.
        # ``Run.cost`` / ``Run.usage`` remain at their initial values
        # (the same shape M1 returned before the M2 wiring).
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="hi")],
                    usage=FakeUsage(
                        input_tokens=42, output_tokens=42, cost=99.0,
                    ),
                ),
            ],
        )
        agent = Agent(
            AgentConfig(
                name="bare",
                function_name="agent",
                middleware=[],  # explicit "no middleware"
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        run = agent.invoke("hello")
        await run.result()

        assert run.cost() == 0.0
        assert run.usage() is None


class TestMultipleInferencesAccumulate:
    """Tool-loop runs with multiple inferences accumulate usage across all turns."""

    async def test_two_turns_sum_usage(self) -> None:
        from fastmcp.client.client import CallToolResult
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool
        from tensorzero.types import ToolCall

        tool_call = ToolCall(
            id="tc-1",
            raw_name="add",
            name="add",
            raw_arguments='{"a": 1, "b": 2}',
            arguments={"a": 1, "b": 2},
        )
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[tool_call],
                    episode_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    usage=FakeUsage(
                        input_tokens=10, output_tokens=5, cost=0.001,
                    ),
                ),
                FakeChatResponse(
                    content=[Text(type="text", text="3")],
                    episode_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    usage=FakeUsage(
                        input_tokens=20, output_tokens=2, cost=0.002,
                    ),
                ),
            ],
        )
        client = FakeMCPClient(
            tools=[
                MCPTool(
                    name="add",
                    description="add two ints",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                    },
                ),
            ],
            call_results={
                "add": CallToolResult(
                    content=[TextContent(type="text", text="3")],
                    structured_content=None,
                    meta=None,
                    data=None,
                    is_error=False,
                ),
            },
        )
        agent = Agent(
            AgentConfig(name="acc", function_name="agent"),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, client),
        )

        run = agent.invoke("compute 1+2")
        await run.result()

        # 0.001 + 0.002 == 0.003.  Use approx-compare to dodge float
        # precision artefacts.
        assert abs(run.cost() - 0.003) < 1e-9
        usage = run.usage()
        assert isinstance(usage, RunUsage)
        assert usage.input_tokens == 30
        assert usage.output_tokens == 7
        assert abs(usage.cost - 0.003) < 1e-9


class TestRunUsageType:
    """``RunUsage`` is the public type for usage aggregates."""

    def test_runusage_defaults_are_zero(self) -> None:
        u = RunUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cost == 0.0

    def test_runusage_is_frozen(self) -> None:
        u = RunUsage(input_tokens=1, output_tokens=2, cost=0.5)
        # frozen=True ⇒ mutation raises FrozenInstanceError.
        import dataclasses

        try:
            u.input_tokens = 99  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        # pragma: no cover — defensive fallback if behaviour changes
        raise AssertionError("RunUsage should be frozen")


