"""``CostTrackingMiddleware`` populates ``Run.cost`` / ``Run.usage``.

v0.12 only tracked cost when guardrails were configured (the tracker
was a closure on ``GuardrailContext``).  v1.0's always-on
:class:`CostTrackingMiddleware` exposes totals on the public
:meth:`Run.cost` and :meth:`Run.usage` accessors regardless of
guardrail configuration.  Tests assert the accessors return live data
under every shape (default chain, opt-out, guardrails-only).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text
from tensorzero.types import Usage as TZUsage

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import AgentResult, RunUsage, ToolProvider, Usage
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


def _build_agent(t0: FakeT0) -> Agent:
    return Agent(
        AgentConfig(name="cost", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


def _two_turn_agent(
    usage1: FakeUsage,
    usage2: FakeUsage,
    *,
    middleware: list[Any] | None = None,
) -> Agent:
    """Agent that runs exactly two inferences carrying the given usages.

    Turn 1 emits a tool call (so the loop runs again); turn 2 emits text
    (so the loop ends).  Used by the context-footprint tests to drive a
    multi-turn conversation with controlled per-turn ``input_tokens``.
    ``middleware`` overrides the default chain when given (e.g. to insert
    an observer after ``CostTrackingMiddleware``).
    """
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
            FakeChatResponse(content=[tool_call], usage=usage1),
            FakeChatResponse(
                content=[Text(type="text", text="done")], usage=usage2,
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
    config = (
        AgentConfig(name="fp", function_name="agent", middleware=middleware)
        if middleware is not None
        else AgentConfig(name="fp", function_name="agent")
    )
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client),
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


class TestCostIsPriced:
    """``cost_is_priced`` distinguishes a real $0 from missing pricing."""

    async def test_false_when_all_costs_none(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="local")],
                    usage=FakeUsage(input_tokens=5, output_tokens=2, cost=None),
                ),
            ],
        )
        agent = _build_agent(t0)
        run = agent.invoke("q")
        await run.result()

        usage = run.usage()
        assert usage is not None
        assert usage.cost == 0.0
        assert usage.cost_is_priced is False

    async def test_true_when_a_cost_is_reported(self) -> None:
        # A real 0.0 (a free model that *does* report cost) counts as
        # priced — the whole point of the flag over a bare cost == 0.0.
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(input_tokens=5, output_tokens=2, cost=0.0),
                ),
            ],
        )
        agent = _build_agent(t0)
        run = agent.invoke("q")
        await run.result()

        usage = run.usage()
        assert usage is not None
        assert usage.cost == 0.0
        assert usage.cost_is_priced is True


class TestCacheTokensTracked:
    """Prompt-cache tokens fold into the cumulative ``RunUsage``."""

    async def test_single_inference_cache_tokens_captured(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="cached")],
                    usage=FakeUsage(
                        input_tokens=100,
                        output_tokens=10,
                        cost=0.005,
                        provider_cache_read_input_tokens=80,
                        provider_cache_write_input_tokens=20,
                    ),
                ),
            ],
        )
        agent = _build_agent(t0)

        run = agent.invoke("hi")
        await run.result()

        usage = run.usage()
        assert isinstance(usage, RunUsage)
        assert usage.provider_cache_read_input_tokens == 80
        assert usage.provider_cache_write_input_tokens == 20
        # Cache tokens are a breakdown of input_tokens, not added to it.
        assert usage.input_tokens == 100


class TestExplicitNoMiddleware:
    """``middleware=[]`` opts out of cost-tracking — placeholders stay zero."""

    async def test_empty_middleware_skips_cost_tracker(self) -> None:
        # Power-user opt-out: passing ``middleware=[]`` disables every
        # default middleware, including ``CostTrackingMiddleware``.
        # ``Run.cost`` / ``Run.usage`` remain at their initial values
        # (the same shape they take when no middleware can populate
        # them).
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
                        input_tokens=10,
                        output_tokens=5,
                        cost=0.001,
                        provider_cache_read_input_tokens=4,
                    ),
                ),
                FakeChatResponse(
                    content=[Text(type="text", text="3")],
                    episode_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                    usage=FakeUsage(
                        input_tokens=20,
                        output_tokens=2,
                        cost=0.002,
                        provider_cache_read_input_tokens=8,
                        provider_cache_write_input_tokens=3,
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
        # Cache tokens accumulate across turns just like input/output.
        assert usage.provider_cache_read_input_tokens == 12  # 4 + 8
        assert usage.provider_cache_write_input_tokens == 3  # 0 + 3


class TestContextFootprint:
    """The live per-turn footprint, distinct from the cumulative sum."""

    async def test_footprint_is_latest_turn_not_sum(self) -> None:
        agent = _two_turn_agent(
            FakeUsage(input_tokens=10, output_tokens=5),
            FakeUsage(input_tokens=20, output_tokens=2),
        )
        run = agent.invoke("compute 1+2")
        await run.result()

        footprint = run.context_footprint()
        assert footprint is not None
        # Footprint == the 2nd (latest) inference, NOT the sum.
        assert footprint.input_tokens == 20
        usage = run.usage()
        assert usage is not None
        assert usage.input_tokens == 30  # the sum
        assert footprint.input_tokens < usage.input_tokens

    async def test_usages_lists_every_turn(self) -> None:
        agent = _two_turn_agent(
            FakeUsage(input_tokens=10, output_tokens=5),
            FakeUsage(input_tokens=20, output_tokens=2),
        )
        run = agent.invoke("compute 1+2")
        await run.result()

        usages = run.usages()
        assert len(usages) == 2
        assert [u.input_tokens for u in usages] == [10, 20]

    async def test_cache_hit_zero_does_not_clobber_footprint(self) -> None:
        # 2nd turn is a whole-response cache hit (all-zero usage); the
        # footprint must carry forward to the prior non-zero inference.
        agent = _two_turn_agent(
            FakeUsage(input_tokens=10, output_tokens=5),
            FakeUsage(input_tokens=0, output_tokens=0),
        )
        run = agent.invoke("compute 1+2")
        await run.result()

        footprint = run.context_footprint()
        assert footprint is not None
        assert footprint.input_tokens == 10  # the 1st turn, not the zero turn
        # usages() still records the zero turn for transparency.
        usages = run.usages()
        assert len(usages) == 2
        assert usages[-1].input_tokens == 0

    async def test_context_remaining_allows_negative(self) -> None:
        agent = _two_turn_agent(
            FakeUsage(input_tokens=10, output_tokens=5),
            FakeUsage(input_tokens=20, output_tokens=2),
        )
        run = agent.invoke("compute 1+2")
        await run.result()

        # footprint.input_tokens == 20
        assert run.context_remaining(100) == 80
        # Over-window → negative, not clamped.
        assert run.context_remaining(15) == -5

    async def test_footprint_and_remaining_none_without_capture(self) -> None:
        # middleware=[] disables CostTrackingMiddleware → no usages
        # captured → no footprint, and context_remaining is None.
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="hi")],
                    usage=FakeUsage(input_tokens=5, output_tokens=2),
                ),
            ],
        )
        agent = Agent(
            AgentConfig(name="bare", function_name="agent", middleware=[]),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )
        run = agent.invoke("hi")
        await run.result()

        assert run.usages() == ()
        assert run.context_footprint() is None
        assert run.context_remaining(1000) is None

    async def test_accessors_are_live_mid_run(self) -> None:
        # Gemini review (#67): context_footprint() must reflect each
        # inference as it lands, not only after the chain unwinds.  An
        # inner middleware's on_inference callback runs right after
        # CostTracking's, so it observes the just-flushed footprint.
        from t0_agents.core.middleware import CostTrackingMiddleware
        from t0_agents.core.middleware.base import Next, RunContext

        seen: list[int | None] = []

        class _Probe:
            name = "probe"

            async def __call__(
                self, ctx: RunContext, next_: Next,
            ) -> AgentResult[Any]:
                async def _observe(response: object) -> None:
                    run = ctx.run
                    fp = run.context_footprint() if run is not None else None
                    seen.append(fp.input_tokens if fp is not None else None)

                ctx.chain_on_inference(_observe)
                return await next_(ctx)

        agent = _two_turn_agent(
            FakeUsage(input_tokens=10, output_tokens=5),
            FakeUsage(input_tokens=20, output_tokens=2),
            middleware=[CostTrackingMiddleware(), _Probe()],
        )
        run = agent.invoke("compute 1+2")
        await run.result()

        # Footprint was live at each inference (10 then 20); with a
        # finally-only flush this would have been [None, None].
        assert seen == [10, 20]


class TestRunUsageType:
    """``RunUsage`` is the public type for usage aggregates."""

    def test_runusage_defaults_are_zero(self) -> None:
        u = RunUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cost == 0.0

    def test_runusage_cache_fields_default_zero(self) -> None:
        # New cache fields are additive + defaulted, so v1.0 construction
        # (no cache kwargs) stays valid and back-compatible.
        u = RunUsage()
        assert u.provider_cache_read_input_tokens == 0
        assert u.provider_cache_write_input_tokens == 0

    def test_runusage_cost_is_priced_defaults_false(self) -> None:
        assert RunUsage().cost_is_priced is False

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


class TestUsageValueType:
    """``Usage`` is the per-inference lossless mirror of ``tensorzero.Usage``."""

    def test_from_tz_copies_all_fields(self) -> None:
        raw = TZUsage(
            input_tokens=10,
            output_tokens=5,
            provider_cache_read_input_tokens=3,
            provider_cache_write_input_tokens=1,
            cost=0.5,
        )
        u = Usage.from_tz(raw)
        assert u.input_tokens == 10
        assert u.output_tokens == 5
        assert u.provider_cache_read_input_tokens == 3
        assert u.provider_cache_write_input_tokens == 1
        assert u.cost == 0.5

    def test_from_tz_is_lossless_for_none(self) -> None:
        # cache fields + cost default to None on the SDK type; the mirror
        # must preserve None (distinct from a real 0 / 0.0).
        raw = TZUsage(input_tokens=0, output_tokens=0)
        u = Usage.from_tz(raw)
        assert u.cost is None
        assert u.provider_cache_read_input_tokens is None
        assert u.provider_cache_write_input_tokens is None

    def test_total_tokens_excludes_cache_breakdown(self) -> None:
        u = Usage(
            input_tokens=7,
            output_tokens=3,
            provider_cache_read_input_tokens=5,
        )
        # cache tokens are a breakdown of input_tokens, not an addend.
        assert u.total_tokens == 10

    def test_usage_is_frozen(self) -> None:
        import dataclasses

        u = Usage(input_tokens=1, output_tokens=1)
        try:
            u.input_tokens = 99  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        # pragma: no cover — defensive fallback if behaviour changes
        raise AssertionError("Usage should be frozen")


