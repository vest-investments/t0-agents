"""``ConversationUsage`` aggregates billing + footprint across Runs.

A conversation is many ``invoke()``s sharing one ``episode_id``.
:class:`ConversationUsage` sums **billing** across the Runs while tracking the
current **footprint** as a *level* (the latest Run's footprint), not a sum —
and carries the level forward over Runs that have no footprint (a failed or
cache-only turn).  These tests drive real Runs (via ``FakeT0``) so the
``Run.usage()`` / ``Run.context_footprint()`` surface M1 exposes is exercised
end-to-end.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents import ConversationUsage
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.run import Run
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


async def _completed_run(
    *,
    input_tokens: int,
    output_tokens: int,
    cost: float | None,
    cache_read: int | None = None,
    cache_write: int | None = None,
    middleware: list[Any] | None = None,
) -> Run[Any]:
    """A single-turn Run carrying the given usage, run to completion.

    One text-only inference → the loop ends after one turn, so
    ``context_footprint()`` is exactly ``input_tokens`` (or ``None`` when that
    is 0) and ``usage()`` is that turn.  ``middleware=[]`` drops
    ``CostTrackingMiddleware`` so ``usage()`` is ``None`` (the no-inference
    shape, for the no-op-record path).  Each Run gets its own (response-issued)
    ``episode_id`` — keying is exercised separately in ``TestEpisodeKeying``.
    """
    t0 = FakeT0(
        [
            FakeChatResponse(
                content=[Text(type="text", text="ok")],
                usage=FakeUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    provider_cache_read_input_tokens=cache_read,
                    provider_cache_write_input_tokens=cache_write,
                ),
            ),
        ],
    )
    agent = Agent(
        AgentConfig(name="c", function_name="agent"),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )
    run = (
        agent.invoke("hi", middleware=middleware)
        if middleware is not None
        else agent.invoke("hi")
    )
    await run.result()
    return run


class TestBillingAccumulates:
    async def test_total_sums_billing_across_runs(self) -> None:
        conv = ConversationUsage("ep-1")
        run1 = await _completed_run(input_tokens=10, output_tokens=5, cost=0.001)
        run2 = await _completed_run(input_tokens=20, output_tokens=2, cost=0.002)

        conv.record(run1)
        conv.record(run2)

        total = conv.total
        assert total.input_tokens == 30
        assert total.output_tokens == 7
        assert total.cost == pytest.approx(0.003)
        assert total.cost_is_priced is True
        assert conv.runs_recorded == 2

    async def test_cache_fields_sum(self) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(
            await _completed_run(
                input_tokens=10, output_tokens=1, cost=0.0,
                cache_read=4, cache_write=2,
            )
        )
        conv.record(
            await _completed_run(
                input_tokens=10, output_tokens=1, cost=0.0,
                cache_read=3, cache_write=1,
            )
        )

        assert conv.total.provider_cache_read_input_tokens == 7
        assert conv.total.provider_cache_write_input_tokens == 3

    async def test_cost_is_priced_is_or_across_runs(self) -> None:
        conv = ConversationUsage("ep-1")
        # An unpriced run (cost=None) followed by a priced one.
        conv.record(await _completed_run(input_tokens=5, output_tokens=1, cost=None))
        assert conv.total.cost_is_priced is False
        conv.record(await _completed_run(input_tokens=5, output_tokens=1, cost=0.0))
        assert conv.total.cost_is_priced is True


class TestFootprintIsLevel:
    async def test_footprint_is_latest_level_not_sum(self) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(await _completed_run(input_tokens=10, output_tokens=5, cost=0.0))
        conv.record(await _completed_run(input_tokens=20, output_tokens=2, cost=0.0))

        footprint = conv.context_footprint()
        assert footprint is not None
        # The latest level (20), NOT the billing sum (30).
        assert footprint.input_tokens == 20
        assert conv.total.input_tokens == 30

    async def test_footprint_carries_forward_over_footprintless_run(self) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(await _completed_run(input_tokens=10, output_tokens=5, cost=0.001))
        # A run whose only inference reports input_tokens == 0 has no footprint
        # (context_footprint() skips zeros) but still has billing.
        conv.record(await _completed_run(input_tokens=0, output_tokens=7, cost=0.002))

        footprint = conv.context_footprint()
        assert footprint is not None
        assert footprint.input_tokens == 10  # carried forward, not blanked
        # Billing still reflects both runs.
        assert conv.total.input_tokens == 10
        assert conv.total.output_tokens == 12
        assert conv.total.cost == pytest.approx(0.003)
        assert conv.runs_recorded == 2


class TestWindowMath:
    async def test_remaining_and_fraction(self) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(await _completed_run(input_tokens=20, output_tokens=2, cost=0.0))

        assert conv.remaining(100) == 80
        assert conv.fraction_used(100) == pytest.approx(0.2)

    async def test_remaining_negative_when_over_window(self) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(await _completed_run(input_tokens=20, output_tokens=2, cost=0.0))

        # Unclamped — the overflow magnitude reaches the caller.
        assert conv.remaining(10) == -10
        assert conv.fraction_used(10) == pytest.approx(2.0)

    @pytest.mark.parametrize("window", [0, -1, -100])
    async def test_fraction_used_non_positive_window_raises(
        self, window: int
    ) -> None:
        conv = ConversationUsage("ep-1")
        conv.record(await _completed_run(input_tokens=20, output_tokens=2, cost=0.0))

        with pytest.raises(ValueError, match="window must be positive"):
            conv.fraction_used(window)


class TestEmptyAndNoop:
    async def test_empty_accumulator(self) -> None:
        conv = ConversationUsage("ep-1")

        assert conv.episode_id == "ep-1"
        assert conv.runs_recorded == 0
        assert conv.context_footprint() is None
        assert conv.remaining(100) is None
        assert conv.fraction_used(100) is None
        # total starts all-zero, unpriced.
        assert conv.total.input_tokens == 0
        assert conv.total.output_tokens == 0
        assert conv.total.cost == 0.0
        assert conv.total.cost_is_priced is False

    async def test_usage_none_run_is_noop_for_billing(self) -> None:
        conv = ConversationUsage("ep-1")
        # middleware=[] drops CostTrackingMiddleware → usage() is None.
        noop_run = await _completed_run(
            input_tokens=99, output_tokens=99, cost=9.9, middleware=[],
        )
        assert noop_run.usage() is None  # precondition

        conv.record(noop_run)
        # No billing folded, no footprint, but it counts as a recorded run.
        assert conv.total.input_tokens == 0
        assert conv.context_footprint() is None
        assert conv.runs_recorded == 1

        # A subsequent real run accumulates normally on top.
        conv.record(await _completed_run(input_tokens=12, output_tokens=3, cost=0.0))
        assert conv.total.input_tokens == 12
        assert conv.runs_recorded == 2


class TestEpisodeKeying:
    async def test_episode_mismatch_is_lenient_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conv = ConversationUsage("ep-A")
        run = await _completed_run(input_tokens=10, output_tokens=1, cost=0.0)
        assert run.episode_id is not None and run.episode_id != "ep-A"

        with caplog.at_level("DEBUG", logger="t0_agents.core.conversation"):
            conv.record(run)

        # Folded in regardless of the episode mismatch...
        assert conv.total.input_tokens == 10
        # ...with a debug breadcrumb naming both episodes.
        assert any(
            "ep-A" in r.message and run.episode_id in r.message
            for r in caplog.records
        )

    async def test_matching_episode_does_not_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        run = await _completed_run(input_tokens=10, output_tokens=1, cost=0.0)
        assert run.episode_id is not None
        conv = ConversationUsage(run.episode_id)

        with caplog.at_level("DEBUG", logger="t0_agents.core.conversation"):
            conv.record(run)

        assert conv.total.input_tokens == 10
        assert caplog.records == []
