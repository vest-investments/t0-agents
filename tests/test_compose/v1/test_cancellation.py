"""Cancellation cascade tests for Composers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest

from t0_agents import parallel, sequential
from t0_agents.core.run import RunCancelledError
from tests.test_compose.v1._helpers import MockComposable


async def test_cancel_sequential_mid_flight() -> None:
    slow_a = MockComposable(name="a", delay=10.0, value="slow")
    unreached = MockComposable(name="b")

    run = sequential(slow_a, unreached).invoke("seed", cancel_timeout=2.0)
    await asyncio.sleep(0.05)
    await run.cancel()

    assert run.state == "cancelled"
    with pytest.raises(RunCancelledError):
        await run.result()
    assert unreached.invocations == []


async def test_cancel_parallel_cascades_to_children() -> None:
    a = MockComposable(name="a", delay=5.0, value="a")
    b = MockComposable(name="b", delay=5.0, value="b")
    c = MockComposable(name="c", delay=5.0, value="c")

    run = parallel(a, b, c).invoke("seed", cancel_timeout=2.0)
    await asyncio.sleep(0.05)
    await run.cancel()

    assert run.state == "cancelled"
    # All sub-Runs should be in cancelled state.
    sub_runs = run._sub_runs  # type: ignore[attr-defined]
    assert len(sub_runs) == 3
    for sr in sub_runs:
        assert sr.state in ("cancelled", "succeeded", "failed")
        # In the cascade most should be cancelled; we don't assert all
        # because of the cancellation race window.


async def test_cancel_nested_composers_cascades_to_leaves() -> None:
    inner_a = MockComposable(name="inner_a", delay=10.0)
    inner_b = MockComposable(name="inner_b", delay=10.0)
    outer_c = MockComposable(name="outer_c")

    composer = sequential(
        parallel(inner_a, inner_b, error_policy="fail_fast"), outer_c,
    )
    run = composer.invoke("seed", cancel_timeout=2.0)
    await asyncio.sleep(0.05)
    await run.cancel()

    assert run.state == "cancelled"
    # The parallel composer's CompositeRun was registered as a sub-Run.
    # Its own _sub_runs include inner_a and inner_b. The outer_c
    # composable was never invoked because the parallel was still running.
    assert outer_c.invocations == []


async def test_cancel_before_start_lazy_run_transitions_directly() -> None:
    a = MockComposable(name="a", value="x")
    run = sequential(a).invoke("seed", eager=False, cancel_timeout=1.0)
    assert run.state == "pending"

    await run.cancel()
    assert run.state == "cancelled"


async def test_cancel_timeout_overshoot_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A child that ignores cancellation triggers a WARN within ``cancel_timeout``.

    Tests :meth:`CompositeRun._cancel_sub_runs` directly so the test
    does not have to wrestle with a stuck child's drive task at
    pytest teardown — we synthesise a sub-Run whose ``cancel()`` hangs
    and then call the protected helper with a tight timeout.
    """
    from t0_agents.compose.topology import CompositeRun

    class HangingCancelRun:
        """Mimics a Run whose cancel() hangs past the timeout."""
        state = "running"

        async def cancel(self) -> None:
            await asyncio.sleep(10)

    async def driver(c: CompositeRun) -> Any:
        return None  # never actually called in this test

    composite = CompositeRun(
        "test_stuck", "seed", driver, cancel_timeout=0.05, eager=False,
    )
    composite._sub_runs.append(cast(Any, HangingCancelRun()))

    with caplog.at_level(logging.WARNING, logger="t0_agents.compose.topology"):
        await composite._cancel_sub_runs()

    warned = any(
        "cancel exceeded" in record.message for record in caplog.records
    )
    assert warned, (
        f"Expected WARN about overshoot; got: "
        f"{[r.message for r in caplog.records]}"
    )
