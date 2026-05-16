"""Tests for :func:`t0_agents.race`."""

from __future__ import annotations

import asyncio

import pytest

from t0_agents import race
from t0_agents.compose.topology import ComposerError
from tests.test_compose.v1._helpers import MockComposable


async def test_race_first_completed_wins() -> None:
    fast = MockComposable(name="fast", value="fast_won", delay=0.01)
    slow = MockComposable(name="slow", value="slow_won", delay=1.0)

    result = await race(fast, slow).invoke("seed", cancel_timeout=1.0).result()
    assert result.value == "fast_won"


async def test_race_loser_cancelled_when_cancel_others_true() -> None:
    fast = MockComposable(name="fast", value="fast_won", delay=0.01)
    slow = MockComposable(name="slow", value="slow_won", delay=1.0)

    run = race(fast, slow, cancel_others=True).invoke("seed", cancel_timeout=1.0)
    await run.result()

    # The Run sub_runs include both fast (succeeded) and slow (cancelled).
    sub_runs = run._sub_runs  # type: ignore[attr-defined]
    states = [sr.state for sr in sub_runs]
    assert "succeeded" in states
    assert "cancelled" in states


async def test_race_cancel_others_false_lets_losers_run() -> None:
    fast = MockComposable(name="fast", value="fast_won", delay=0.01)
    slow = MockComposable(name="slow", value="slow_won", delay=0.1)

    run = race(fast, slow, cancel_others=False).invoke("seed")
    result = await run.result()
    assert result.value == "fast_won"

    # Let the slow Run finish in the background.
    await asyncio.sleep(0.2)
    sub_runs = run._sub_runs  # type: ignore[attr-defined]
    # Both finished successfully — neither was cancelled.
    assert all(sr.state == "succeeded" for sr in sub_runs)


async def test_race_winner_failure_wraps_as_composer_error() -> None:
    """If the first task to settle is a failure, ComposerError surfaces it."""
    boom_fast = MockComposable(
        name="boom", raises=ValueError("first to fail"), delay=0.01,
    )
    slow_ok = MockComposable(name="slow", value="ok", delay=2.0)

    composer = race(boom_fast, slow_ok)
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke("seed", cancel_timeout=1.0).result()

    assert excinfo.value.node is boom_fast
    assert isinstance(excinfo.value.cause, ValueError)


async def test_race_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"race.*at least one"):
        await race().invoke("seed").result()
