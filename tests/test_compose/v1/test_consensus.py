"""Tests for :func:`t0_agents.consensus`."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import pytest

from t0_agents import consensus
from t0_agents.compose.topology import ComposerError
from tests.test_compose.v1._helpers import MockComposable


def _majority(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


async def test_consensus_majority_vote() -> None:
    a = MockComposable(name="a", value="positive")
    b = MockComposable(name="b", value="positive")
    c = MockComposable(name="c", value="negative")

    result = await consensus(a, b, c, merge=_majority).invoke("seed").result()
    assert result.value == "positive"


async def test_consensus_tie_resolved_by_merge() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")

    # The user-provided merge wins ties by alphabetic order.
    def tie_break(values: list[str]) -> str:
        return sorted(values)[0]

    result = await consensus(a, b, merge=tie_break).invoke("seed").result()
    assert result.value == "A"


async def test_consensus_failure_propagates_as_composer_error() -> None:
    a = MockComposable(name="a", value="ok")
    boom = MockComposable(name="boom", raises=RuntimeError("vote crashed"))

    with pytest.raises(ComposerError) as excinfo:
        await consensus(a, boom, merge=_majority).invoke("seed").result()
    assert excinfo.value.node is boom
    assert isinstance(excinfo.value.cause, RuntimeError)


async def test_consensus_merge_exception_wraps_with_composer_self() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")

    def bad_merge(values: list[Any]) -> Any:
        raise ValueError("can't merge")

    composer = consensus(a, b, merge=bad_merge)
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke("seed").result()
    assert excinfo.value.node is composer


async def test_consensus_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"consensus.*at least one"):
        await consensus(merge=_majority).invoke("seed").result()


async def test_consensus_fail_fast_does_not_wait_for_slow_voters() -> None:
    """consensus surfaces failure without waiting for slow proposers.

    Regression guard for the gemini-code-assist PR #33 review:
    consensus has no ``continue`` policy — any sub-Run failure aborts
    the vote.  The driver must therefore cancel siblings the moment
    one proposer raises, rather than waiting on all proposers to
    settle.
    """
    fast_boom = MockComposable(
        name="boom", raises=RuntimeError("vote crashed"), delay=0.01,
    )
    slow_a = MockComposable(name="slow_a", value="A", delay=5.0)
    slow_b = MockComposable(name="slow_b", value="B", delay=5.0)

    loop = asyncio.get_event_loop()
    start = loop.time()
    with pytest.raises(ComposerError):
        await consensus(
            fast_boom, slow_a, slow_b, merge=_majority, cancel_timeout=2.0,
        ).invoke("seed").result()
    elapsed = loop.time() - start
    assert elapsed < 2.5, (
        f"consensus fail-fast waited too long: {elapsed:.2f}s"
    )
