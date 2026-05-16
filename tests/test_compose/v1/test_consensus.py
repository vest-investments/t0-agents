"""Tests for :func:`t0_agents.consensus`."""

from __future__ import annotations

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
