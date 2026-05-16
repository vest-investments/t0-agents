"""Tests for :func:`t0_agents.map_reduce`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from t0_agents import map_reduce
from t0_agents.compose.topology import ComposerError
from t0_agents.core.types import AgentResult
from tests.test_compose.v1._helpers import MockComposable


def _sum_reducer(results: list[AgentResult[Any]]) -> int:
    return sum(r.value for r in results)


async def test_map_reduce_scatter_then_aggregate() -> None:
    node = MockComposable(name="square", transform=lambda x: x * x)

    composer = map_reduce(node, reducer=_sum_reducer)
    result = await composer.invoke([1, 2, 3, 4]).result()

    assert result.value == 1 + 4 + 9 + 16
    assert node.invocations == [1, 2, 3, 4]


async def test_map_reduce_concurrency_bound() -> None:
    """``concurrency=N`` caps the active call count to N."""
    in_flight: list[int] = []
    peak = {"value": 0}

    class TrackingComposable:
        name = "tracker"

        def invoke(self, input_value: Any, /, **kwargs: Any) -> Any:
            from tests.test_compose.v1._helpers import MockRun

            in_flight.append(input_value)
            peak["value"] = max(peak["value"], len(in_flight))

            class _Run(MockRun):
                async def _drive(self) -> None:
                    await asyncio.sleep(0.05)
                    in_flight.remove(input_value)
                    await super()._drive()

            return _Run(value=input_value * 2)

    node = TrackingComposable()
    items = list(range(10))
    result = await map_reduce(
        node, reducer=_sum_reducer, concurrency=3,
    ).invoke(items).result()

    assert peak["value"] <= 3
    assert result.value == sum(i * 2 for i in items)


async def test_map_reduce_unbounded_concurrency() -> None:
    node = MockComposable(name="x", transform=lambda x: x + 1)
    result = await map_reduce(
        node, reducer=_sum_reducer, concurrency=None,
    ).invoke([1, 2, 3]).result()
    assert result.value == 2 + 3 + 4


async def test_map_reduce_rejects_str_input() -> None:
    node = MockComposable(name="echo")
    with pytest.raises(TypeError, match="iterable"):
        await map_reduce(node, reducer=_sum_reducer).invoke("hello").result()


async def test_map_reduce_rejects_non_iterable() -> None:
    node = MockComposable(name="echo")
    with pytest.raises(TypeError, match="iterable"):
        await map_reduce(node, reducer=_sum_reducer).invoke(42).result()


async def test_map_reduce_empty_iterable() -> None:
    node = MockComposable(name="echo")
    result = await map_reduce(
        node, reducer=lambda rs: "empty-ok" if not rs else "non-empty",
    ).invoke([]).result()
    assert result.value == "empty-ok"


async def test_map_reduce_failure_wraps_with_child_node() -> None:
    node = MockComposable(name="boom", raises=ValueError("item explode"))
    with pytest.raises(ComposerError) as excinfo:
        await map_reduce(node, reducer=_sum_reducer).invoke([1, 2]).result()
    assert excinfo.value.node is node
