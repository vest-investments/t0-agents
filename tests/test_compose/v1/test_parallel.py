"""Tests for :func:`t0_agents.parallel`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from t0_agents import parallel
from t0_agents.compose.topology import ComposerError
from t0_agents.core.types import AgentResult
from tests.test_compose.v1._helpers import MockComposable


async def test_parallel_default_reducer_collects_values() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")
    c = MockComposable(name="c", value="C")

    result = await parallel(a, b, c).invoke("seed").result()

    assert sorted(result.value) == ["A", "B", "C"]
    assert a.invocations == ["seed"]
    assert b.invocations == ["seed"]
    assert c.invocations == ["seed"]


async def test_parallel_custom_reducer() -> None:
    a = MockComposable(name="a", value=10)
    b = MockComposable(name="b", value=20)
    c = MockComposable(name="c", value=30)

    def sum_reducer(results: list[AgentResult[Any]]) -> int:
        return sum(r.value for r in results)

    result = await parallel(a, b, c, reducer=sum_reducer).invoke("x").result()
    assert result.value == 60


async def test_parallel_fail_fast_raises_composer_error_with_failing_node() -> None:
    ok = MockComposable(name="ok", value="ok")
    boom = MockComposable(name="boom", raises=ValueError("explosion"))

    with pytest.raises(ComposerError) as excinfo:
        await parallel(ok, boom).invoke("seed").result()

    assert excinfo.value.node is boom
    assert isinstance(excinfo.value.cause, ValueError)


async def test_parallel_continue_reducer_sees_mixed_list() -> None:
    a = MockComposable(name="a", value=1)
    boom = MockComposable(name="boom", raises=RuntimeError("nope"))
    c = MockComposable(name="c", value=3)

    def reducer(results: list[Any]) -> dict[str, Any]:
        ok = [r.value for r in results if not isinstance(r, BaseException)]
        errs = [type(r).__name__ for r in results if isinstance(r, BaseException)]
        return {"ok": ok, "errors": errs}

    result = await parallel(
        a, boom, c, reducer=reducer, error_policy="continue",
    ).invoke("seed").result()

    assert result.value == {"ok": [1, 3], "errors": ["RuntimeError"]}


async def test_parallel_reducer_exception_wraps_as_composer_error() -> None:
    a = MockComposable(name="a", value=1)
    b = MockComposable(name="b", value=2)

    def bad_reducer(results: list[AgentResult[Any]]) -> Any:
        raise TypeError("reducer broke")

    composer = parallel(a, b, reducer=bad_reducer)
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke("seed").result()

    assert excinfo.value.node is composer
    assert isinstance(excinfo.value.cause, TypeError)


async def test_parallel_inference_ids_aggregated() -> None:
    a = MockComposable(name="a", value="x", inference_ids=["a1"])
    b = MockComposable(name="b", value="y", inference_ids=["b1", "b2"])

    result = await parallel(a, b).invoke("seed").result()
    assert sorted(result.inference_ids) == ["a1", "b1", "b2"]


async def test_parallel_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"parallel.*at least one"):
        await parallel().invoke("seed").result()


async def test_parallel_fail_fast_does_not_wait_for_slow_siblings() -> None:
    """fail_fast must surface the exception without waiting for slow peers.

    Regression guard for the gemini-code-assist PR #33 review: the
    previous implementation used ``asyncio.gather(...,
    return_exceptions=True)`` which waited for every sub-Run before
    propagating; the fix uses ``asyncio.wait(...,
    return_when=FIRST_EXCEPTION)`` so the cascade cancels siblings
    the moment one child raises.
    """
    fast_boom = MockComposable(
        name="boom", raises=ValueError("nope"), delay=0.01,
    )
    slow_a = MockComposable(name="slow_a", value="A", delay=5.0)
    slow_b = MockComposable(name="slow_b", value="B", delay=5.0)

    loop = asyncio.get_event_loop()
    start = loop.time()
    with pytest.raises(ComposerError):
        await parallel(
            fast_boom, slow_a, slow_b, cancel_timeout=2.0,
        ).invoke("seed").result()
    elapsed = loop.time() - start
    assert elapsed < 2.5, (
        f"fail_fast waited too long: {elapsed:.2f}s "
        "(should surface before slow siblings complete)"
    )
