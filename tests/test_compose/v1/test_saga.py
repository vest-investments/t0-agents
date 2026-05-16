"""Tests for :func:`t0_agents.saga`."""

from __future__ import annotations

import pytest

from t0_agents import saga
from t0_agents.compose.topology import ComposerError
from tests.test_compose.v1._helpers import MockComposable


async def test_saga_all_actions_succeed_returns_final_value() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")
    c = MockComposable(name="c", value="C")
    comp_a = MockComposable(name="comp_a", value="comp_a_done")
    comp_b = MockComposable(name="comp_b", value="comp_b_done")
    comp_c = MockComposable(name="comp_c", value="comp_c_done")

    result = await saga(
        (a, comp_a), (b, comp_b), (c, comp_c),
    ).invoke("seed").result()

    assert result.value == "C"
    # No compensation should have fired.
    assert comp_a.invocations == []
    assert comp_b.invocations == []
    assert comp_c.invocations == []


async def test_saga_step_failure_triggers_reverse_compensation() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")
    boom = MockComposable(name="boom", raises=RuntimeError("step 3 failed"))
    comp_a = MockComposable(name="comp_a")
    comp_b = MockComposable(name="comp_b")
    comp_boom = MockComposable(name="comp_boom")

    order: list[str] = []
    comp_a._transform = lambda x: order.append("a") or x  # type: ignore[assignment]
    comp_b._transform = lambda x: order.append("b") or x  # type: ignore[assignment]

    with pytest.raises(ComposerError) as excinfo:
        await saga(
            (a, comp_a), (b, comp_b), (boom, comp_boom),
        ).invoke("seed").result()
    assert excinfo.value.node is boom

    # Reverse order: comp_b first, then comp_a.
    assert order == ["b", "a"]
    # comp_boom never runs because boom never succeeded.
    assert comp_boom.invocations == []


async def test_saga_none_compensate_skipped() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")
    boom = MockComposable(name="boom", raises=ValueError("nope"))
    comp_a = MockComposable(name="comp_a")

    with pytest.raises(ComposerError):
        await saga(
            (a, comp_a), (b, None), (boom, None),
        ).invoke("seed").result()

    # Only comp_a ran; the None compensate for b was skipped.
    assert comp_a.invocations == ["seed"]


async def test_saga_compensate_receives_same_input_as_action() -> None:
    a = MockComposable(name="a", value="A_OUT")
    b = MockComposable(name="b", value="B_OUT")
    boom = MockComposable(name="boom", raises=RuntimeError("die"))
    comp_a = MockComposable(name="comp_a")
    comp_b = MockComposable(name="comp_b")

    with pytest.raises(ComposerError):
        await saga(
            (a, comp_a), (b, comp_b), (boom, None),
        ).invoke("seed_input").result()

    # comp_a received the same input action a received (seed_input).
    assert comp_a.invocations == ["seed_input"]
    # comp_b received the same input action b received (chain of a's value).
    assert comp_b.invocations == ["A_OUT"]


async def test_saga_failing_compensate_logged_not_blocking() -> None:
    a = MockComposable(name="a", value="A")
    b = MockComposable(name="b", value="B")
    boom = MockComposable(name="boom", raises=RuntimeError("die"))
    comp_a = MockComposable(name="comp_a")
    comp_b_broken = MockComposable(name="comp_b_broken", raises=ValueError("rollback failed"))

    with pytest.raises(ComposerError) as excinfo:
        await saga(
            (a, comp_a), (b, comp_b_broken), (boom, None),
        ).invoke("seed").result()
    assert excinfo.value.node is boom

    # comp_a still ran, even though comp_b_broken raised.
    assert comp_a.invocations == ["seed"]
    assert comp_b_broken.invocations == ["A"]


async def test_saga_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"saga.*at least one"):
        await saga().invoke("seed").result()
