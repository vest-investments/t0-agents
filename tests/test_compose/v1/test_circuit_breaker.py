"""Tests for :func:`t0_agents.circuit_breaker`."""

from __future__ import annotations

import asyncio

import pytest

from t0_agents import circuit_breaker
from t0_agents.compose.topology import ComposerError
from tests.test_compose.v1._helpers import MockComposable


async def test_circuit_breaker_closed_passes_through() -> None:
    node = MockComposable(name="node", value="ok")
    breaker = circuit_breaker(node, threshold=3, reset_timeout=30.0)

    result = await breaker.invoke("seed").result()
    assert result.value == "ok"
    assert breaker._state == "closed"
    assert breaker._consecutive_failures == 0


async def test_circuit_breaker_opens_after_threshold_failures() -> None:
    node = MockComposable(name="node", raises=RuntimeError("fail"))
    breaker = circuit_breaker(node, threshold=3, reset_timeout=30.0)

    for _ in range(3):
        with pytest.raises(ComposerError):
            await breaker.invoke("seed").result()

    assert breaker._state == "open"
    assert breaker._consecutive_failures == 3


async def test_circuit_breaker_open_refuses_without_invoking_node() -> None:
    node = MockComposable(name="node", raises=RuntimeError("fail"))
    breaker = circuit_breaker(node, threshold=2, reset_timeout=30.0)

    # Open the breaker.
    for _ in range(2):
        with pytest.raises(ComposerError):
            await breaker.invoke("seed").result()
    assert breaker._state == "open"
    invocations_before_refusal = len(node.invocations)

    # Next call should refuse without invoking.
    with pytest.raises(ComposerError) as excinfo:
        await breaker.invoke("seed").result()
    assert "circuit_breaker is open" in str(excinfo.value.cause)
    assert len(node.invocations) == invocations_before_refusal


async def test_circuit_breaker_half_open_after_reset_timeout_success_closes() -> None:
    node = MockComposable(name="node", raises=RuntimeError("fail"))
    breaker = circuit_breaker(node, threshold=1, reset_timeout=0.05)

    # Trip the breaker.
    with pytest.raises(ComposerError):
        await breaker.invoke("seed").result()
    assert breaker._state == "open"

    # Wait past reset_timeout.
    await asyncio.sleep(0.1)

    # Replace node to start succeeding.
    breaker.child = MockComposable(name="recovered", value="ok")  # type: ignore[misc]

    result = await breaker.invoke("seed").result()
    assert result.value == "ok"
    assert breaker._state == "closed"
    assert breaker._consecutive_failures == 0


async def test_circuit_breaker_half_open_failure_reopens() -> None:
    node = MockComposable(name="node", raises=RuntimeError("fail"))
    breaker = circuit_breaker(node, threshold=1, reset_timeout=0.05)

    with pytest.raises(ComposerError):
        await breaker.invoke("seed").result()
    assert breaker._state == "open"
    first_opened_at = breaker._opened_at

    await asyncio.sleep(0.1)

    # Half-open attempt fails again — should re-open with fresh opened_at.
    with pytest.raises(ComposerError):
        await breaker.invoke("seed").result()
    assert breaker._state == "open"
    assert breaker._opened_at is not None
    assert breaker._opened_at > (first_opened_at or 0)
