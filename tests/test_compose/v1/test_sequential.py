"""Tests for :func:`t0_agents.sequential`."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from t0_agents import sequential
from t0_agents.compose.topology import ComposerError
from t0_agents.core.run import RunCancelledError
from t0_agents.core.types import AgentResult
from tests.test_compose.v1._helpers import MockComposable


async def test_sequential_pipeline_default_chain() -> None:
    a = MockComposable(name="a", value="step_a_out")
    b = MockComposable(name="b", value="step_b_out")
    c = MockComposable(name="c", value="step_c_out")

    composer = sequential(a, b, c)
    result = await composer.invoke("seed").result()

    assert result.value == "step_c_out"
    assert a.invocations == ["seed"]
    assert b.invocations == ["step_a_out"]
    assert c.invocations == ["step_b_out"]


async def test_sequential_aggregates_inference_ids() -> None:
    a = MockComposable(name="a", value="x", inference_ids=["inf_a1", "inf_a2"])
    b = MockComposable(name="b", value="y", inference_ids=["inf_b1"])

    composer = sequential(a, b)
    result = await composer.invoke("seed").result()

    assert result.inference_ids == ["inf_a1", "inf_a2", "inf_b1"]
    assert result.turns == 2  # one turn per mock


async def test_sequential_custom_chain_dict_to_str() -> None:
    a = MockComposable(name="extract", value={"query": "the answer"})
    b = MockComposable(name="answer", transform=lambda x: f"got: {x}")

    composer = sequential(
        a,
        b,
        chain=lambda r, _: cast(dict[str, Any], r.value)["query"],
    )
    result = await composer.invoke("input").result()

    assert result.value == "got: the answer"
    assert b.invocations == ["the answer"]


async def test_sequential_default_chain_json_encodes_non_str() -> None:
    """Default chain coerces dict outputs to JSON for the next step."""
    a = MockComposable(name="extract", value={"k": 1})
    b = MockComposable(name="echo")  # transform-None means echo input

    result = await sequential(a, b).invoke("seed").result()

    assert b.invocations == ['{"k": 1}']
    assert result.value == '{"k": 1}'


async def test_sequential_first_step_failure_wraps_as_composer_error() -> None:
    boom = MockComposable(name="boom", raises=ValueError("nope"))
    unreached = MockComposable(name="unreached")

    composer = sequential(boom, unreached)
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke("seed").result()

    assert excinfo.value.node is boom
    assert isinstance(excinfo.value.cause, ValueError)
    assert unreached.invocations == []


async def test_sequential_cancel_mid_pipeline() -> None:
    a = MockComposable(name="a", value="done_a")
    b = MockComposable(name="b", delay=10.0, value="slow_b")
    c = MockComposable(name="c")

    run = sequential(a, b, c).invoke("seed", cancel_timeout=1.0)
    await asyncio.sleep(0.05)  # let a finish, b start
    await run.cancel()

    assert run.state == "cancelled"
    with pytest.raises(RunCancelledError):
        await run.result()
    assert c.invocations == []  # never reached


async def test_sequential_empty_raises() -> None:
    composer = sequential()
    with pytest.raises(ValueError, match=r"sequential.*at least one"):
        await composer.invoke("seed").result()


async def test_sequential_first_sub_run_episode_id_propagates() -> None:
    a = MockComposable(name="a", episode_id="ep_root", value="x")
    b = MockComposable(name="b", episode_id="ep_b_internal", value="y")

    result: AgentResult[Any] = await sequential(a, b).invoke("seed").result()
    assert result.episode_id == "ep_root"
