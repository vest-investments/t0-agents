"""Tests for :func:`t0_agents.conditional`."""

from __future__ import annotations

from typing import Any

import pytest

from t0_agents import conditional
from t0_agents.compose.topology import ComposerError
from tests.test_compose.v1._helpers import MockComposable


async def test_conditional_routes_by_router_key() -> None:
    happy_path = MockComposable(name="happy", value="HAPPY")
    sad_path = MockComposable(name="sad", value="SAD")

    composer = conditional(
        router=lambda input: input["mood"],
        routes={"happy": happy_path, "sad": sad_path},
    )
    result = await composer.invoke({"mood": "sad"}).result()
    assert result.value == "SAD"
    assert sad_path.invocations == [{"mood": "sad"}]
    assert happy_path.invocations == []


async def test_conditional_falls_through_to_default() -> None:
    primary = MockComposable(name="primary", value="P")
    fallback = MockComposable(name="fallback", value="F")

    composer = conditional(
        router=lambda _: "unknown",
        routes={"primary": primary},
        default=fallback,
    )
    result = await composer.invoke({}).result()
    assert result.value == "F"


async def test_conditional_unknown_route_no_default_raises() -> None:
    composer = conditional(
        router=lambda _: "unknown",
        routes={"primary": MockComposable(name="p")},
    )
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke({}).result()
    assert excinfo.value.node is composer
    assert isinstance(excinfo.value.cause, ValueError)


async def test_conditional_router_exception_wraps_with_composer_self() -> None:
    def broken_router(_: Any) -> str:
        raise KeyError("bad key")

    composer = conditional(
        router=broken_router,
        routes={"x": MockComposable(name="x")},
    )
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke({}).result()
    assert excinfo.value.node is composer
    assert isinstance(excinfo.value.cause, KeyError)


async def test_conditional_target_failure_wraps_with_target_node() -> None:
    boom = MockComposable(name="boom", raises=RuntimeError("target broke"))
    composer = conditional(
        router=lambda _: "boom",
        routes={"boom": boom},
    )
    with pytest.raises(ComposerError) as excinfo:
        await composer.invoke({}).result()
    assert excinfo.value.node is boom
