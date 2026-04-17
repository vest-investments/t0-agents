"""Unit tests for :func:`t0_agents.cli.loader.load_agent_factory`."""

from __future__ import annotations

import pytest
import typer

from t0_agents.cli.loader import load_agent_factory


async def test_loader_valid_spec_returns_factory() -> None:
    factory = load_agent_factory("tests.test_cli._factories:fake_factory")
    assert callable(factory)
    async with factory() as agent:
        assert agent.config.name == "test-agent"


def test_loader_missing_colon_raises() -> None:
    with pytest.raises(typer.BadParameter, match="'module:attr'"):
        load_agent_factory("tests.test_cli._factories.fake_factory")


def test_loader_empty_module_raises() -> None:
    with pytest.raises(typer.BadParameter, match="non-empty module"):
        load_agent_factory(":fake_factory")


def test_loader_empty_attr_raises() -> None:
    with pytest.raises(typer.BadParameter, match="non-empty module"):
        load_agent_factory("tests.test_cli._factories:")


def test_loader_missing_module_raises() -> None:
    with pytest.raises(typer.BadParameter, match="Could not import module"):
        load_agent_factory("definitely.not.a.real.module:attr")


def test_loader_missing_attr_raises() -> None:
    with pytest.raises(typer.BadParameter, match="has no attribute"):
        load_agent_factory(
            "tests.test_cli._factories:does_not_exist_xyz",
        )


def test_loader_bare_agent_instance_raises_with_factory_hint() -> None:
    with pytest.raises(typer.BadParameter) as exc_info:
        load_agent_factory(
            "tests.test_cli._factories:fake_agent_instance",
        )
    message = str(exc_info.value)
    assert "bare Agent instance" in message
    assert "@asynccontextmanager" in message


def test_loader_non_callable_raises() -> None:
    with pytest.raises(typer.BadParameter, match="resolves to dict"):
        load_agent_factory(
            "tests.test_cli._factories:not_callable",
        )


def test_loader_callable_non_agent_is_accepted() -> None:
    factory = load_agent_factory(
        "tests.test_cli._factories:not_a_factory",
    )
    assert callable(factory)
