"""Unit tests for the ``t0-agents info`` subcommand."""

from __future__ import annotations

from typer.testing import CliRunner

from t0_agents.cli.app import app

_GLOW_TWO_SPEC = "tests.test_cli._factories:glow_app_two"


def test_info_prints_agent_config_fields() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["info", "tests.test_cli._factories:fake_factory"],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "AgentConfig" in out
    assert "test-agent" in out
    assert "agent" in out  # function_name
    assert "max_turns" in out


def test_info_invalid_spec_exits_non_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info", "bogus-spec"])
    assert result.exit_code != 0
    assert "module:attr" in result.output


def test_info_missing_module_exits_non_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["info", "definitely.not.a.module.anywhere:x"],
    )
    assert result.exit_code != 0
    assert "Could not import" in result.output


def test_info_bare_agent_shows_factory_hint() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["info", "tests.test_cli._factories:fake_agent_instance"],
    )
    assert result.exit_code != 0
    assert "@asynccontextmanager" in result.output


def test_info_glow_app_with_two_agents_names_a_flag_it_accepts() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info", _GLOW_TWO_SPEC])
    assert result.exit_code != 0
    assert "alpha, beta" in result.output
    assert "--agent" in result.output


def test_info_glow_app_selection_by_agent_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["info", _GLOW_TWO_SPEC, "--agent", "beta"])
    assert result.exit_code == 0, result.output
    assert "AgentConfig" in result.output
    assert "beta" in result.output
