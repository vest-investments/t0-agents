"""Unit tests for the ``t0-agents info`` subcommand."""

from __future__ import annotations

from typer.testing import CliRunner

from t0_agents.cli.app import app


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
