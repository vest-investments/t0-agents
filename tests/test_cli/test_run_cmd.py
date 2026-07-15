"""Offline tests for the ``t0-agents run`` REPL."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from t0_agents.cli.app import app
from t0_agents.stream.events import (
    DoneEvent,
    TextDeltaEvent,
)
from tests.test_cli import _factories

_SCRIPTED_SPEC = "tests.test_cli._factories:scripted_factory"


def test_run_happy_path_single_turn() -> None:
    _factories.SCRIPTED_TURNS.append(
        [
            TextDeltaEvent(id="t-1", delta="hello "),
            TextDeltaEvent(id="t-1", delta="world"),
            DoneEvent(result=_factories.emit_agent_result("hello world", "ep-1")),
        ],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", _SCRIPTED_SPEC], input="Hi there\n")
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output
    assert len(_factories.SCRIPTED_CALLS) == 1
    assert _factories.SCRIPTED_CALLS[0]["user_message"] == "Hi there"


def test_run_empty_input_is_skipped() -> None:
    _factories.SCRIPTED_TURNS.append(
        [DoneEvent(result=_factories.emit_agent_result("answered"))],
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", _SCRIPTED_SPEC],
        input="\n  \nreal question\n",
    )
    assert result.exit_code == 0, result.output
    assert len(_factories.SCRIPTED_CALLS) == 1
    assert _factories.SCRIPTED_CALLS[0]["user_message"] == "real question"


def test_run_episode_id_threaded_across_turns() -> None:
    _factories.SCRIPTED_TURNS.extend(
        [
            [DoneEvent(result=_factories.emit_agent_result("one", "ep-A"))],
            [DoneEvent(result=_factories.emit_agent_result("two", "ep-A"))],
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", _SCRIPTED_SPEC],
        input="first\nsecond\n",
    )
    assert result.exit_code == 0, result.output
    assert len(_factories.SCRIPTED_CALLS) == 2
    assert _factories.SCRIPTED_CALLS[0]["episode_id"] is None
    assert _factories.SCRIPTED_CALLS[1]["episode_id"] == "ep-A"


def test_run_stream_error_does_not_crash_repl() -> None:
    _factories.SCRIPTED_EXCEPTIONS.extend([RuntimeError("boom"), None])
    _factories.SCRIPTED_TURNS.append(
        [DoneEvent(result=_factories.emit_agent_result("recovered"))],
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", _SCRIPTED_SPEC],
        input="trigger\nrecover\n",
    )
    assert result.exit_code == 0, result.output
    assert "boom" in result.output
    assert len(_factories.SCRIPTED_CALLS) == 2


def test_run_invalid_spec_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "not-a-spec"])
    assert result.exit_code != 0
    assert "module:attr" in result.output


def test_run_keyboard_interrupt_at_prompt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C at the 'You:' prompt should quit instead of crashing."""
    from rich import prompt as _rich_prompt

    def _raise_interrupt(*args: object, **kwargs: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(_rich_prompt.Prompt, "ask", classmethod(_raise_interrupt))
    runner = CliRunner()
    result = runner.invoke(app, ["run", _SCRIPTED_SPEC])
    assert result.exit_code == 0, result.output
    assert "bye" in result.output


# ---------------------------------------------------------------------------
# One-shot message + glow App specs (M7)
# ---------------------------------------------------------------------------

_GLOW_SPEC = "tests.test_cli._factories:glow_app"


def test_run_one_shot_message_renders_and_exits() -> None:
    _factories.SCRIPTED_TURNS.append(
        [
            TextDeltaEvent(id="t-1", delta="one-shot reply"),
            DoneEvent(result=_factories.emit_agent_result("one-shot reply", "ep-1")),
        ],
    )
    runner = CliRunner()
    # No stdin at all: the one-shot path must never prompt.
    result = runner.invoke(app, ["run", _SCRIPTED_SPEC, "just this"])
    assert result.exit_code == 0, result.output
    assert "one-shot reply" in result.output
    assert len(_factories.SCRIPTED_CALLS) == 1
    assert _factories.SCRIPTED_CALLS[0]["user_message"] == "just this"


def test_run_glow_app_spec_one_shot_with_agent_flag() -> None:
    _factories.SCRIPTED_TURNS.append(
        [
            TextDeltaEvent(id="t-1", delta="from glow"),
            DoneEvent(result=_factories.emit_agent_result("from glow")),
        ],
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", _GLOW_SPEC, "hola", "--agent", "solo"])
    assert result.exit_code == 0, result.output
    assert "from glow" in result.output
    assert len(_factories.SCRIPTED_CALLS) == 1
    assert _factories.SCRIPTED_CALLS[0]["user_message"] == "hola"
