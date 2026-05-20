"""Unit tests for :mod:`t0_agents.cli.elicit_tui`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, cast

import pytest
from fastmcp.client.elicitation import ElicitResult
from mcp.types import ElicitRequestFormParams, ElicitRequestURLParams
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from t0_agents.cli.elicit_tui import (
    make_rich_elicitation_handler,
    rich_elicitation_handler,
)


def _console() -> Console:
    return Console(
        force_terminal=False,
        color_system=None,
        width=80,
        highlight=False,
    )


def _ctx() -> Any:
    return cast(Any, None)


@pytest.fixture
def prompt_scripts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Yield a list; tests append scripted answers, monkey-patched Prompts pop."""
    scripted: list[Any] = []

    def _fake_prompt(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if not scripted:
            raise AssertionError(
                f"{cls.__name__}.ask called with no scripted answer remaining",
            )
        return scripted.pop(0)

    for cls in (Prompt, IntPrompt, FloatPrompt, Confirm):
        monkeypatch.setattr(cls, "ask", classmethod(_fake_prompt))
    yield scripted


async def test_binary_approval_accept(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(True)
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(
        message="Proceed?",
        requestedSchema={"type": "object", "properties": {}},
    )
    result = await handler("Proceed?", None, params, _ctx())
    assert isinstance(result, ElicitResult)
    assert result.action == "accept"
    assert result.content is None


async def test_binary_approval_decline(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(False)
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(
        message="Proceed?",
        requestedSchema={"type": "object", "properties": {}},
    )
    result = await handler("Proceed?", None, params, _ctx())
    assert result.action == "decline"
    assert result.content is None


async def test_typed_string_prompt(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append("Alice")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Your name"},
        },
        "required": ["name"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(
        message="Tell me about yourself", requestedSchema=schema,
    )
    result = await handler("x", str, params, _ctx())
    assert result.action == "accept"
    assert result.content == {"name": "Alice"}


async def test_typed_integer_prompt(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(42)
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "description": "Age in years"},
        },
        "required": ["age"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="Age?", requestedSchema=schema)
    result = await handler("Age?", int, params, _ctx())
    assert result.content == {"age": 42}


async def test_typed_number_prompt(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(3.14)
    schema = {
        "type": "object",
        "properties": {"pi": {"type": "number"}},
        "required": ["pi"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="Pi?", requestedSchema=schema)
    result = await handler("Pi?", float, params, _ctx())
    assert result.content == {"pi": 3.14}


async def test_typed_boolean_prompt(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(True)
    schema = {
        "type": "object",
        "properties": {"agree": {"type": "boolean"}},
        "required": ["agree"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="Agree?", requestedSchema=schema)
    result = await handler("Agree?", bool, params, _ctx())
    assert result.content == {"agree": True}


async def test_enum_coerces_back_to_original_type(
    prompt_scripts: list[Any],
) -> None:
    prompt_scripts.append("2")
    schema = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "integer",
                "enum": [1, 2, 3],
                "description": "Pick a number",
            },
        },
        "required": ["choice"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="?", requestedSchema=schema)
    result = await handler("?", int, params, _ctx())
    assert result.content == {"choice": 2}
    assert isinstance(result.content["choice"], int)  # type: ignore[index]


async def test_scalar_wrap_single_value_property(
    prompt_scripts: list[Any],
) -> None:
    prompt_scripts.append("hi")
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="Say something", requestedSchema=schema)
    result = await handler("Say something", str, params, _ctx())
    assert result.content == {"value": "hi"}


async def test_multi_field_form_with_defaults_iterates_every_property(
    prompt_scripts: list[Any],
) -> None:
    """Regression for the example-13 ``ReminderConfirmation`` schema.

    The dataclass produces ``confirm`` (required boolean), ``channel``
    (string + enum + default), and ``note`` (string + default ""). The
    handler must prompt for **all three** in order and return content
    that round-trips through Pydantic — if any field gets dropped or
    coerced to ``None`` the server-side ``ctx.elicit`` raises a
    ``ValidationError`` and the tool comes back as ``is_error=True``,
    which is exactly what we saw in the screenshot the user shared.
    """
    prompt_scripts.extend([True, "email", ""])
    schema = {
        "type": "object",
        "properties": {
            "confirm": {
                "description": "Confirm the reminder?",
                "title": "Confirm",
                "type": "boolean",
            },
            "channel": {
                "default": "email",
                "description": "Delivery channel",
                "enum": ["email", "sms", "push"],
                "title": "Channel",
                "type": "string",
            },
            "note": {
                "default": "",
                "description": "Optional note for the reminder",
                "title": "Note",
                "type": "string",
            },
        },
        "required": ["confirm"],
        "title": "ReminderConfirmation",
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(
        message="Reminder at tomorrow: Pagar la renta",
        requestedSchema=schema,
    )
    result = await handler(
        "Reminder at tomorrow: Pagar la renta", None, params, _ctx(),
    )
    assert result.action == "accept"
    assert result.content == {
        "confirm": True,
        "channel": "email",
        "note": "",
    }
    assert prompt_scripts == [], "all three fields must be consumed"


async def test_const_returns_value_without_prompt(
    prompt_scripts: list[Any],
) -> None:
    schema = {
        "type": "object",
        "properties": {"fixed": {"type": "string", "const": "pinned"}},
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="?", requestedSchema=schema)
    result = await handler("?", str, params, _ctx())
    assert result.content == {"fixed": "pinned"}
    assert prompt_scripts == []


async def test_unknown_type_falls_back_to_string(
    prompt_scripts: list[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt_scripts.append("freeform")
    schema = {
        "type": "object",
        "properties": {
            "mystery": {"type": "mystery-type"},
        },
        "required": ["mystery"],
    }
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="?", requestedSchema=schema)
    with caplog.at_level("WARNING", logger="t0_agents.cli.elicit_tui"):
        result = await handler("?", None, params, _ctx())
    assert result.content == {"mystery": "freeform"}
    assert any("unknown schema type" in r.message for r in caplog.records)


async def test_empty_properties_fallback_to_binary(
    prompt_scripts: list[Any],
) -> None:
    prompt_scripts.append(True)
    schema = {"type": "object", "properties": {}}
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(message="Proceed?", requestedSchema=schema)
    result = await handler("Proceed?", None, params, _ctx())
    assert result.action == "accept"
    assert result.content is None


async def test_url_elicitation_prompts_for_done(prompt_scripts: list[Any]) -> None:
    prompt_scripts.append(True)
    params = ElicitRequestURLParams(
        message="Visit this URL",
        url="https://example.com/auth",
        elicitationId="e-1",
    )
    handler = make_rich_elicitation_handler(console=_console())
    result = await handler("Visit this URL", None, params, _ctx())
    assert result.action == "accept"
    assert result.content is None


async def test_keyboard_interrupt_returns_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _interrupt(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(Confirm, "ask", classmethod(_interrupt))
    handler = make_rich_elicitation_handler(console=_console())
    params = ElicitRequestFormParams(
        message="Proceed?", requestedSchema={"type": "object", "properties": {}},
    )
    result = await handler("Proceed?", None, params, _ctx())
    assert result.action == "cancel"
    assert result.content is None


async def test_default_rich_elicitation_handler_shortcut(
    prompt_scripts: list[Any],
) -> None:
    """The public :func:`rich_elicitation_handler` wraps the factory."""
    prompt_scripts.append(True)
    params = ElicitRequestFormParams(
        message="Proceed?", requestedSchema={"type": "object", "properties": {}},
    )
    result = await rich_elicitation_handler("Proceed?", None, params, _ctx())
    assert result.action == "accept"


async def test_handler_pauses_active_renderer_during_prompt(
    prompt_scripts: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Live ↔ Prompt bug fix: when an active renderer is installed,
    the zero-config ``rich_elicitation_handler`` must pause the
    renderer before prompting and resume after.
    """
    from t0_agents.cli import tui as _tui

    class _TrackingRenderer:
        console = _console()
        pauses: ClassVar[list[str]] = []

        def pause(self) -> None:
            _TrackingRenderer.pauses.append("pause")

        def resume(self) -> None:
            _TrackingRenderer.pauses.append("resume")

    def _scripted_confirm(cls: Any, *args: Any, **kwargs: Any) -> Any:
        _TrackingRenderer.pauses.append("prompt")
        return True

    monkeypatch.setattr("rich.prompt.Confirm.ask", classmethod(_scripted_confirm))

    fake = _TrackingRenderer()
    _tui._set_active_renderer(cast(Any, fake))
    try:
        params = ElicitRequestFormParams(
            message="Proceed?",
            requestedSchema={"type": "object", "properties": {}},
        )
        result = await rich_elicitation_handler("Proceed?", None, params, _ctx())
    finally:
        _tui._set_active_renderer(None)

    assert result.action == "accept"
    assert _TrackingRenderer.pauses == ["pause", "prompt", "resume"]


async def test_handler_sees_renderer_across_task_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the ContextVar → module-global switch.

    The ``Client(mcp)`` session runs its ``_receive_loop`` on a task
    that was created at ``__aenter__`` time — i.e. **before**
    :meth:`TUIRenderer.consume` installs the active renderer.  The
    elicitation callback fires from that task.  A ``ContextVar`` would
    not propagate; a module-level binding does.  We simulate the
    condition by launching the handler from a separately-created
    ``asyncio.Task`` after the registration, which is the same
    scenario in miniature.
    """
    import asyncio

    from t0_agents.cli import tui as _tui

    calls: list[str] = []

    class _TrackingRenderer:
        console = _console()

        def pause(self) -> None:
            calls.append("pause")

        def resume(self) -> None:
            calls.append("resume")

    def _scripted(cls: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("prompt")
        return True

    monkeypatch.setattr("rich.prompt.Confirm.ask", classmethod(_scripted))

    _tui._set_active_renderer(cast(Any, _TrackingRenderer()))
    try:
        async def _spawn() -> Any:
            params = ElicitRequestFormParams(
                message="Proceed?",
                requestedSchema={"type": "object", "properties": {}},
            )
            return await rich_elicitation_handler("Proceed?", None, params, _ctx())

        task = asyncio.create_task(_spawn())
        result = await task
    finally:
        _tui._set_active_renderer(None)

    assert result.action == "accept"
    assert calls == ["pause", "prompt", "resume"]


async def test_handler_without_active_renderer_skips_pause(
    prompt_scripts: list[Any],
) -> None:
    """No ContextVar => the handler falls back to a plain Console and
    does not attempt to call pause/resume (nothing would be paused
    anyway).  Regression guard for the shortcut usage in tests and
    non-CLI callers.
    """
    prompt_scripts.append(True)
    params = ElicitRequestFormParams(
        message="Proceed?",
        requestedSchema={"type": "object", "properties": {}},
    )
    result = await rich_elicitation_handler("Proceed?", None, params, _ctx())
    assert result.action == "accept"
