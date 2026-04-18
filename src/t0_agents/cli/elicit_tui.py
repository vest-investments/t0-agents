"""Rich-based FastMCP elicitation handler for the CLI.

The CLI's ``run`` subcommand plugs :func:`rich_elicitation_handler` into
``fastmcp.Client(mcp, elicitation_handler=...)`` so a tool that calls
``ctx.elicit(...)`` gets its prompt rendered with Rich.

The handler relies on the strict elicitation-schema subset that FastMCP
validates at server construction time: elicitation schemas are always a
flat ``{"type": "object", "properties": {...}}`` with primitive field
types (``string``, ``integer``, ``number``, ``boolean``) plus ``enum``
/ ``const`` and preserved ``description`` / ``default``.  Nested
objects and arrays of objects are rejected server-side, so the handler
can iterate the top-level properties without worrying about recursion.
The one exception — unknown ``type`` values from non-FastMCP servers —
falls back to :class:`~rich.prompt.Prompt` in string mode with a logged
warning.

Pair with :class:`~t0_agents.cli.tui.TUIRenderer` when a Live region is
active: :func:`make_rich_elicitation_handler` takes an optional
``renderer`` and pauses / resumes the panel around each prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.client.elicitation import ElicitResult
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ElicitRequestFormParams, ElicitRequestParams
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from t0_agents.cli.tui import TUIRenderer, get_active_renderer

logger = logging.getLogger(__name__)


async def rich_elicitation_handler(
    message: str,
    response_type: type[Any] | None,
    params: ElicitRequestParams,
    context: RequestContext[Any, LifespanContextT],
) -> ElicitResult[dict[str, Any]]:
    """Default zero-configuration Rich elicitation handler.

    When called inside :meth:`TUIRenderer.consume` (i.e. the ``run``
    REPL is streaming into a live panel) the handler picks up the
    active renderer from
    :func:`~t0_agents.cli.tui.get_active_renderer` and pauses / resumes
    the panel around each Rich prompt.  Outside that context (standalone
    ``Client`` usage, tests, non-CLI callers) it falls back to a plain
    :class:`~rich.console.Console` with no coordination.

    Plug into a standalone ``Client`` directly::

        Client(mcp, elicitation_handler=rich_elicitation_handler)
    """
    renderer = get_active_renderer()
    handler = make_rich_elicitation_handler(renderer=renderer)
    return await handler(message, response_type, params, context)


def make_rich_elicitation_handler(
    *,
    renderer: TUIRenderer | None = None,
    console: Console | None = None,
) -> Callable[
    [str, type[Any] | None, ElicitRequestParams, RequestContext[Any, Any]],
    Awaitable[ElicitResult[dict[str, Any]]],
]:
    """Build an ``ElicitationHandler`` coupled to a Rich console.

    Routing rules:

    - URL-mode elicitation → print the URL and ask for completion.
    - Form-mode with an empty schema → binary
      :class:`~rich.prompt.Confirm` (accept / decline).
    - Form-mode with properties → one Rich prompt per property,
      returning an ``accept`` result whose ``content`` mirrors the
      schema.

    The ``response_type`` argument from FastMCP is ignored when the
    schema is already available — driving the UI from the schema keeps
    the rendering deterministic and avoids round-tripping through
    Pydantic validation.

    Args:
        renderer: Optional :class:`TUIRenderer` whose Live region should
            be paused while the prompt blocks on user input.  When
            ``None`` the handler assumes no Live region is active and
            prompts directly.
        console: Rich console to prompt against.  Defaults to a fresh
            :class:`Console`.  Pass the renderer's own console if you
            want the prompt styling to match the panel.

    Returns:
        A callable matching FastMCP's
        :data:`fastmcp.client.elicitation.ElicitationHandler` type alias.
    """
    _configured_renderer = renderer
    _configured_console = console

    async def _handler(
        message: str,
        response_type: type[Any] | None,
        params: ElicitRequestParams,
        context: RequestContext[Any, LifespanContextT],
    ) -> ElicitResult[dict[str, Any]]:
        active_renderer = _configured_renderer or get_active_renderer()
        active_console = (
            _configured_console
            or (active_renderer.console if active_renderer is not None else Console())
        )
        if active_renderer is not None:
            active_renderer.pause()
        try:
            return _run_prompt(message, response_type, params, active_console)
        except (KeyboardInterrupt, EOFError):
            active_console.print("[yellow]elicitation cancelled[/yellow]")
            return ElicitResult(action="cancel", content=None)
        except Exception:
            # FastMCP wraps any handler exception as ``ErrorData`` and
            # propagates it as a tool-side error; logging the traceback
            # locally is the only way to diagnose what raised, since the
            # server-side message is just ``str(e)``.
            logger.exception("elicit handler raised; returning error to MCP")
            raise
        finally:
            if active_renderer is not None:
                active_renderer.resume()

    return _handler


def _run_prompt(
    message: str,
    response_type: type[Any] | None,
    params: ElicitRequestParams,
    console: Console,
) -> ElicitResult[dict[str, Any]]:
    """Synchronous core — separated so both paths share the error handling."""
    if not isinstance(params, ElicitRequestFormParams):
        console.print(f"[bold]{message}[/bold]")
        url = getattr(params, "url", None)
        if url:
            console.print(f"[cyan]{url}[/cyan]")
        ok = Confirm.ask(
            "Have you completed the out-of-band flow?",
            default=False,
            console=console,
        )
        return ElicitResult(
            action="accept" if ok else "decline",
            content=None,
        )

    schema = params.requestedSchema or {}
    properties: dict[str, dict[str, Any]] = schema.get("properties") or {}
    if not properties:
        ok = Confirm.ask(message, default=False, console=console)
        return ElicitResult(
            action="accept" if ok else "decline",
            content=None,
        )

    console.print(f"[bold]{message}[/bold]")

    required: set[str] = set(schema.get("required") or [])
    content: dict[str, Any] = {}
    for name, prop in properties.items():
        content[name] = _prompt_for_field(name, prop, required, console)

    return ElicitResult(action="accept", content=content)


def _prompt_for_field(
    name: str,
    prop: dict[str, Any],
    required: set[str],
    console: Console,
) -> Any:
    """Render a single property using the appropriate Rich prompt."""
    label = prop.get("description") or name
    default = prop.get("default")
    enum = prop.get("enum")
    const = prop.get("const")

    if const is not None:
        return const

    if enum is not None:
        choices = [str(value) for value in enum]
        default_str = str(default) if default is not None else None
        if default_str is not None and default_str not in choices:
            default_str = None
        answer = Prompt.ask(
            label,
            choices=choices,
            default=default_str,
            console=console,
        )
        for original in enum:
            if str(original) == answer:
                return original
        return answer

    ftype = prop.get("type", "string")
    if ftype == "boolean":
        return Confirm.ask(
            label,
            default=bool(default) if default is not None else False,
            console=console,
        )
    if ftype == "integer":
        if default is not None:
            return IntPrompt.ask(label, default=int(default), console=console)
        return IntPrompt.ask(label, console=console)
    if ftype == "number":
        if default is not None:
            return FloatPrompt.ask(label, default=float(default), console=console)
        return FloatPrompt.ask(label, console=console)
    if ftype == "string":
        default_str = default if isinstance(default, str) else None
        if default_str is not None:
            value = Prompt.ask(label, default=default_str, console=console)
        elif name in required:
            value = Prompt.ask(label, console=console)
        else:
            value = Prompt.ask(label, default="", console=console)
            if value == "":
                return None
        return value

    logger.warning(
        "rich_elicitation_handler: unknown schema type %r for field %r; "
        "falling back to plain string prompt.",
        ftype,
        name,
    )
    default_str = default if isinstance(default, str) else None
    if default_str is not None:
        return Prompt.ask(label, default=default_str, console=console)
    return Prompt.ask(label, console=console)


