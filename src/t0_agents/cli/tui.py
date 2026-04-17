"""Rich renderer for :class:`~t0_agents.stream.events.AgentEvent` streams.

:class:`TUIRenderer` consumes an :class:`~typing.AsyncIterator` of
``AgentEvent`` (usually produced by :meth:`Agent.stream`) and keeps a
live Rich :class:`~rich.panel.Panel` up to date as deltas arrive.  The
panel is wrapped in a :class:`~rich.live.Live` region with a
configurable refresh rate.

The renderer exposes :meth:`pause` / :meth:`resume` so an elicitation
handler can temporarily stop the Live region while it runs a
:class:`~rich.prompt.Prompt`.  ``Live`` redirects ``stdout`` /
``stderr`` and installs a render hook on the ``Console``; calling
``input`` while either is in place corrupts the frame and blocks the
prompt's text behind the panel.

Rich does not ship a native ``pause`` / ``resume`` API
(``rich/live.py`` exposes only ``start`` / ``stop``), nor does its
documentation describe a "pause + prompt + resume" pattern, so we
implement it ourselves with ``Live.stop()`` and ``Live.start(refresh=
True)``.  Source observation: ``Live.stop()`` does not ``join()`` the
refresh thread, but it acquires ``Live._lock`` before signalling
``done``; if a ``refresh()`` is already in flight, the lock contention
forces it to complete before ``stop()`` returns, and the double check
``if not self.done.is_set()`` inside the refresh loop guarantees no
further frame is painted afterwards.  Re-entering ``Live.start()``
re-pushes the render hook, re-redirects stdout/stderr, and spins up a
fresh ``_RefreshThread``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)

_ARGS_MAX_CHARS: int = 60

_active_renderer: TUIRenderer | None = None
_active_renderer_lock = threading.Lock()
"""Process-wide reference to the currently-active :class:`TUIRenderer`.

The ``Client(mcp)`` session opens its own :mod:`anyio` task group on
``__aenter__`` (``mcp/shared/session.py:222``), which runs the MCP
``_receive_loop`` — the task that ultimately invokes the
elicitation callback.  That task's :mod:`contextvars` context is a
snapshot taken at ``Client.__aenter__`` time, which is *before* the
REPL's :meth:`TUIRenderer.consume` runs, so a :class:`ContextVar` does
not propagate.  A plain module-level binding guarded by a
:class:`threading.Lock` is the pragmatic solution: the TUI is a
single-process, single-active-REPL surface, and the lock makes the
read/write race-free against Rich's refresh thread.
"""


def get_active_renderer() -> TUIRenderer | None:
    """Return the currently-active :class:`TUIRenderer`, if any."""
    with _active_renderer_lock:
        return _active_renderer


def _set_active_renderer(renderer: TUIRenderer | None) -> None:
    """Install / clear the active renderer atomically."""
    global _active_renderer
    with _active_renderer_lock:
        _active_renderer = renderer


def _format_args(arguments: dict[str, object]) -> str:
    """Render tool arguments as a compact JSON snippet, truncated."""
    if not arguments:
        return ""
    try:
        rendered = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(arguments)
    if len(rendered) > _ARGS_MAX_CHARS:
        return rendered[: _ARGS_MAX_CHARS - 3] + "..."
    return rendered


class TUIRenderer:
    """Drive a Rich ``Live`` panel from an :class:`AgentEvent` stream.

    Construct once per CLI session; call :meth:`consume` once per user
    turn.  Between turns the renderer preserves :attr:`last_episode_id`
    and :attr:`last_result` but resets the per-turn text / tool / error
    state so the new turn renders on a fresh panel.

    Args:
        agent_name: Title displayed on the panel.
        console: Optional Rich console; defaults to a freshly constructed
            :class:`~rich.console.Console`.
        verbose: If ``True``, :class:`ThoughtEvent` and
            :class:`ThoughtDeltaEvent` content is rendered in the panel.
            Default hides them — reasoning traces are noisy.
        refresh_per_second: Live refresh rate; higher values render
            deltas more smoothly at the cost of CPU.
    """

    agent_name: str
    console: Console
    verbose: bool
    last_episode_id: str | None
    last_result: AgentResult | None

    def __init__(
        self,
        agent_name: str,
        *,
        console: Console | None = None,
        verbose: bool = False,
        refresh_per_second: float = 20.0,
    ) -> None:
        self.agent_name = agent_name
        self.console = console or Console()
        self.verbose = verbose
        self._refresh_per_second = refresh_per_second
        self._text_buffers: dict[str, str] = {}
        self._text_order: list[str] = []
        self._tool_calls: dict[str, dict[str, str]] = {}
        self._tool_order: list[str] = []
        self._thoughts: dict[str, str] = {}
        self._thought_order: list[str] = []
        self._errors: list[str] = []
        self._live: Live | None = None
        self.last_episode_id = None
        self.last_result = None

    # --------------------------------------------------------------
    # Public surface
    # --------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-turn state without touching ``last_*`` fields."""
        self._text_buffers.clear()
        self._text_order.clear()
        self._tool_calls.clear()
        self._tool_order.clear()
        self._thoughts.clear()
        self._thought_order.clear()
        self._errors.clear()

    async def consume(
        self,
        events: AsyncIterator[AgentEvent],
    ) -> AgentResult | None:
        """Drive the panel from an ``AgentEvent`` iterator.

        Returns the :class:`AgentResult` carried by the final
        :class:`DoneEvent`, or ``None`` if the iterator completed
        without emitting one.  Exceptions from the iterator propagate
        unchanged after the panel stops.

        While ``consume`` is running, the renderer installs itself as
        the process's *active* renderer (see :func:`get_active_renderer`).
        The zero-config :func:`~t0_agents.cli.elicit_tui.rich_elicitation_handler`
        picks that reference up and pauses / resumes the Live region
        around every Rich prompt — without that coordination, the
        refresh thread keeps painting the panel at
        ``refresh_per_second`` Hz while the user is typing their
        answer and stacks copies of the frame onto the terminal.
        """
        self.reset()
        _set_active_renderer(self)
        try:
            with Live(
                self._render(),
                console=self.console,
                refresh_per_second=self._refresh_per_second,
                transient=False,
            ) as live:
                self._live = live
                try:
                    async for event in events:
                        self._update(event)
                        live.update(self._render())
                finally:
                    self._live = None
        finally:
            _set_active_renderer(None)
        return self.last_result

    def pause(self) -> None:
        """Stop the live region so external prompts can write to stdout.

        Safe to call when no Live region is active or when it is already
        stopped.  Paired with :meth:`resume`.
        """
        if self._live is not None and self._live.is_started:
            self._live.stop()

    def resume(self) -> None:
        """Resume the live region after a :meth:`pause`.

        Safe to call when no Live region is active.  Called by the
        Rich-based elicitation handler after a prompt completes.
        """
        if self._live is not None and not self._live.is_started:
            self._live.start(refresh=True)

    # --------------------------------------------------------------
    # Event processing
    # --------------------------------------------------------------

    def _update(self, event: AgentEvent) -> None:
        match event:
            case TextDeltaEvent(id=eid, delta=delta):
                if eid not in self._text_buffers:
                    self._text_order.append(eid)
                self._text_buffers[eid] = self._text_buffers.get(eid, "") + delta
            case TextEndEvent():
                pass
            case ToolCallEvent(id=eid, name=name, arguments=arguments):
                if eid not in self._tool_calls:
                    self._tool_order.append(eid)
                self._tool_calls[eid] = {
                    "name": name,
                    "args": _format_args(arguments),
                    "state": "pending",
                }
            case ToolResultEvent(id=eid, name=name, is_error=is_error):
                existing = self._tool_calls.get(
                    eid, {"name": name, "args": "", "state": "pending"},
                )
                if eid not in self._tool_calls:
                    self._tool_order.append(eid)
                existing["state"] = "error" if is_error else "ok"
                self._tool_calls[eid] = existing
            case ToolCallDeltaEvent():
                pass
            case ThoughtDeltaEvent(id=eid, delta=delta):
                if self.verbose:
                    if eid not in self._thoughts:
                        self._thought_order.append(eid)
                    self._thoughts[eid] = self._thoughts.get(eid, "") + delta
            case ThoughtEvent(id=eid, text=text):
                if self.verbose:
                    if eid not in self._thoughts:
                        self._thought_order.append(eid)
                    self._thoughts[eid] = text
            case ErrorEvent(message=message):
                self._errors.append(message)
            case TurnEndEvent():
                pass
            case DoneEvent(result=result):
                self.last_result = result
                self.last_episode_id = result.episode_id

    # --------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------

    def _render(self) -> Panel:
        body = Text()

        for eid in self._tool_order:
            call = self._tool_calls[eid]
            state = call["state"]
            name = call["name"]
            args = call["args"]
            marker = {"pending": "🔧", "ok": "✓", "error": "❌"}[state]
            style = {"pending": "cyan", "ok": "green", "error": "red"}[state]
            if args:
                body.append(f"{marker} {name}({args})\n", style=style)
            else:
                body.append(f"{marker} {name}\n", style=style)

        for eid in self._thought_order:
            thought = self._thoughts[eid].rstrip("\n")
            if thought:
                body.append(thought + "\n", style="italic dim")

        for message in self._errors:
            body.append(f"! {message}\n", style="bold red")

        assistant_parts: list[str] = []
        for eid in self._text_order:
            assistant_parts.append(self._text_buffers[eid])
        if assistant_parts:
            if body.plain:
                body.append("\n")
            body.append("".join(assistant_parts), style="white")

        if not body.plain:
            body.append("…", style="dim")

        return Panel(body, title=self.agent_name, border_style="cyan")

    # --------------------------------------------------------------
    # Test/introspection helpers
    # --------------------------------------------------------------

    def render_text(self) -> str:
        """Render the current panel to a plain string.

        Used by tests to assert what the user would see without having
        to wrestle with terminal escape sequences.
        """
        panel = self._render()
        with self.console.capture() as capture:
            self.console.print(panel)
        return capture.get()
