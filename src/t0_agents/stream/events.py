"""Typed event union emitted by :meth:`Run.events`.

Every event carries a ``type`` discriminator literal so callers can ``match``
on the variant without isinstance checks.  Every delta event carries the
underlying TensorZero chunk's ``id`` (see contract §3.3 correlation key) so
consumers can pair a ``TextDeltaEvent`` with its terminating
``TextEndEvent``, accumulate ``ToolCallDeltaEvent`` s into a single
``ToolCallEvent``, and so on.

JSON serialisation at the HTTP boundary is ``dataclasses.asdict(event)`` —
each event shape maps 1:1 to the wire envelope so browser clients can decode
generically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from t0_agents.core.types import AgentResult


@dataclass(slots=True)
class TextDeltaEvent:
    """One delta of assistant text for the content block ``id``."""

    id: str
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndEvent:
    """Emitted after the final ``TextDeltaEvent`` for ``id``."""

    id: str
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ToolCallDeltaEvent:
    """Partial tool-call name or arguments as TensorZero streams them."""

    id: str
    raw_name_delta: str = ""
    raw_arguments_delta: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(slots=True)
class ToolCallEvent:
    """Assembled tool-call ready for dispatch.

    ``arguments`` is the parsed JSON dict.  When the LLM produced invalid
    JSON the assembler emits an :class:`ErrorEvent` and yields a
    ``ToolCallEvent`` with ``arguments={}`` so the correlation ``id`` is
    still observable by consumers.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultEvent:
    """Result of executing the :class:`ToolCallEvent` with matching ``id``.

    ``result`` is the model-visible serialized string and stays the source of
    truth for the model.  The rest is additive observability:

    * ``is_error`` — typed error flag from the executor (the dispatch error
      branch or ``CallToolResult.is_error``), not a sniff of ``result``.
    * ``structured`` — the tool's raw ``CallToolResult.structured_content``
      (a plain dict) when it returned one, else ``None``.
    * ``duration_ms`` — the ``call_tool`` round-trip in milliseconds; ``None``
      when the call never dispatched (hallucinated name / invalid JSON args).
    * ``server`` — namespace of the owning mounted server, or ``None`` for an
      unprefixed / single-server mount (the ``statuses()`` attribution rule).
    """

    id: str
    name: str
    result: str
    is_error: bool = False
    structured: dict[str, Any] | None = None
    duration_ms: float | None = None
    server: str | None = None
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class ToolProgressEvent:
    """Progress notification from the MCP tool executing under ``id``.

    ``id`` matches the :class:`ToolCallEvent` (and eventual
    :class:`ToolResultEvent`) it reports on.  Emitted only while streaming —
    fastmcp delivers server-side ``ctx.report_progress(...)`` calls to a
    per-call handler, and the loop forwards them here instead of dropping
    them.  ``total`` and ``message`` are ``None`` when the server omits them.
    """

    id: str
    progress: float
    total: float | None = None
    message: str | None = None
    type: Literal["tool_progress"] = "tool_progress"


@dataclass(slots=True)
class ThoughtDeltaEvent:
    """One delta of reasoning text for the thought block ``id``."""

    id: str
    delta: str
    type: Literal["thought_delta"] = "thought_delta"


@dataclass(slots=True)
class ThoughtEvent:
    """Assembled reasoning text after the final delta for ``id``."""

    id: str
    text: str
    type: Literal["thought"] = "thought"


@dataclass(slots=True)
class TurnEndEvent:
    """Emitted once per turn, after all blocks for that turn are assembled."""

    turn: int
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class DoneEvent:
    """Final event: carries the full :class:`AgentResult`."""

    result: AgentResult[Any]
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    """Non-fatal anomaly surfaced inline — the stream continues after it.

    Used for recoverable artefacts — invalid tool-call JSON, unknown chunk
    types, recoverable ``TensorZeroError`` retries.  Fatal errors
    (``MaxTurnsExceededError``, ``GuardrailBlockedError``) never convert to
    this event: with the default ``Run.events(on_error="raise")`` they raise
    out of the async iterator; with ``on_error="event"`` they end the stream
    as a single terminal :class:`RunErrorEvent`.
    """

    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: Literal["error"] = "error"


@dataclass(slots=True)
class RunErrorEvent:
    """Terminal event: the run failed and the stream ends here.

    Emitted only under ``Run.events(on_error="event")``, in place of the
    fatal exception that ``on_error="raise"`` would re-raise.  A stream in
    that mode ends with exactly one ``DoneEvent | RunErrorEvent``.  Contrast
    with :class:`ErrorEvent`, which is recoverable and inline.

    ``error_type`` is the exception class name (e.g.
    ``"MaxTurnsExceededError"``) and ``detail`` carries structured,
    JSON-serialisable extras (e.g. ``{"max_turns": 20}``).  ``exception`` is
    the original exception for in-process consumers (logging, re-raise); it
    is deliberately excluded from the codec and never crosses a wire.
    """

    error_type: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = field(default=None, repr=False, compare=False)
    type: Literal["run_error"] = "run_error"


AgentEvent = (
    TextDeltaEvent
    | TextEndEvent
    | ToolCallDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | ToolProgressEvent
    | ThoughtDeltaEvent
    | ThoughtEvent
    | TurnEndEvent
    | DoneEvent
    | ErrorEvent
    | RunErrorEvent
)
