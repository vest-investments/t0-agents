"""Typed event union emitted by :meth:`Agent.stream`.

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
    """Result of executing the :class:`ToolCallEvent` with matching ``id``."""

    id: str
    name: str
    result: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


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

    result: AgentResult
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    """Non-fatal anomaly surfaced inline.

    Fatal errors (``MaxTurnsExceededError``, ``GuardrailBlockedError``) still
    raise out of the async iterator rather than converting to this event.
    Used for recoverable artefacts — invalid tool-call JSON, unknown chunk
    types, recoverable ``TensorZeroError`` retries.
    """

    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    type: Literal["error"] = "error"


AgentEvent = (
    TextDeltaEvent
    | TextEndEvent
    | ToolCallDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | ThoughtDeltaEvent
    | ThoughtEvent
    | TurnEndEvent
    | DoneEvent
    | ErrorEvent
)
