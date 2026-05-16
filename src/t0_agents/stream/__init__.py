"""Streaming primitives for :meth:`Run.events` and the SSE / WS serving layer.

Public surface:

* :data:`AgentEvent` and every variant dataclass (``TextDeltaEvent``,
  ``TextEndEvent``, ``ToolCallDeltaEvent``, ``ToolCallEvent``,
  ``ToolResultEvent``, ``ThoughtDeltaEvent``, ``ThoughtEvent``,
  ``TurnEndEvent``, ``DoneEvent``, ``ErrorEvent``).
* :class:`StreamingAgentLoop` — streaming analogue of
  :func:`t0_agents.mcp.loop.run_agent`.
* :class:`TurnStreamAssembler` — folds T0 chunks into ``AgentEvent`` s.

``stream/`` has zero new third-party dependencies (only ``tensorzero`` +
``mcp/``), so these symbols are re-exported unconditionally from the
top-level :mod:`t0_agents` package without an extras gate.
"""

from __future__ import annotations

from t0_agents.stream.assembler import TurnStreamAssembler
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
from t0_agents.stream.runner import StreamingAgentLoop

__all__ = [
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "StreamingAgentLoop",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThoughtDeltaEvent",
    "ThoughtEvent",
    "ToolCallDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStreamAssembler",
]
