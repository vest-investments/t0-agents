"""Streaming primitives for :meth:`Run.events` and the SSE / WS serving layer.

Public surface:

* :data:`AgentEvent` and every variant dataclass (``TextDeltaEvent``,
  ``TextEndEvent``, ``ToolCallDeltaEvent``, ``ToolCallEvent``,
  ``ToolResultEvent``, ``ToolProgressEvent``, ``ThoughtDeltaEvent``,
  ``ThoughtEvent``, ``TurnEndEvent``, ``DoneEvent``, ``ErrorEvent``,
  ``RunErrorEvent``).
* :class:`StreamingAgentLoop` — streaming analogue of
  :func:`t0_agents.mcp.loop.run_agent`.
* :class:`TurnStreamAssembler` — folds T0 chunks into ``AgentEvent`` s.

``stream/`` has zero new third-party dependencies (only ``tensorzero`` +
``mcp/``), so these symbols are re-exported unconditionally from the
top-level :mod:`t0_agents` package without an extras gate.
"""

from __future__ import annotations

from t0_agents.stream.assembler import TurnStreamAssembler
from t0_agents.stream.codec import CODEC_VERSION, event_from_dict, event_to_dict
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    RunErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.stream.runner import StreamingAgentLoop
from t0_agents.stream.status import Status, split_tool_identity, synthesize_statuses

__all__ = [
    "CODEC_VERSION",
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "RunErrorEvent",
    "Status",
    "StreamingAgentLoop",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThoughtDeltaEvent",
    "ThoughtEvent",
    "ToolCallDeltaEvent",
    "ToolCallEvent",
    "ToolProgressEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStreamAssembler",
    "event_from_dict",
    "event_to_dict",
    "split_tool_identity",
    "synthesize_statuses",
]
