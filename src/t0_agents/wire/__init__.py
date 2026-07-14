"""``t0_agents.wire`` — the versioned wire contract for stream consumers.

A frozen :data:`WireFrame` union with a dict codec
(:func:`frame_to_dict` / :func:`frame_from_dict`), evolved additively
under :data:`WIRE_VERSION`. Pure projection stages over the union live
in :mod:`t0_agents.wire.stages`.

The package publishes stable dicts to *any* transport and deliberately
imports none — buses, SSE serialisers, and sockets are consumer
infrastructure. See ``docs/wire.md`` for the contract, the version
policy, and composition examples.

``wire/`` has zero new third-party dependencies, so these symbols are
importable unconditionally without an extras gate; the contract's
identity is the namespace — consumers import from ``t0_agents.wire``.
"""

from __future__ import annotations

from t0_agents.wire.frames import (
    WIRE_VERSION,
    DoneFrame,
    ElicitationRequestFrame,
    ElicitationResolvedFrame,
    ErrorFrame,
    InitFrame,
    StatusFrame,
    TextDeltaFrame,
    TextEndFrame,
    ThoughtDeltaFrame,
    ThoughtFrame,
    ToolCallDeltaFrame,
    ToolCallFrame,
    ToolProgressFrame,
    ToolResultFrame,
    TurnEndFrame,
    WireFrame,
    frame_from_dict,
    frame_to_dict,
)
from t0_agents.wire.stages import (
    MAX_ARG_BYTES,
    MAX_RAW_BYTES,
    MAX_RESULT_BYTES,
    project,
)

__all__ = [
    "MAX_ARG_BYTES",
    "MAX_RAW_BYTES",
    "MAX_RESULT_BYTES",
    "WIRE_VERSION",
    "DoneFrame",
    "ElicitationRequestFrame",
    "ElicitationResolvedFrame",
    "ErrorFrame",
    "InitFrame",
    "StatusFrame",
    "TextDeltaFrame",
    "TextEndFrame",
    "ThoughtDeltaFrame",
    "ThoughtFrame",
    "ToolCallDeltaFrame",
    "ToolCallFrame",
    "ToolProgressFrame",
    "ToolResultFrame",
    "TurnEndFrame",
    "WireFrame",
    "frame_from_dict",
    "frame_to_dict",
    "project",
]
