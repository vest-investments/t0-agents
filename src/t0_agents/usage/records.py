"""Records a :class:`~t0_agents.usage.ledger.UsageLedger` receives.

One :class:`InferenceUsage` per gateway inference, one
:class:`RunUsageRecord` per run — frozen, plain dataclasses a consumer
persists with whatever store it owns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from t0_agents.core.types import RunUsage, Usage

RunOutcome = Literal["completed", "cancelled", "failed"]
"""How a run ended, as recorded on :attr:`RunUsageRecord.outcome`."""


@dataclass(frozen=True, slots=True)
class InferenceUsage:
    """One gateway inference, as the gateway reported it.

    ``inference_id`` and ``episode_id`` are the gateway's own ids — never
    a corrective-retry sentinel, which only ever appears in
    :attr:`~t0_agents.core.types.AgentResult.inference_ids`.  ``usage`` is
    the lossless per-inference :class:`~t0_agents.core.types.Usage`, so an
    unpriced call keeps ``cost=None``.  ``tool_calls`` names the tool calls
    the inference emitted, in content order, by the name the model wrote —
    a name the agent does not know is kept as written.  ``variant_name``
    and ``finish_reason`` are ``None`` when the response did not carry
    them; the streaming path's synthetic response carries neither.
    ``at`` is aware UTC.
    """

    inference_id: str
    episode_id: str
    function_name: str | None
    variant_name: str | None
    usage: Usage
    tool_calls: tuple[str, ...]
    finish_reason: str | None
    at: datetime


@dataclass(frozen=True, slots=True)
class RunUsageRecord:
    """One run of the middleware chain, handed to a ledger once it ends.

    ``run_id`` is minted by the library (a UUIDv7) and is the key a ledger
    deduplicates on.  ``inferences`` holds every inference the run made,
    in order — a tool loop or a JSON retry produces several; a run that
    failed before its first inference, or a stream that ended without a
    usage chunk, holds none.  ``usage`` sums them the way
    :class:`~t0_agents.core.types.RunUsage` does (``None`` → ``0``,
    ``cost_is_priced`` when any inference was priced).  A delegated
    sub-agent's inferences are not included — its own chain records them.
    ``episode_id`` is ``None`` only when no inference fired and the caller
    supplied none.  ``tags`` are the tags the gateway received, verbatim.
    ``error_class`` is the exception's class name when ``outcome`` is
    ``"failed"``, else ``None``.  Timestamps are aware UTC.
    """

    run_id: str
    episode_id: str | None
    function_name: str | None
    tags: Mapping[str, str]
    inferences: tuple[InferenceUsage, ...]
    usage: RunUsage
    turns: int
    outcome: RunOutcome
    error_class: str | None
    started_at: datetime
    finished_at: datetime
