"""``UsageLedgerMiddleware`` — capture every inference, record one run.

Sits inside ``cost_tracking`` and outside ``json_retry`` in the default
chain, so it observes every inference a run makes — each tool-loop round
and each JSON retry — and hands the configured
:class:`~t0_agents.usage.ledger.UsageLedger` one
:class:`~t0_agents.usage.records.RunUsageRecord` when the run ends.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from tensorzero.types import ToolCall
from tensorzero.util import uuid7

from t0_agents.core.middleware.base import Next, RunContext
from t0_agents.core.types import AgentResult, RunUsage, Usage
from t0_agents.usage.ledger import UsageLedger
from t0_agents.usage.records import InferenceUsage, RunOutcome, RunUsageRecord

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 2.0
"""Seconds a ledger write may take before it is abandoned and logged."""


def capture_inference(response: Any, *, function_name: str | None) -> InferenceUsage:
    """Build the :class:`InferenceUsage` for one ``on_inference`` response.

    Reads the gateway's ``inference_id`` / ``episode_id`` / ``usage`` and,
    where the response carries them, ``variant_name``, ``finish_reason``
    and the ``tool_call`` content blocks — by the model's own name when
    the agent did not recognise it.
    """
    finish_reason = getattr(response, "finish_reason", None)
    return InferenceUsage(
        inference_id=str(response.inference_id),
        episode_id=str(response.episode_id),
        function_name=function_name,
        variant_name=getattr(response, "variant_name", None),
        usage=Usage.from_tz(response.usage),
        tool_calls=tuple(
            block.name or block.raw_name
            for block in getattr(response, "content", None) or []
            if isinstance(block, ToolCall)
        ),
        finish_reason=(
            None
            if finish_reason is None
            else str(getattr(finish_reason, "value", finish_reason))
        ),
        at=datetime.now(UTC),
    )


def sum_usages(inferences: Sequence[InferenceUsage]) -> RunUsage:
    """Sum per-inference usages the way :class:`RunUsage` does (``None`` → ``0``)."""
    priced = [i.usage.cost for i in inferences if i.usage.cost is not None]
    return RunUsage(
        input_tokens=sum(i.usage.input_tokens for i in inferences),
        output_tokens=sum(i.usage.output_tokens for i in inferences),
        cost=float(sum(priced)),
        provider_cache_read_input_tokens=sum(
            i.usage.provider_cache_read_input_tokens or 0 for i in inferences
        ),
        provider_cache_write_input_tokens=sum(
            i.usage.provider_cache_write_input_tokens or 0 for i in inferences
        ),
        cost_is_priced=bool(priced),
    )


async def record_best_effort(
    ledger: UsageLedger, record: RunUsageRecord, *, timeout: float
) -> None:
    """Hand ``record`` to ``ledger`` without letting the ledger affect the run.

    A ledger that raises, or has not returned after ``timeout`` seconds,
    is logged by run id and exception class — never by message, which a
    store may fill with its own query text — and the caller continues.
    """
    try:
        await asyncio.wait_for(ledger.record(record), timeout)
    except Exception as exc:
        logger.warning(
            "usage ledger record failed for run %s: %s",
            record.run_id,
            type(exc).__name__,
        )


async def record_single_inference(
    ledger: UsageLedger,
    response: Any,
    *,
    function_name: str | None,
    tags: Mapping[str, str] | None,
    started_at: datetime,
) -> None:
    """Record a run that was one direct inference, outside any chain.

    The shape :func:`~t0_agents.core.structured.run_structured_inference`
    produces: one :class:`InferenceUsage`, ``turns=1``,
    ``outcome="completed"``, ``tags`` as the caller passed them, written
    through :func:`record_best_effort` with :data:`DEFAULT_TIMEOUT`.
    """
    inference = capture_inference(response, function_name=function_name)
    await record_best_effort(
        ledger,
        RunUsageRecord(
            run_id=str(uuid7()),
            episode_id=inference.episode_id,
            function_name=function_name,
            tags=dict(tags or {}),
            inferences=(inference,),
            usage=sum_usages((inference,)),
            turns=1,
            outcome="completed",
            error_class=None,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
        timeout=DEFAULT_TIMEOUT,
    )


class UsageLedgerMiddleware:
    """Record one :class:`RunUsageRecord` per run onto a :class:`UsageLedger`.

    Each inference is captured as it happens through ``ctx.on_inference``
    — the ids are the gateway's, so a corrective-retry sentinel never
    reaches the ledger and a stream that ended without a usage chunk adds
    no row.  The record is built in ``finally`` and written on every
    outcome: ``"completed"``, ``"failed"`` (the exception's class name is
    recorded and the exception propagates) and ``"cancelled"`` (the
    cancellation propagates).  Unlike :class:`MemoryMiddleware`, which
    saves only a run that produced a result, a run that failed before its
    first inference is recorded too.  The write goes through
    :func:`record_best_effort`: bounded by ``timeout`` seconds, logged and
    swallowed on failure, so the ledger cannot change the run's result.
    ``tags`` are the merged map the gateway received.  A delegated
    sub-agent's inferences are recorded by its own chain, never folded in.
    """

    name = "usage_ledger"

    def __init__(
        self, ledger: UsageLedger, *, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self._ledger = ledger
        self._timeout = timeout

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        run_id = str(uuid7())
        started_at = datetime.now(UTC)
        function_name = ctx.config.function_name
        inferences: list[InferenceUsage] = []

        async def capture(response: Any) -> None:
            inferences.append(
                capture_inference(response, function_name=function_name)
            )

        ctx.chain_on_inference(capture)

        result: AgentResult[Any] | None = None
        outcome: RunOutcome = "failed"
        error_class: str | None = None
        try:
            result = await next_(ctx)
            outcome = "completed"
            return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            error_class = type(exc).__name__
            raise
        finally:
            agent = ctx.run.agent if ctx.run is not None else None
            tags = (
                agent._merge_tags(ctx.tags)
                if agent is not None
                else dict(ctx.tags or {})
            )
            record = RunUsageRecord(
                run_id=run_id,
                episode_id=inferences[0].episode_id if inferences else ctx.episode_id,
                function_name=function_name,
                tags=tags,
                inferences=tuple(inferences),
                usage=sum_usages(inferences),
                turns=result.turns if result is not None else len(inferences),
                outcome=outcome,
                error_class=error_class,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            await record_best_effort(self._ledger, record, timeout=self._timeout)
