"""The :class:`UsageLedger` protocol — where a run's usage is persisted."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from t0_agents.usage.records import RunUsageRecord


@runtime_checkable
class UsageLedger(Protocol):
    """Repository-style sink for :class:`RunUsageRecord` values.

    The library calls :meth:`record` exactly once per run, after the run
    has ended — completed, failed or cancelled alike — from inside the
    middleware chain, so calls for concurrent runs overlap.  A ledger must
    be idempotent on :attr:`RunUsageRecord.run_id`: a store that receives
    a record twice keeps one row.  The caller bounds each call with a
    timeout and logs a failure instead of raising it, so a slow or broken
    store cannot change a run's result — and a ledger has no way to fail
    a run.  There is no read side: a consumer queries the store it owns.
    """

    async def record(self, run: RunUsageRecord) -> None: ...
