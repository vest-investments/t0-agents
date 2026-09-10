"""Ephemeral, dict-backed UsageLedger implementation."""

from __future__ import annotations

from t0_agents.usage.records import RunUsageRecord


class InMemoryUsageLedger:
    """Ephemeral ledger that keeps every record in process memory.

    The double for tests and single-process tools: :attr:`runs` returns the
    records in the order they were recorded.  A repeated ``run_id`` is
    dropped, as the protocol's idempotency requires.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunUsageRecord] = {}

    @property
    def runs(self) -> tuple[RunUsageRecord, ...]:
        return tuple(self._runs.values())

    async def record(self, run: RunUsageRecord) -> None:
        self._runs.setdefault(run.run_id, run)
