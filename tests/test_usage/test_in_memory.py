"""``InMemoryUsageLedger`` — the in-process reference backend."""

from __future__ import annotations

from t0_agents.usage import InMemoryUsageLedger
from tests.test_usage._records import record


class TestInMemoryUsageLedger:
    async def test_records_are_kept_in_order(self) -> None:
        ledger = InMemoryUsageLedger()
        await ledger.record(record("run-1"))
        await ledger.record(record("run-2"))
        assert [r.run_id for r in ledger.runs] == ["run-1", "run-2"]

    async def test_a_repeated_run_id_is_kept_once(self) -> None:
        ledger = InMemoryUsageLedger()
        await ledger.record(record("run-1"))
        await ledger.record(record("run-1"))
        assert len(ledger.runs) == 1
