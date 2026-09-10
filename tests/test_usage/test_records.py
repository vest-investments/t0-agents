"""``InferenceUsage`` / ``RunUsageRecord`` value semantics and the protocol."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from t0_agents.core.types import RunUsage, Usage
from t0_agents.usage import InferenceUsage, RunUsageRecord, UsageLedger


def _inference() -> InferenceUsage:
    return InferenceUsage(
        inference_id="inf-1",
        episode_id="ep-1",
        function_name="agent",
        variant_name="v1",
        usage=Usage(input_tokens=3, output_tokens=2),
        tool_calls=("add",),
        finish_reason="stop",
        at=datetime.now(UTC),
    )


def _record() -> RunUsageRecord:
    now = datetime.now(UTC)
    return RunUsageRecord(
        run_id="run-1",
        episode_id="ep-1",
        function_name="agent",
        tags={"k": "v"},
        inferences=(_inference(),),
        usage=RunUsage(input_tokens=3, output_tokens=2),
        turns=1,
        outcome="completed",
        error_class=None,
        started_at=now,
        finished_at=now,
    )


class TestRecordsAreValues:
    def test_inference_usage_is_frozen(self) -> None:
        inference = _inference()
        with pytest.raises(dataclasses.FrozenInstanceError):
            inference.inference_id = "other"  # type: ignore[misc]

    def test_run_usage_record_is_frozen(self) -> None:
        record = _record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.outcome = "failed"  # type: ignore[misc]


class TestUsageLedgerProtocol:
    def test_an_object_with_an_async_record_method_satisfies_it(self) -> None:
        class Sink:
            async def record(self, run: RunUsageRecord) -> None: ...

        assert isinstance(Sink(), UsageLedger)

    def test_an_object_without_record_does_not(self) -> None:
        class NotASink:
            async def save(self, run: RunUsageRecord) -> None: ...

        assert not isinstance(NotASink(), UsageLedger)
