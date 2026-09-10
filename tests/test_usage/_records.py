"""A fully populated record pair the backend tests read back by value."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from t0_agents.core.types import RunUsage, Usage
from t0_agents.usage import InferenceUsage, RunUsageRecord

AT = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def record(run_id: str = "run-1", *, episode_id: str = "ep-1") -> RunUsageRecord:
    first = InferenceUsage(
        inference_id=f"{run_id}-inf-1",
        episode_id=episode_id,
        function_name="agent",
        variant_name="v1",
        usage=Usage(input_tokens=10, output_tokens=1),
        tool_calls=("add", "ghost"),
        finish_reason="tool_call",
        at=AT,
    )
    second = InferenceUsage(
        inference_id=f"{run_id}-inf-2",
        episode_id=episode_id,
        function_name="agent",
        variant_name=None,
        usage=Usage(
            input_tokens=20,
            output_tokens=2,
            provider_cache_read_input_tokens=4,
            cost=0.5,
        ),
        tool_calls=(),
        finish_reason="stop",
        at=AT + timedelta(milliseconds=500),
    )
    return RunUsageRecord(
        run_id=run_id,
        episode_id=episode_id,
        function_name="agent",
        tags={"k": "v"},
        inferences=(first, second),
        usage=RunUsage(
            input_tokens=30,
            output_tokens=3,
            cost=0.5,
            provider_cache_read_input_tokens=4,
            cost_is_priced=True,
        ),
        turns=2,
        outcome="completed",
        error_class=None,
        started_at=AT,
        finished_at=AT + timedelta(seconds=1),
    )
