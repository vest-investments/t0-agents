"""``run_structured_inference(usage_ledger=)`` — a direct JSON call is one run."""

from __future__ import annotations

from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.structured import run_structured_inference
from t0_agents.testing import MockGateway
from t0_agents.usage import InMemoryUsageLedger, RunUsageRecord


class _RaisingLedger(InMemoryUsageLedger):
    async def record(self, run: RunUsageRecord) -> None:
        raise RuntimeError("secret-content")


async def test_a_direct_json_call_is_one_recorded_run() -> None:
    ledger = InMemoryUsageLedger()
    t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
    result = await run_structured_inference(
        cast(AsyncTensorZeroGateway, t0),
        function_name="extract",
        user_message="hello",
        tags={"k": "v"},
        usage_ledger=ledger,
    )

    assert len(ledger.runs) == 1
    record = ledger.runs[0]
    assert [i.inference_id for i in record.inferences] == [
        result.inference_id,
        ] == [str(t0.responses[0].inference_id)]
    assert record.episode_id == result.episode_id
    assert record.function_name == "extract"
    assert record.tags == {"k": "v"}
    assert record.turns == 1
    assert record.outcome == "completed"
    assert record.inferences[0].tool_calls == ()
    assert record.inferences[0].variant_name == "mock_variant"
    assert record.started_at <= record.finished_at


async def test_a_raising_ledger_leaves_the_result_intact() -> None:
    t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
    result = await run_structured_inference(
        cast(AsyncTensorZeroGateway, t0),
        function_name="extract",
        user_message="hello",
        usage_ledger=_RaisingLedger(),
    )
    assert result.parsed == {"a": 1}
