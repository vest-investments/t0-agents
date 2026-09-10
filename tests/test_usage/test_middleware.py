"""``UsageLedgerMiddleware`` through ``Agent.invoke`` — what a ledger receives."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from time import perf_counter
from typing import Any, cast
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, TensorZeroError
from tensorzero.types import FinishReason, Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import RunCancelledError
from t0_agents.core.middleware import CostTrackingMiddleware
from t0_agents.core.types import ToolProvider
from t0_agents.usage import (
    InMemoryUsageLedger,
    RunUsageRecord,
    UsageLedgerMiddleware,
)
from t0_agents.usage.middleware import capture_inference
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage

EPISODE = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


class _RaisingLedger(InMemoryUsageLedger):
    async def record(self, run: RunUsageRecord) -> None:
        await super().record(run)
        raise RuntimeError("secret-content")


class _SlowLedger(InMemoryUsageLedger):
    async def record(self, run: RunUsageRecord) -> None:
        await asyncio.sleep(1.0)
        await super().record(run)


class _ParkedT0:
    """Gateway whose inference never answers — the run cancels mid-inference."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def inference(self, **kwargs: Any) -> Any:
        self.entered.set()
        await asyncio.Event().wait()


def _add_tool_client() -> FakeMCPClient:
    return FakeMCPClient(
        tools=[
            MCPTool(
                name="add",
                description="add two ints",
                inputSchema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            ),
        ],
        call_results={
            "add": CallToolResult(
                content=[TextContent(type="text", text="3")],
                structured_content=None,
                meta=None,
                data=None,
                is_error=False,
            ),
        },
    )


def _tool_call(name: str | None, raw_name: str) -> ToolCall:
    return ToolCall(
        id="tc-1",
        raw_name=raw_name,
        name=name,
        raw_arguments='{"a": 1, "b": 2}',
        arguments={"a": 1, "b": 2},
    )


def _agent(
    t0: Any,
    ledger: InMemoryUsageLedger,
    *,
    timeout: float = 2.0,
    client: FakeMCPClient | None = None,
) -> Agent:
    return Agent(
        AgentConfig(
            name="acc",
            function_name="agent",
            middleware=[
                CostTrackingMiddleware(),
                UsageLedgerMiddleware(ledger, timeout=timeout),
            ],
        ),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client or _add_tool_client()),
    )


def _two_turn_responses(first_call: ToolCall) -> list[FakeChatResponse]:
    return [
        FakeChatResponse(
            content=[first_call],
            episode_id=EPISODE,
            usage=FakeUsage(
                input_tokens=10,
                output_tokens=5,
                cost=0.001,
                provider_cache_read_input_tokens=4,
            ),
        ),
        FakeChatResponse(
            content=[Text(type="text", text="3")],
            episode_id=EPISODE,
            usage=FakeUsage(
                input_tokens=20,
                output_tokens=2,
                cost=0.002,
                provider_cache_read_input_tokens=8,
                provider_cache_write_input_tokens=3,
            ),
        ),
    ]


class TestOneRecordPerRun:
    async def test_a_tool_loop_yields_one_inference_row_per_gateway_response(
        self,
    ) -> None:
        ledger = InMemoryUsageLedger()
        responses = _two_turn_responses(_tool_call("add", "add"))
        agent = _agent(FakeT0(list(responses)), ledger)
        run = agent.invoke(
            "compute 1+2", tags={"k": "v"}
        )
        result = await run.result()

        assert len(ledger.runs) == 1
        record = ledger.runs[0]
        assert [i.inference_id for i in record.inferences] == [
            str(r.inference_id) for r in responses
        ]
        assert record.usage == run.usage()
        assert [i.tool_calls for i in record.inferences] == [("add",), ()]
        assert record.turns == result.turns == 2
        assert record.outcome == "completed"
        assert record.error_class is None
        assert record.episode_id == str(EPISODE)
        assert record.function_name == "agent"
        assert record.tags == {"t0_agents.agent": "acc", "k": "v"}
        assert record.started_at.tzinfo is UTC
        assert record.started_at <= record.finished_at
        assert [i.usage.cost for i in record.inferences] == [0.001, 0.002]
        assert all(
            i.variant_name is None and i.finish_reason is None
            for i in record.inferences
        )

    async def test_an_unknown_tool_is_recorded_by_the_name_the_model_wrote(
        self,
    ) -> None:
        ledger = InMemoryUsageLedger()
        t0 = FakeT0(list(_two_turn_responses(_tool_call(None, "ghost"))))
        agent = _agent(t0, ledger)
        await agent.invoke("call ghost").result()

        assert [i.tool_calls for i in ledger.runs[0].inferences] == [("ghost",), ()]

    async def test_a_corrective_retry_sentinel_never_reaches_the_ledger(
        self,
    ) -> None:
        ledger = InMemoryUsageLedger()
        real = FakeChatResponse(
            content=[Text(type="text", text="ok")], usage=FakeUsage(input_tokens=1)
        )
        t0 = FakeT0(
            [
                TensorZeroError(status_code=400, text="tool 'x' not in request.tools"),
                real,
            ]
        )
        agent = _agent(t0, ledger)
        result = await agent.invoke("hi").result()

        assert result.inference_ids == [
            "error-recoverable-turn-1",
            str(real.inference_id),
        ]
        assert [i.inference_id for i in ledger.runs[0].inferences] == [
            str(real.inference_id)
        ]


class TestEveryOutcomeIsRecorded:
    async def test_a_failed_run_records_the_exception_class_and_still_raises(
        self,
    ) -> None:
        ledger = InMemoryUsageLedger()
        agent = _agent(FakeT0([RuntimeError("boom")]), ledger)
        run = agent.invoke("hi")
        with pytest.raises(RuntimeError, match="boom"):
            await run.result()

        record = ledger.runs[0]
        assert record.outcome == "failed"
        assert record.error_class == "RuntimeError"
        assert record.inferences == ()
        assert record.turns == 0
        assert record.episode_id is None

    async def test_a_cancelled_run_records_cancelled_and_still_cancels(self) -> None:
        ledger = InMemoryUsageLedger()
        t0 = _ParkedT0()
        agent = _agent(t0, ledger, client=FakeMCPClient())
        run = agent.invoke("hi")
        waiter = asyncio.create_task(run.result())
        await asyncio.wait_for(t0.entered.wait(), 5)

        await run.cancel()

        with pytest.raises(RunCancelledError):
            await waiter
        assert [r.outcome for r in ledger.runs] == ["cancelled"]
        assert ledger.runs[0].error_class is None


class TestTheLedgerCannotAffectTheRun:
    async def test_a_raising_ledger_is_logged_by_run_id_and_class_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ledger = _RaisingLedger()
        t0 = FakeT0([FakeChatResponse(content=[Text(type="text", text="ok")])])
        with caplog.at_level(logging.WARNING, logger="t0_agents.usage.middleware"):
            agent = _agent(t0, ledger)
            result = await agent.invoke("hi").result()

        assert result.response == "ok"
        warnings = [
            r for r in caplog.records if r.name == "t0_agents.usage.middleware"
        ]
        assert len(warnings) == 1
        assert ledger.runs[0].run_id in warnings[0].getMessage()
        assert "RuntimeError" in warnings[0].getMessage()
        assert warnings[0].exc_info is None
        assert "secret-content" not in caplog.text

    async def test_a_hanging_ledger_is_cut_off_at_the_timeout(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ledger = _SlowLedger()
        t0 = FakeT0([FakeChatResponse(content=[Text(type="text", text="ok")])])
        started = perf_counter()
        with caplog.at_level(logging.WARNING, logger="t0_agents.usage.middleware"):
            agent = _agent(t0, ledger, timeout=0.05)
            result = await agent.invoke("hi").result()

        assert result.response == "ok"
        assert perf_counter() - started < 0.5
        assert ledger.runs == ()
        assert "TimeoutError" in caplog.text


class TestCaptureInference:
    def test_a_finish_reason_enum_is_stored_as_its_value(self) -> None:
        response = FakeChatResponse(content=[])
        response.finish_reason = FinishReason.STOP  # type: ignore[attr-defined]
        assert capture_inference(response, function_name="agent").finish_reason == "stop"

    def test_a_response_without_variant_or_finish_reason_records_none(self) -> None:
        response = FakeChatResponse(content=[Text(type="text", text="x")])
        captured = capture_inference(response, function_name=None)
        assert captured.variant_name is None
        assert captured.finish_reason is None
        assert captured.tool_calls == ()
        assert captured.usage.cost is None
