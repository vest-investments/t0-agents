"""E4 — tool progress crosses the real FastMCP boundary.

``execute_tool_calls`` (and the parallel variant) accept a
``progress_callback``; the loop binds it per ``ToolCall`` block into a
fastmcp ``progress_handler``, so server-side ``ctx.report_progress(...)``
notifications arrive tagged with the block id.  These tests run against
the shared in-memory FastMCP server — the genuine notification path, no
transport fakes.
"""

from __future__ import annotations

from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.mcp.loop import execute_tool_calls, execute_tool_calls_parallel, run_agent
from t0_agents.tooling import FilteredToolProvider
from tests._fake_t0 import FakeChatResponse, FakeT0
from tests.server import mcp

Notification = tuple[str, float, float | None, str | None]


def _progress_call(steps: int, call_id: str = "c1") -> ToolCall:
    return ToolCall(
        id=call_id,
        raw_name="count_with_progress",
        raw_arguments=f'{{"steps": {steps}}}',
        name="count_with_progress",
        arguments={"steps": steps},
    )


class _Recorder:
    def __init__(self) -> None:
        self.notifications: list[Notification] = []

    async def __call__(
        self, tool_call_id: str, progress: float, total: float | None, message: str | None
    ) -> None:
        self.notifications.append((tool_call_id, progress, total, message))


class TestSequentialExecutor:
    async def test_progress_reaches_the_callback_with_the_block_id(self) -> None:
        recorder = _Recorder()
        async with Client(mcp) as client:
            results = await execute_tool_calls(
                client, [_progress_call(3)], 5.0, progress_callback=recorder
            )
        assert len(results) == 1
        assert "counted 3" in str(results[0].result)
        assert recorder.notifications == [
            ("c1", 1.0, 3.0, "step 1"),
            ("c1", 2.0, 3.0, "step 2"),
            ("c1", 3.0, 3.0, "step 3"),
        ]

    async def test_no_callback_is_the_quiet_default(self) -> None:
        async with Client(mcp) as client:
            results = await execute_tool_calls(client, [_progress_call(2)], 5.0)
        assert "counted 2" in str(results[0].result)


class TestParallelExecutor:
    async def test_concurrent_notifications_stay_tagged_per_call(self) -> None:
        recorder = _Recorder()
        blocks = [_progress_call(2, "c1"), _progress_call(3, "c2")]
        async with Client(mcp) as client:
            results = await execute_tool_calls_parallel(
                client, blocks, 5.0, progress_callback=recorder
            )
        assert len(results) == 2
        by_call: dict[str, list[Notification]] = {"c1": [], "c2": []}
        for notification in recorder.notifications:
            by_call[notification[0]].append(notification)
        assert [n[1] for n in by_call["c1"]] == [1.0, 2.0]
        assert [n[1] for n in by_call["c2"]] == [1.0, 2.0, 3.0]


class TestWrapperForwarding:
    async def test_filtered_tool_provider_forwards_progress(self) -> None:
        recorder = _Recorder()
        async with Client(mcp) as client:
            filtered: Any = FilteredToolProvider(client, {"count_with_progress"})
            results = await execute_tool_calls(
                filtered, [_progress_call(2)], 5.0, progress_callback=recorder
            )
        assert "counted 2" in str(results[0].result)
        assert [n[1] for n in recorder.notifications] == [1.0, 2.0]


class TestRunAgent:
    async def test_progress_callback_reaches_run_agent_callers(self) -> None:
        """Parity fix: ``run_agent`` forwards ``progress_callback`` to the
        executors, so server-side progress reaches non-streaming callers —
        historically only the streaming loop wired it."""
        recorder = _Recorder()
        fake_t0 = FakeT0(
            responses=[
                FakeChatResponse(content=[_progress_call(2)]),
                FakeChatResponse(content=[Text(type="text", text="done")]),
            ],
        )
        async with Client(mcp) as client:
            result = await run_agent(
                cast(AsyncTensorZeroGateway, fake_t0),
                client,
                "count",
                "agent",
                tool_timeout=5.0,
                progress_callback=recorder,
            )
        assert result.value == "done"
        assert recorder.notifications == [
            ("c1", 1.0, 2.0, "step 1"),
            ("c1", 2.0, 2.0, "step 2"),
        ]
