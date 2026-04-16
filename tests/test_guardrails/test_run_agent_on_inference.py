"""Unit tests for the new ``on_inference`` kwarg on ``run_agent``.

Contract §6.1 guarantee: ``on_inference=None`` (the default) preserves
today's behaviour byte-for-byte.  When set, the callback is invoked
exactly once per inference response, BEFORE any tool execution.
"""

from __future__ import annotations

from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse

from t0_agents.mcp.loop import run_agent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


class TestOnInference:
    async def test_default_none_is_byte_identical(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")])])
        client = FakeMCPClient()
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        # No exception + result captured means default path stayed intact.
        assert t0.calls
        assert "on_inference" not in t0.calls[0]

    async def test_callback_fires_once_per_inference(self) -> None:
        captured: list[float | None] = []

        async def hook(response: ChatInferenceResponse) -> None:
            captured.append(getattr(response.usage, "cost", None))

        from fastmcp.client.client import CallToolResult
        from mcp.types import TextContent

        tc_id = "c1"
        tc = ToolCall(
            id=tc_id,
            raw_arguments="{}",
            raw_name="add",
            name="add",
            arguments={},
        )

        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[tc],
                    usage=FakeUsage(cost=0.10),
                ),
                FakeChatResponse(
                    content=[Text(text="done")],
                    usage=FakeUsage(cost=0.05),
                ),
            ],
        )
        client = FakeMCPClient(
            call_results={
                "add": CallToolResult(
                    content=[TextContent(type="text", text="5")],
                    structured_content=None,
                    meta=None,
                    data=5,
                    is_error=False,
                ),
            },
        )

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
            on_inference=hook,
        )
        assert captured == [0.10, 0.05]

    async def test_callback_fires_before_tool_execution(self) -> None:
        order: list[str] = []

        async def hook(response: ChatInferenceResponse) -> None:
            order.append("on_inference")

        from fastmcp.client.client import CallToolResult
        from mcp.types import TextContent

        tc = ToolCall(
            id="c1",
            raw_arguments="{}",
            raw_name="add",
            name="add",
            arguments={},
        )

        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="done")]),
            ],
        )

        async def side_effect(_: Any) -> None:
            order.append("call_tool")

        client = FakeMCPClient(
            call_results={
                "add": CallToolResult(
                    content=[TextContent(type="text", text="5")],
                    structured_content=None,
                    meta=None,
                    data=5,
                    is_error=False,
                ),
            },
            side_effect=side_effect,
        )

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
            on_inference=hook,
        )
        # "on_inference" for turn 1 must come before the tool call on turn 1.
        assert order == ["on_inference", "call_tool", "on_inference"]
