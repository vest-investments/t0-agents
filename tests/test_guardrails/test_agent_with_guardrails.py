"""End-to-end tests for ``Agent`` + guardrails.

Uses the shared ``FakeT0`` / ``FakeMCPClient`` fakes so the tests stay
offline (``uv run pytest -m "not integration"``).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.client.client import CallToolResult
from fastmcp.client.elicitation import ElicitResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import ToolProvider
from t0_agents.guardrails import (
    block_tools,
    max_cost,
    max_tool_calls,
    require_approval,
)
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


def _tool(name: str = "add") -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _tool_call(name: str, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        raw_arguments="{}",
        raw_name=name,
        name=name,
        arguments={},
    )


def _ok(text: str = "ok") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        meta=None,
        data=text,
        is_error=False,
    )


def _make_agent(
    *,
    t0: FakeT0,
    client: FakeMCPClient,
    **cfg: Any,
) -> Agent:
    config = AgentConfig(name="t", function_name="agent", **cfg)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client),
    )


class TestEmptyGuardrailsByteIdentity:
    async def test_no_wrapper_when_guardrails_empty(self) -> None:
        """``guardrails=[]`` must behave byte-identical to Phase B."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")])])
        client = FakeMCPClient()
        agent = _make_agent(t0=t0, client=client, guardrails=[])

        result = await agent.run("go")
        assert result.response == "hi"
        assert result.guardrail_log == []
        # Verify run_agent got on_inference=None (not in FakeT0's call record
        # because it's a loop-level kwarg not forwarded to inference).
        # The behaviour check is the empty guardrail_log.


class TestBlockToolsInAgent:
    async def test_block_tools_short_circuits_execution(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("danger", "c1")]),
                FakeChatResponse(content=[Text(text="done")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("danger")],
            call_results={},  # never called
        )
        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[block_tools(["danger"])],
        )

        result = await agent.run("go")
        assert result.response == "done"
        assert client.call_tool_invocations == []  # never ran
        # Rejection logged.
        assert any(e.kind == "rejection" for e in result.guardrail_log)


class TestMaxToolCallsInAgent:
    async def test_per_run_cap_aborts(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("add", "c1")]),
                FakeChatResponse(content=[_tool_call("add", "c2")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("add")],
            call_results={"add": _ok("5")},
        )
        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[max_tool_calls(per_run=1)],
        )

        with pytest.raises(GuardrailBlockedError, match="Tool-call cap"):
            await agent.run("go")


class TestMaxCostInAgent:
    async def test_cost_accumulation_aborts_next_turn(self) -> None:
        """Turn 1 inference spends the budget; turn 2 tool call aborts."""
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[_tool_call("add", "c1")],
                    usage=FakeUsage(cost=1.0),
                ),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("add")],
            call_results={"add": _ok("5")},
        )
        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[max_cost(per_episode=0.5)],
        )

        with pytest.raises(GuardrailBlockedError, match="Cost limit"):
            await agent.run("go")


class TestRequireApprovalInAgent:
    async def test_accept_path_runs_tool(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("send_email", "c1")]),
                FakeChatResponse(content=[Text(text="sent")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("send_email")],
            call_results={"send_email": _ok("sent")},
        )

        async def _accept(*_: Any) -> ElicitResult[Any]:
            return ElicitResult(action="accept", content={})

        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[require_approval(tools=["send_email"], handler=_accept)],
        )

        result = await agent.run("go")
        assert result.response == "sent"
        assert client.call_tool_invocations == [("send_email", {}, 30.0)]

    async def test_decline_path_short_circuits_without_calling_tool(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("send_email", "c1")]),
                FakeChatResponse(content=[Text(text="noted")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("send_email")],
            call_results={},  # never called
        )

        async def _decline(*_: Any) -> ElicitResult[Any]:
            return ElicitResult(action="decline")

        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[require_approval(tools=["send_email"], handler=_decline)],
        )

        result = await agent.run("go")
        assert result.response == "noted"
        assert client.call_tool_invocations == []
        rejections = [e for e in result.guardrail_log if e.kind == "rejection"]
        assert len(rejections) == 1
        assert rejections[0].tool_name == "send_email"


class TestAfterRunWithFatalAbort:
    async def test_after_run_called_with_none_on_abort(self) -> None:
        seen: list[Any] = []

        class _SpyGuardrail:
            name = "spy"

            async def before_run(self, ctx: Any) -> None: ...

            async def before_tool(self, ctx: Any, tc: Any) -> str | None:
                raise GuardrailBlockedError("stop")

            async def after_tool(
                self, ctx: Any, tc: Any, result_str: str
            ) -> None: ...

            async def after_run(self, ctx: Any, result: Any) -> None:
                seen.append(result)

        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("add", "c1")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("add")],
            call_results={"add": _ok()},
        )
        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[cast(Any, _SpyGuardrail())],
        )

        with pytest.raises(GuardrailBlockedError):
            await agent.run("go")
        # after_run fires even though run_agent aborted.
        assert seen == [None]


class TestContextIsolationAcrossRuns:
    async def test_counter_resets_between_runs(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call("add", "c1")]),
                FakeChatResponse(content=[Text(text="first done")]),
                FakeChatResponse(content=[_tool_call("add", "c2")]),
                FakeChatResponse(content=[Text(text="second done")]),
            ],
        )
        client = FakeMCPClient(
            tools=[_tool("add")],
            call_results={"add": _ok("5")},
        )
        agent = _make_agent(
            t0=t0,
            client=client,
            guardrails=[max_tool_calls(per_run=1)],
        )

        r1 = await agent.run("one")
        assert r1.response == "first done"
        # Second run starts from a fresh GuardrailContext — would raise
        # if state leaked.
        r2 = await agent.run("two")
        assert r2.response == "second done"
