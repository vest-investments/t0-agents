"""Parity tests: Agent.run() must produce identical results to run_agent.

Each test here mirrors a corresponding test in ``tests/test_mcp/test_loop.py``
(class ``TestRunAgent``), but exercises the code path through ``Agent.run()``
instead of calling ``run_agent`` directly. If these tests diverge from the
loop tests, the Agent wrapper is not faithfully delegating.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, Thought, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

# ---------------------------------------------------------------------------
# Fixture builders (local)
# ---------------------------------------------------------------------------


def _mcp_tool(name: str = "add") -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _make_result(
    *,
    data: Any = None,
    is_error: bool = False,
) -> CallToolResult:
    return CallToolResult(
        content=[],
        structured_content=None,
        meta=None,
        data=data,
        is_error=is_error,
    )


def _tool_call(
    *,
    name: str | None = "add",
    raw_name: str = "add",
    call_id: str = "call-1",
    arguments: dict[str, Any] | None = None,
    raw_arguments: str = '{"a": 2, "b": 3}',
) -> ToolCall:
    args = arguments if arguments is not None else {"a": 2, "b": 3}
    return ToolCall(
        id=call_id,
        raw_arguments=raw_arguments,
        raw_name=raw_name,
        arguments=args,
        name=name,
    )


def _agent(t0: FakeT0, client: FakeMCPClient, **overrides: Any) -> Agent:
    defaults: dict[str, Any] = {"name": "parity", "function_name": "agent"}
    defaults.update(overrides)
    return Agent(
        AgentConfig(**defaults),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client),
    )


# ---------------------------------------------------------------------------
# Parity: TestRunAgent mirrors
# ---------------------------------------------------------------------------


class TestAgentParity:
    async def test_text_only_first_turn(self) -> None:
        """Mirrors test_terminates_on_text_only_first_turn."""
        episode = uuid4()
        inference = uuid4()
        t0 = FakeT0([
            FakeChatResponse(
                content=[Text(text="hello world")],
                episode_id=episode,
                inference_id=inference,
            ),
        ])
        agent = _agent(t0, FakeMCPClient())
        result = await agent.run("Just say hi.")

        assert result.response == "hello world"
        assert result.episode_id == str(episode)
        assert result.inference_ids == [str(inference)]
        assert result.turns == 1

    async def test_thought_only_response(self) -> None:
        """Mirrors test_filters_thought_blocks_from_final_response."""
        t0 = FakeT0([
            FakeChatResponse(
                content=[Thought(text="reasoning..."), Text(text="answer")]
            ),
        ])
        agent = _agent(t0, FakeMCPClient())
        result = await agent.run("Q?")

        assert result.response == "answer"
        assert "reasoning" not in result.response

    async def test_tool_call_round_trip(self) -> None:
        """Mirrors test_single_turn_assembles_user_assistant_user_tools_history."""
        episode = uuid4()
        tc = _tool_call(name="add", call_id="c1")
        t0 = FakeT0([
            FakeChatResponse(content=[tc], episode_id=episode),
            FakeChatResponse(
                content=[Text(text="the answer is 5")], episode_id=episode
            ),
        ])
        client = FakeMCPClient(
            tools=[_mcp_tool("add")],
            call_results={"add": _make_result(data=5)},
        )
        agent = _agent(t0, client)
        result = await agent.run("compute 2+3")

        assert result.response == "the answer is 5"
        assert result.turns == 2

        # Verify message history on second inference call.
        second_messages = t0.calls[1]["input"]["messages"]
        assert len(second_messages) == 3
        assert second_messages[0]["role"] == "user"
        assert second_messages[1]["role"] == "assistant"
        assert second_messages[2]["role"] == "user"
        results_block = second_messages[2]["content"]
        assert len(results_block) == 1
        assert results_block[0].name == "add"

    async def test_max_turns_exceeded(self) -> None:
        """Mirrors test_max_turns_raises_max_turns_exceeded_error."""
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call(call_id="c1")]),
            FakeChatResponse(content=[_tool_call(call_id="c2")]),
        ])
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        agent = _agent(t0, client, max_turns=2)

        try:
            await agent.run("loop forever")
        except MaxTurnsExceededError as e:
            assert e.max_turns == 2
        else:
            raise AssertionError("expected MaxTurnsExceededError")

    async def test_parallel_tools(self) -> None:
        """Mirrors test_parallel_tools_flag_uses_parallel_executor."""
        completion_order: list[str] = []

        async def slow_then_fast(name: str | None) -> None:
            assert name is not None
            await asyncio.sleep(0.05 if name == "add" else 0.01)
            completion_order.append(name)

        tc_a = _tool_call(name="add", raw_name="add", call_id="a")
        tc_b = _tool_call(name="greet", raw_name="greet", call_id="b")
        t0 = FakeT0([
            FakeChatResponse(content=[tc_a, tc_b]),
            FakeChatResponse(content=[Text(text="done")]),
        ])
        client = FakeMCPClient(
            tools=[_mcp_tool("add"), _mcp_tool("greet")],
            call_results={
                "add": _make_result(data=1),
                "greet": _make_result(data="hi"),
            },
            side_effect=slow_then_fast,
        )
        agent = _agent(t0, client, parallel_tools=True)
        await agent.run("do both")

        # Input order preserved in results.
        last = t0.calls[1]["input"]["messages"][-1]
        assert last["role"] == "user"
        assert [r.name for r in last["content"]] == ["add", "greet"]
        # Concurrency: greet finished before add.
        assert completion_order == ["greet", "add"]

    async def test_dryrun_forwarded(self) -> None:
        """Mirrors test_dryrun_kwarg_forwards_to_inference."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _agent(t0, FakeMCPClient(), dryrun=True)
        await agent.run("hello")

        assert t0.calls[0]["dryrun"] is True
