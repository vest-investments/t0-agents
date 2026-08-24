"""Robustness and error handling (unit tests).

The error contract itself lives in ``src/t0_agents/mcp/loop.py``. These tests
validate that every error path (ToolError, McpError, invalid JSON, timeout,
retry, ValidationError-through-ToolError, max_turns, and None-description
tools) behaves correctly.

All tests reuse ``FakeT0`` / ``FakeChatResponse`` from ``tests._fake_t0`` and
the ``FakeMCPClient`` / ``_tool_call`` / ``_make_result`` / ``_mcp_error`` /
``_mcp_tool`` helpers defined once in ``tests.test_mcp.test_loop``.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import run_agent
from t0_agents.core.exceptions import MaxTurnsExceededError
from tests._fake_t0 import FakeChatResponse, FakeT0
from tests.test_mcp.test_loop import (
    FakeMCPClient,
    _make_result,
    _mcp_error,
    _mcp_tool,
    _tool_call,
)

# --- TestRobustness ---------------------------------------------------------


class TestRobustness:
    """Error handling validated against the error contract."""

    async def test_tool_error_caught(self) -> None:
        """``ToolError`` from FastMCP is caught and surfaced to the LLM
        as a ``ToolResult`` with an error payload. ``run_agent`` does not crash.

        Two-turn flow: turn 1 emits a ``ToolCall`` that hits a ``ToolError``;
        turn 2 emits a terminating ``Text`` block. We assert the loop reached
        turn 2 and the second inference saw the error ``ToolResult`` in its
        message history.
        """
        tc = _tool_call(name="fail_always", raw_name="fail_always", call_id="c1")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="I noticed the tool failed.")]),
            ]
        )
        client = FakeMCPClient(
            tools=[_mcp_tool("fail_always")],
            call_results={"fail_always": ToolError("boom")},
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "trigger the failing tool",
            "agent",
        )
        assert result.turns == 2
        assert result.response == "I noticed the tool failed."

        # Second inference saw the error ToolResult packed into a user message.
        second_messages = t0.calls[1]["input"]["messages"]
        last = second_messages[-1]
        assert last["role"] == "user"
        error_block = last["content"][0]
        assert error_block.name == "fail_always"
        assert error_block.result.startswith("ERROR: tool execution failed:")
        assert "boom" in error_block.result

    async def test_mcp_error_caught(self) -> None:
        """``McpError`` from mcp (timeout, protocol) is caught and surfaced as
        an error ``ToolResult``. ``run_agent`` does not crash.

        We raise a plain (non-timeout) McpError to distinguish from the timeout
        case: the catch chain routes both through the same branch but the
        meaning differs per the integration contract.
        """
        tc = _tool_call(call_id="c1")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="protocol error noted")]),
            ]
        )
        client = FakeMCPClient(
            call_results={"add": _mcp_error("transport closed", code=-32603)},
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "try the add tool",
            "agent",
        )
        assert result.turns == 2

        error_block = t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert error_block.result.startswith("ERROR: MCP protocol error")
        assert "transport closed" in error_block.result

    async def test_invalid_json_detected(self) -> None:
        """``block.arguments is None`` (LLM produced invalid JSON) is
        detected and reported as an error ``ToolResult`` without invoking the
        underlying tool.
        """
        tc = _tool_call(null_args=True, raw_arguments="{this is not json")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="I will try again next time.")]),
            ]
        )
        client = FakeMCPClient(call_results={})  # empty: call_tool must not fire
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        assert result.turns == 2

        # The tool was never called (invalid JSON arguments contract).
        assert client.call_tool_invocations == []

        error_block = t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert error_block.result.startswith("ERROR: invalid JSON arguments")
        assert "{this is not json" in error_block.result

    async def test_trailing_brace_is_salvaged_and_dispatched(self) -> None:
        """A gateway-zeroed call whose only defect is one trailing ``}`` executes
        with its full arguments and feeds the model a real result, not an error."""
        tc = _tool_call(null_args=True, raw_arguments='{"a": 2, "b": 3}}')
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="5")]),
            ]
        )
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        assert result.turns == 2
        assert client.call_tool_invocations == [("add", {"a": 2, "b": 3}, 30.0)]

        tool_block = t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert tool_block.result == "5"

    async def test_timeout_surfaces_as_mcperror(self) -> None:
        """``Client.call_tool(..., timeout=N)`` raises ``McpError``
        (code ``-32000``) on timeout, which the loop catches via the McpError
        branch. This test pins the timeout-specific flavor: the error code
        and the "timed out" message reach the LLM intact.
        """
        tc = _tool_call(call_id="c1")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="the tool was too slow")]),
            ]
        )
        client = FakeMCPClient(
            call_results={"add": _mcp_error("timed out", code=-32000)},
        )
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
            tool_timeout=0.5,
        )
        # Loop forwarded our timeout value to call_tool.
        assert client.call_tool_invocations[0][2] == 0.5

        error_block = t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert error_block.result.startswith("ERROR: MCP protocol error")
        assert "timed out" in error_block.result

    async def test_loop_continues_after_error_and_llm_retries(self) -> None:
        """After an error ``ToolResult``, the LLM can retry -- ``run_agent``
        keeps the conversation alive.

        Scenario pinned:
        * Turn 1 response: ``ToolCall(fail_always)`` → ``ToolError`` → error result
        * Turn 2 response: ``ToolCall(add)`` → success → normal result
        * Turn 3 response: ``Text("5")`` → loop terminates

        We assert ``turns == 3``, the final response is non-empty, and
        turn 2's inference input carries the error ``ToolResult`` from the
        failing first call so the LLM had context to retry.
        """
        tc_fail = _tool_call(name="fail_always", raw_name="fail_always", call_id="c1")
        tc_add = _tool_call(name="add", raw_name="add", call_id="c2")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc_fail]),
                FakeChatResponse(content=[tc_add]),
                FakeChatResponse(content=[Text(text="the answer is 5")]),
            ]
        )
        client = FakeMCPClient(
            tools=[_mcp_tool("fail_always"), _mcp_tool("add")],
            call_results={
                "fail_always": ToolError("intentional"),
                "add": _make_result(data=5),
            },
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "try fail_always, then the add tool",
            "agent",
        )
        assert result.turns == 3
        assert result.response == "the answer is 5"

        # Turn 2's inference carried the failing result so the LLM saw the error.
        second_inference_messages = t0.calls[1]["input"]["messages"]
        error_result_block = second_inference_messages[-1]["content"][0]
        assert error_result_block.name == "fail_always"
        assert "intentional" in error_result_block.result

        # Turn 3's inference carried both results — the error AND the successful
        # retry — proving the loop kept accumulating history across the retry.
        third_inference_messages = t0.calls[2]["input"]["messages"]
        last_user = third_inference_messages[-1]
        assert last_user["role"] == "user"
        assert last_user["content"][0].name == "add"
        assert last_user["content"][0].result == "5"

    async def test_validation_error_surfaces_through_tool_error(self) -> None:
        """A pydantic ``ValidationError`` on the FastMCP server side (invalid
        tool arguments) becomes an MCP error response which ``Client.call_tool``
        re-raises as ``ToolError`` -- flowing through the ToolError branch
        without needing a separate catch.

        The FastMCP SDK source (``fastmcp/client/mixins/tools.py:387-393``)
        shows the server pre-converts all pydantic validation failures to
        ``ToolError`` before they reach the client. We pin the contract so a
        future SDK change that raises ``ValidationError`` directly would fail
        this test loudly.
        """
        tc = _tool_call(call_id="c1")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc]),
                FakeChatResponse(content=[Text(text="I saw the validation error.")]),
            ]
        )
        client = FakeMCPClient(
            call_results={
                "add": ToolError(
                    "Input validation error: 'b' must be a valid int, got 'foo'"
                )
            },
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "add 2 + foo (invalid)",
            "agent",
        )
        assert result.turns == 2

        error_block = t0.calls[1]["input"]["messages"][-1]["content"][0]
        assert error_block.result.startswith("ERROR: tool execution failed:")
        assert "validation error" in error_block.result.lower()

    async def test_max_turns_raises_with_correct_limit(self) -> None:
        """When the LLM keeps returning tool calls past ``max_turns``,
        ``run_agent`` raises ``MaxTurnsExceededError``.

        ``e.max_turns`` carries the configured limit so callers can surface a
        useful error message.
        """
        custom_limit = 3
        t0 = FakeT0(
            [
                FakeChatResponse(content=[_tool_call(call_id="c1")]),
                FakeChatResponse(content=[_tool_call(call_id="c2")]),
                FakeChatResponse(content=[_tool_call(call_id="c3")]),
            ]
        )
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        try:
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], client),
                "keep looping",
                "agent",
                max_turns=custom_limit,
            )
        except MaxTurnsExceededError as e:
            assert e.max_turns == custom_limit
            assert str(custom_limit) in str(e)
        else:
            raise AssertionError("expected MaxTurnsExceededError")

    async def test_none_description_round_trip(self) -> None:
        """A tool with ``description=None`` flows through ``convert_tools``
        to T0's ``additional_tools`` as ``description=""`` — the loop does not
        crash and T0 receives a well-formed payload.

        This pins the end-to-end integration: the MCPTool has ``description=None``,
        ``run_agent`` discovers it, ``convert_tools`` coerces to ``""``, and
        the captured ``additional_tools`` on the first inference reflects the
        fallback. Mock-side we only guarantee no crash and the correct adapter
        output.
        """
        none_desc_tool = MCPTool(
            name="no_desc_tool",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")])])
        client = FakeMCPClient(tools=[none_desc_tool])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "just say hi",
            "agent",
        )
        assert result.response == "hi"
        # The one inference call saw our tool with an empty-string description.
        forwarded_tools = t0.calls[0]["additional_tools"]
        assert len(forwarded_tools) == 1
        assert forwarded_tools[0]["name"] == "no_desc_tool"
        assert forwarded_tools[0]["description"] == ""


# --- Sanity: retry flow preserves episode_id --------------------------------


class TestRobustnessEpisode:
    """A small companion to TestRobustness: episode_id is preserved across
    error recovery. Pinned separately so the retry test above stays focused
    on the retry mechanism."""

    async def test_episode_id_survives_error_then_success(self) -> None:
        """The error path shares the same episode_id as the successful retry."""
        episode = uuid4()
        tc_fail = _tool_call(name="fail_always", raw_name="fail_always", call_id="c1")
        tc_add = _tool_call(name="add", raw_name="add", call_id="c2")
        t0 = FakeT0(
            [
                FakeChatResponse(content=[tc_fail], episode_id=episode),
                FakeChatResponse(content=[tc_add], episode_id=episode),
                FakeChatResponse(content=[Text(text="done")], episode_id=episode),
            ]
        )
        client = FakeMCPClient(
            tools=[_mcp_tool("fail_always"), _mcp_tool("add")],
            call_results={
                "fail_always": ToolError("boom"),
                "add": _make_result(data=7),
            },
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "two-step",
            "agent",
        )
        assert result.episode_id == str(episode)
        assert result.turns == 3
