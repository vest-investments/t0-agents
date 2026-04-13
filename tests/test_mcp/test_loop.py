"""Unit tests for the agent loop.

Uses hand-written stubs (``FakeT0``, ``FakeMCPClient``, ``FakeChatResponse``)
instead of ``unittest.mock`` so the suite stays mypy-strict-clean and reads
straightforwardly. ``run_agent`` does ``cast(ChatInferenceResponse, raw)``
rather than ``isinstance``-checking, so the duck-typed ``FakeChatResponse``
satisfies the runtime contract without subclassing the real Rust-backed type.

Phase 3 gates covered here:
* F3.4 — termination on text-only response (``TestRunAgent``)
* F3.5 — N tool calls in one response collapse into ONE ``Message(role="user")``
* F3.6 — message history is ``user → assistant → user(tools) → ...``
* F3.3 — ``episode_id`` capture + propagation across turns
* F3.8 — parallel executor preserves order, errors are isolated per call
* M3.1 error contract — ``ToolError`` / ``McpError`` / ``Exception`` / null args
  (the unit tests pin the behavior; F5.x in Phase 5 will validate it as a gate)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ToolError
from mcp import McpError
from mcp.types import ContentBlock, ErrorData, TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, Thought, ToolCall

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.mcp import (
    execute_tool_calls,
    execute_tool_calls_parallel,
    has_tool_calls,
    run_agent,
)

# --- Fixture builders -------------------------------------------------------


def _text(value: str) -> TextContent:
    return TextContent(type="text", text=value)


def _make_result(
    *,
    data: Any = None,
    content: list[ContentBlock] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """Build a synthetic FastMCP CallToolResult (mirrors test_adapter._make_result)."""
    return CallToolResult(
        content=content or [],
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
    null_args: bool = False,
) -> ToolCall:
    """Build a TensorZero ToolCall (mirrors test_adapter._tool_call).

    ``null_args=True`` reproduces the EC-2 case (T0 sets ``arguments=None`` when
    the LLM emitted invalid JSON; ``raw_arguments`` carries the raw string).
    Otherwise ``arguments`` defaults to ``{"a": 2, "b": 3}`` for the ``add`` tool.
    """
    if null_args:
        args: dict[str, Any] | None = None
    else:
        args = arguments if arguments is not None else {"a": 2, "b": 3}
    return ToolCall(
        id=call_id,
        raw_arguments=raw_arguments,
        raw_name=raw_name,
        arguments=args,
        name=name,
    )


def _mcp_tool(name: str = "add") -> MCPTool:
    """Build a minimal mcp.types.Tool for list_tools() stubs."""
    return MCPTool(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _mcp_error(message: str = "boom", code: int = -32000) -> McpError:
    """Build an McpError. -32000 is the default tool-timeout code per FastMCP."""
    return McpError(ErrorData(code=code, message=message))


@dataclass
class FakeChatResponse:
    """Duck-typed stand-in for ``tensorzero.types.ChatInferenceResponse``.

    Only carries the three attributes ``run_agent`` reads (``content``,
    ``episode_id``, ``inference_id``); everything else on the real dataclass
    (``usage``, ``raw_usage``, etc.) is unused.
    """

    content: list[Any]
    episode_id: UUID = field(default_factory=uuid4)
    inference_id: UUID = field(default_factory=uuid4)


class FakeT0:
    """Records every ``inference()`` call and returns canned responses in order."""

    def __init__(self, responses: list[FakeChatResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> FakeChatResponse:
        # Snapshot mutable args at call time. Real T0 serializes the request
        # immediately; without snapshotting, ``run_agent`` mutating ``messages``
        # later in the loop would retroactively change what tests see in
        # ``self.calls[N]``.
        captured: dict[str, Any] = dict(kwargs)
        if isinstance(captured.get("input"), dict):
            captured["input"] = dict(captured["input"])
            if isinstance(captured["input"].get("messages"), list):
                captured["input"]["messages"] = [
                    dict(m) if isinstance(m, dict) else m
                    for m in captured["input"]["messages"]
                ]
        self.calls.append(captured)
        if not self._queue:
            raise AssertionError("FakeT0 ran out of canned responses")
        return self._queue.pop(0)


class FakeMCPClient:
    """Stand-in for ``fastmcp.Client``.

    ``call_results`` maps tool name → either a ``CallToolResult`` (returned) or
    an ``Exception`` (raised). For tests that need to script per-call ordering
    independently, use ``call_outcomes`` (a list popped in order).
    """

    def __init__(
        self,
        *,
        tools: list[MCPTool] | None = None,
        call_results: dict[str, CallToolResult | Exception] | None = None,
        call_outcomes: list[CallToolResult | Exception] | None = None,
        side_effect: Any = None,
    ) -> None:
        self._tools = tools or [_mcp_tool("add")]
        self._call_results = call_results or {}
        self._call_outcomes = list(call_outcomes) if call_outcomes else None
        self._side_effect = side_effect
        self.call_tool_invocations: list[
            tuple[str | None, dict[str, Any] | None, float | None]
        ] = []

    async def list_tools(self) -> list[MCPTool]:
        return self._tools

    async def call_tool(
        self,
        name: str | None,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **_: Any,
    ) -> CallToolResult:
        self.call_tool_invocations.append((name, arguments, timeout))
        if self._side_effect is not None:
            await self._side_effect(name)
        if self._call_outcomes is not None:
            outcome: CallToolResult | Exception = self._call_outcomes.pop(0)
        else:
            key = name or ""
            if key not in self._call_results:
                raise AssertionError(f"FakeMCPClient: no canned result for {name!r}")
            outcome = self._call_results[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --- TestHasToolCalls -------------------------------------------------------


class TestHasToolCalls:
    def test_returns_true_with_toolcall(self) -> None:
        assert has_tool_calls([_tool_call()]) is True

    def test_returns_false_text_only(self) -> None:
        assert has_tool_calls([Text(text="just text")]) is False

    def test_ignores_non_toolcall_blocks(self) -> None:
        """Thought + Text alone should not trigger the loop continuation path."""
        assert (
            has_tool_calls([Thought(text="thinking..."), Text(text="hi")]) is False
        )

    def test_returns_true_with_mixed_blocks(self) -> None:
        """Thought + ToolCall: any ToolCall present is enough."""
        assert (
            has_tool_calls([Thought(text="..."), _tool_call(), Text(text="ok")])
            is True
        )

    def test_returns_false_for_empty_list(self) -> None:
        assert has_tool_calls([]) is False


# --- TestExecuteToolCalls ---------------------------------------------------


class TestExecuteToolCalls:
    """Sequential per M3.1, including the full error contract."""

    async def test_happy_path_one_call(self) -> None:
        client = FakeMCPClient(
            call_results={"add": _make_result(data=5)},
        )
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call(call_id="c1")],
        )
        assert len(results) == 1
        assert results[0].name == "add"
        assert results[0].id == "c1"
        assert results[0].result == "5"  # serialize_tool_result(5) → json.dumps → "5"

    async def test_multiple_calls_preserve_input_order(self) -> None:
        client = FakeMCPClient(
            call_results={
                "add": _make_result(data=5),
                "greet": _make_result(data="hi"),
            },
        )
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [
                _tool_call(name="add", raw_name="add", call_id="c1"),
                _tool_call(name="greet", raw_name="greet", call_id="c2"),
            ],
        )
        assert [r.name for r in results] == ["add", "greet"]
        assert [r.id for r in results] == ["c1", "c2"]

    async def test_arguments_none_returns_error_result(self) -> None:
        """EC-2: arguments=None means LLM produced invalid JSON. We never call
        the tool and return an error ToolResult instead."""
        client = FakeMCPClient(call_results={})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call(null_args=True, raw_arguments="{not valid json")],
        )
        assert len(results) == 1
        assert results[0].result.startswith("ERROR: invalid JSON arguments")
        assert "{not valid json" in results[0].result
        # Verify call_tool was NOT invoked when arguments is None.
        assert client.call_tool_invocations == []

    async def test_tool_error_caught(self) -> None:
        """F5.1 (anticipated): ToolError → error ToolResult, loop continues."""
        client = FakeMCPClient(call_results={"add": ToolError("division by zero")})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call()],
        )
        assert len(results) == 1
        assert results[0].result.startswith("ERROR: tool execution failed:")
        assert "division by zero" in results[0].result

    async def test_mcp_error_caught(self) -> None:
        """F5.2 / F5.4 (anticipated): McpError (incl. timeout -32000) → error result."""
        client = FakeMCPClient(call_results={"add": _mcp_error("timed out")})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call()],
        )
        assert results[0].result.startswith("ERROR: MCP protocol error")
        assert "timed out" in results[0].result

    async def test_unexpected_exception_caught(self) -> None:
        """Belt-and-braces: any other Exception → error ToolResult, loop continues."""
        client = FakeMCPClient(call_results={"add": RuntimeError("kaboom")})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call()],
        )
        assert results[0].result.startswith("ERROR: unexpected RuntimeError:")
        assert "kaboom" in results[0].result

    async def test_skips_non_toolcall_blocks(self) -> None:
        """Text and Thought blocks are silently skipped."""
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [Text(text="prefix"), Thought(text="..."), _tool_call()],
        )
        assert len(results) == 1
        assert results[0].name == "add"

    async def test_name_falls_back_to_raw_name_when_name_is_none(self) -> None:
        """When LLM hallucinates a name (validated name=None), the result still
        carries a string in the name field (raw_name)."""
        client = FakeMCPClient(call_results={})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call(name=None, raw_name="hallucinated", null_args=True)],
        )
        assert results[0].name == "hallucinated"

    async def test_name_none_with_valid_args_does_not_call_tool(self) -> None:
        """Hallucinated name + valid args: still an EC-2-like failure. We must
        never reach call_tool with a None name (would TypeError on the FastMCP
        side); the error is reported back to the LLM via ToolResult.result."""
        client = FakeMCPClient(call_results={})
        results = await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call(name=None, raw_name="bogus_tool", arguments={"x": 1})],
        )
        assert len(results) == 1
        assert results[0].name == "bogus_tool"
        assert "bogus_tool" in results[0].result
        assert "not in the schema" in results[0].result
        assert client.call_tool_invocations == []

    async def test_timeout_is_forwarded_to_call_tool(self) -> None:
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        await execute_tool_calls(
            cast(Client[Any], client),
            [_tool_call()],
            timeout=12.5,
        )
        assert client.call_tool_invocations[0][2] == 12.5


# --- TestExecuteToolCallsParallel -------------------------------------------


class TestExecuteToolCallsParallel:
    """M3.7: same contract as sequential, but dispatched via asyncio.gather."""

    async def test_runs_concurrently_preserves_input_order(self) -> None:
        """Even when later tools complete first, the result list matches input order."""
        completion_order: list[str] = []

        async def slow_then_fast(name: str | None) -> None:
            # The first call sleeps longer than the second. If we used a
            # sequential await, the second call would still complete second.
            # With gather, the second completes first — but the result list
            # must still be ordered by INPUT order. completion_order tracks
            # actual finish order to assert concurrency happened.
            assert name is not None
            delay = 0.05 if name == "add" else 0.01
            await asyncio.sleep(delay)
            completion_order.append(name)

        client = FakeMCPClient(
            call_results={
                "add": _make_result(data="add-result"),
                "greet": _make_result(data="greet-result"),
            },
            side_effect=slow_then_fast,
        )
        results = await execute_tool_calls_parallel(
            cast(Client[Any], client),
            [
                _tool_call(name="add", raw_name="add", call_id="c1"),
                _tool_call(name="greet", raw_name="greet", call_id="c2"),
            ],
        )
        assert [r.name for r in results] == ["add", "greet"]  # input order
        # Concurrency check: greet finished BEFORE add despite being scheduled
        # second. If execution were serial, the order would be ["add","greet"].
        assert completion_order == ["greet", "add"]

    async def test_empty_returns_empty(self) -> None:
        """No ToolCall blocks → no asyncio.gather call, returns []."""
        client = FakeMCPClient(call_results={})
        assert await execute_tool_calls_parallel(
            cast(Client[Any], client),
            [Text(text="just text")],
        ) == []

    async def test_errors_isolated_per_call(self) -> None:
        """One failing call must not poison the others."""
        client = FakeMCPClient(
            call_results={
                "add": _make_result(data=5),
                "greet": ToolError("greet failed"),
            },
        )
        results = await execute_tool_calls_parallel(
            cast(Client[Any], client),
            [
                _tool_call(name="add", raw_name="add", call_id="c1"),
                _tool_call(name="greet", raw_name="greet", call_id="c2"),
            ],
        )
        assert results[0].result == "5"
        assert results[1].result.startswith("ERROR: tool execution failed:")
        assert "greet failed" in results[1].result


# --- TestRunAgent -----------------------------------------------------------


class TestRunAgent:
    async def test_terminates_on_text_only_first_turn(self) -> None:
        """F3.4: response with only Text blocks → loop returns immediately."""
        episode = uuid4()
        inference = uuid4()
        t0 = FakeT0([
            FakeChatResponse(
                content=[Text(text="hello world")],
                episode_id=episode,
                inference_id=inference,
            ),
        ])
        client = FakeMCPClient()
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "Just say hi.",
            "agent",
        )
        assert result.response == "hello world"
        assert result.episode_id == str(episode)
        assert result.inference_ids == [str(inference)]
        assert result.turns == 1

    async def test_filters_thought_blocks_from_final_response(self) -> None:
        """Reasoning models emit Thought before Text; Thought must NOT leak
        into AgentResult.response (only Text.text values are joined)."""
        t0 = FakeT0([
            FakeChatResponse(content=[Thought(text="reasoning..."), Text(text="answer")]),
        ])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], FakeMCPClient()),
            "Q?",
            "agent",
        )
        assert result.response == "answer"
        assert "reasoning" not in result.response

    async def test_skips_text_blocks_with_none_text(self) -> None:
        """Text.text is Optional[str] (template form). None entries must not
        crash the join — they're filtered out."""
        t0 = FakeT0([
            FakeChatResponse(
                content=[Text(arguments={"foo": "bar"}), Text(text="real text")]
            ),
        ])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], FakeMCPClient()),
            "Q?",
            "agent",
        )
        assert result.response == "real text"

    async def test_single_turn_assembles_user_assistant_user_tools_history(self) -> None:
        """F3.6: after one tool round-trip, the inference history sent to T0
        must be exactly user → assistant(tool_call) → user(tool_results)."""
        episode = uuid4()
        tc = _tool_call(name="add", call_id="c1")
        t0 = FakeT0([
            FakeChatResponse(content=[tc], episode_id=episode),
            FakeChatResponse(content=[Text(text="the answer is 5")], episode_id=episode),
        ])
        client = FakeMCPClient(
            tools=[_mcp_tool("add")],
            call_results={"add": _make_result(data=5)},
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "compute 2+3",
            "agent",
        )
        assert result.response == "the answer is 5"
        assert result.turns == 2

        # Second inference's input.messages: walk and assert the structure.
        second_messages = t0.calls[1]["input"]["messages"]
        assert len(second_messages) == 3
        assert second_messages[0]["role"] == "user"
        assert second_messages[0]["content"] == "compute 2+3"
        assert second_messages[1]["role"] == "assistant"
        assert second_messages[1]["content"] == [tc]
        assert second_messages[2]["role"] == "user"
        # F3.6: exactly one ToolResult, in a user message.
        results_block = second_messages[2]["content"]
        assert len(results_block) == 1
        assert results_block[0].name == "add"
        assert results_block[0].id == "c1"

    async def test_two_tool_calls_collapse_into_one_user_message(self) -> None:
        """F3.5: N tool calls in one response → ONE Message(role="user")
        carrying N ToolResults. Not N separate user messages."""
        tc_a = _tool_call(name="add", raw_name="add", call_id="a", arguments={"a": 1, "b": 2})
        tc_b = _tool_call(name="greet", raw_name="greet", call_id="b", arguments={"name": "x"})
        t0 = FakeT0([
            FakeChatResponse(content=[tc_a, tc_b]),
            FakeChatResponse(content=[Text(text="done")]),
        ])
        client = FakeMCPClient(
            tools=[_mcp_tool("add"), _mcp_tool("greet")],
            call_results={
                "add": _make_result(data=3),
                "greet": _make_result(data="Hello, x!"),
            },
        )
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "do both",
            "agent",
        )
        second_messages = t0.calls[1]["input"]["messages"]
        # The user(tools) block is the LAST message; it must contain BOTH results.
        last = second_messages[-1]
        assert last["role"] == "user"
        assert len(last["content"]) == 2
        assert {r.name for r in last["content"]} == {"add", "greet"}
        assert {r.id for r in last["content"]} == {"a", "b"}

    async def test_episode_id_propagates_across_turns(self) -> None:
        """F3.3: episode_id from the first response is forwarded on every later
        turn (and the second response's episode_id flows through to subsequent
        calls too — they must all be tied to the same episode)."""
        ep = uuid4()
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call()], episode_id=ep),
            FakeChatResponse(content=[Text(text="ok")], episode_id=ep),
        ])
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        # First call: episode_id=None (caller didn't pass one in)
        assert t0.calls[0]["episode_id"] is None
        # Second call: episode_id is the str of the first response's episode_id
        assert t0.calls[1]["episode_id"] == str(ep)
        assert result.episode_id == str(ep)

    async def test_caller_supplied_episode_id_is_used_on_first_call(self) -> None:
        """F3.7 enabler: caller passes a previous episode_id to continue."""
        prev = str(uuid4())
        new_ep = uuid4()
        t0 = FakeT0([
            FakeChatResponse(content=[Text(text="hi again")], episode_id=new_ep),
        ])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], FakeMCPClient()),
            "follow up",
            "agent",
            episode_id=prev,
        )
        assert t0.calls[0]["episode_id"] == prev
        # Output episode_id reflects the response (in real T0, it'd match prev)
        assert result.episode_id == str(new_ep)

    async def test_inference_ids_collected_in_order(self) -> None:
        ids = [uuid4(), uuid4(), uuid4()]
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call(call_id="c1")], inference_id=ids[0]),
            FakeChatResponse(content=[_tool_call(call_id="c2")], inference_id=ids[1]),
            FakeChatResponse(content=[Text(text="end")], inference_id=ids[2]),
        ])
        client = FakeMCPClient(
            call_results={"add": _make_result(data=5)},
        )
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        assert result.inference_ids == [str(ids[0]), str(ids[1]), str(ids[2])]
        assert result.turns == 3

    async def test_max_turns_raises_max_turns_exceeded_error(self) -> None:
        """If the LLM keeps returning ToolCalls past max_turns, raise."""
        # Three responses, all tool calls. With max_turns=2, the loop should
        # raise after the 2nd inference call (since the 2nd response also has
        # a tool_call and we've used our budget).
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call(call_id="c1")]),
            FakeChatResponse(content=[_tool_call(call_id="c2")]),
        ])
        client = FakeMCPClient(call_results={"add": _make_result(data=5)})
        try:
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], client),
                "loop forever",
                "agent",
                max_turns=2,
            )
        except MaxTurnsExceededError as e:
            assert e.max_turns == 2
        else:
            raise AssertionError("expected MaxTurnsExceededError")

    async def test_parallel_tools_flag_uses_parallel_executor(self) -> None:
        """F3.8 unit: when parallel_tools=True, two tool calls execute via
        asyncio.gather — proven by reverse completion order matching the
        TestExecuteToolCallsParallel pattern."""
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
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "do both",
            "agent",
            parallel_tools=True,
        )
        # Second inference saw a single user message with two ordered results.
        last = t0.calls[1]["input"]["messages"][-1]
        assert last["role"] == "user"
        assert [r.name for r in last["content"]] == ["add", "greet"]
        # Concurrency check: greet finished first despite being scheduled second.
        assert completion_order == ["greet", "add"]

    async def test_dryrun_kwarg_forwards_to_inference(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], FakeMCPClient()),
            "hello",
            "agent",
            dryrun=True,
        )
        assert t0.calls[0]["dryrun"] is True

    async def test_tools_discovered_once_and_forwarded_each_turn(self) -> None:
        """Tool discovery happens at startup; the converted t0_tools dict is
        passed in EVERY inference call (not just the first)."""
        client = FakeMCPClient(
            tools=[_mcp_tool("add"), _mcp_tool("greet")],
            call_results={"add": _make_result(data=5)},
        )
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call()]),
            FakeChatResponse(content=[Text(text="done")]),
        ])
        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], client),
            "go",
            "agent",
        )
        for call in t0.calls:
            tools = call["additional_tools"]
            assert {t["name"] for t in tools} == {"add", "greet"}
