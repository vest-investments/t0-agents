"""Integration tests — require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse

from t0_agents import run_agent
from t0_agents.mcp import build_tool_result, convert_tools
from tests.server import mcp

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Helper to narrow `build_http`'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_t0_accepts_additional_tools(mcp_client: Client[Any]) -> None:
    """F1.3: T0 accepts the converted additional_tools without error."""
    async with mcp_client:
        tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "Say hi briefly."}]},
            additional_tools=t0_tools,
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    assert response.content is not None
    assert response.inference_id is not None
    assert response.episode_id is not None


async def test_tool_choice_forces_tool_call(mcp_client: Client[Any]) -> None:
    """F1.4 (deviation): a directive tool_choice yields a ToolCall end-to-end.

    The checklist literally says `tool_choice="required"`. Groq's native API
    rejects `"required"` for `gpt-oss-20b` with "'tool_choice' : value must
    be a string (auto or none) or an object like `{'type': 'function', ...}`",
    so we use `tool_choice={"specific": "add"}` — TensorZero maps this to the
    object form Groq always accepts. The spirit of F1.4 (prove the plumbing
    end-to-end: `additional_tools` reach the LLM, `tool_choice` is honored,
    response contains a `ToolCall` block) is preserved.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
    t0_tools = convert_tools(tools)

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={
                "messages": [
                    {"role": "user", "content": "Compute 2 plus 3."},
                ]
            },
            additional_tools=t0_tools,
            tool_choice={"specific": "add"},
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    tool_calls = [b for b in response.content if isinstance(b, ToolCall)]
    assert tool_calls, (
        f"tool_choice={{'specific':'add'}} produced no ToolCall. "
        f"content={response.content!r}"
    )
    assert tool_calls[0].name == "add"


async def test_pydantic_param_schema_accepted_by_t0() -> None:
    """F1.5: tool with a Pydantic-model parameter produces a schema T0 accepts."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    # Restrict to the Pydantic-param tool to narrow the failure mode: if T0
    # rejects the schema, the error is unambiguously about `search`.
    search_tool = [t for t in convert_tools(tools) if t["name"] == "search"]
    assert len(search_tool) == 1

    async with await _gateway_factory() as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "Find something interesting."}]},
            additional_tools=search_tool,
            dryrun=True,
        )

    assert isinstance(response, ChatInferenceResponse)
    assert response.content is not None


async def test_add_tool_roundtrip_final_text(mcp_client: Client[Any]) -> None:
    """F2.1 + F2.7 + F2.8: full two-turn roundtrip using ``add``.

    1. Inference #1 with ``tool_choice={"specific":"add"}`` forces a ToolCall.
    2. We execute it through FastMCP (``raise_on_error=False``, F2.1).
    3. ``build_tool_result`` produces a ``ToolResult`` (F2.7 shape).
    4. Inference #2 carries the conversation back; T0 must accept the
       ``ToolResult`` in a user message and the LLM must produce at least one
       ``Text`` block that mentions the computed value (F2.7 + F2.8).

    Episode id is reused between turns so T0 ties the two inferences together.
    ``dryrun=True`` keeps ClickHouse clean (same pattern as Phase 1).
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
        t0_tools = convert_tools(tools)

        async with await _gateway_factory() as t0:
            first_turn = await t0.inference(
                function_name="agent",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please compute 2 + 3 using the add tool.",
                        }
                    ]
                },
                additional_tools=t0_tools,
                tool_choice={"specific": "add"},
                dryrun=True,
            )

            assert isinstance(first_turn, ChatInferenceResponse)
            tool_calls = [b for b in first_turn.content if isinstance(b, ToolCall)]
            assert tool_calls, f"no ToolCall in first turn: {first_turn.content!r}"
            tc = tool_calls[0]
            assert tc.name == "add"
            assert tc.arguments is not None, (
                f"LLM produced invalid JSON arguments: {tc.raw_arguments!r}"
            )

            # F2.1 — execute the tool through FastMCP with the validated args
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error, (
                f"add tool unexpectedly errored: {mcp_result!r}"
            )

            # F2.7 — build the ToolResult T0 will consume
            t0_result = build_tool_result(tc, mcp_result)
            assert t0_result.name == "add"
            assert t0_result.id == tc.id
            assert "5" in t0_result.result, (
                f"serialized result should mention the sum 5: {t0_result.result!r}"
            )

            second_turn = await t0.inference(
                function_name="agent",
                episode_id=first_turn.episode_id,
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Please compute 2 + 3 using the add tool.",
                        },
                        {"role": "assistant", "content": first_turn.content},
                        {"role": "user", "content": [t0_result]},
                    ]
                },
                additional_tools=t0_tools,
                dryrun=True,
            )

        # F2.8 — LLM produces a coherent response that incorporates the result
        assert isinstance(second_turn, ChatInferenceResponse)
        text_blocks = [b for b in second_turn.content if isinstance(b, Text)]
        assert text_blocks, (
            "second turn produced no Text blocks — the LLM did not respond "
            f"after the tool result. content={second_turn.content!r}"
        )
        final_text = " ".join(b.text for b in text_blocks)
        assert "5" in final_text, (
            f"final answer does not mention the sum 5: {final_text!r}"
        )


async def test_search_pydantic_roundtrip(mcp_client: Client[Any]) -> None:
    """F2.1 + F2.3 + F2.7 with the Pydantic-parameter ``search`` tool.

    Focuses on F2.3 (non-trivial ``.data`` payload serialization — the
    ``search`` tool returns a list, which goes through ``serialize_tool_result``'s
    ``json.dumps(..., default=str)`` fallback branch) and F2.7 (T0 accepts the
    ``ToolResult`` in the next turn without raising). We deliberately do **not**
    pin F2.8 here because the ``search`` tool in ``tests/server.py`` is typed
    ``-> list[dict]`` (no params) and FastMCP's content-block serializer
    sometimes stringifies it as ``["Root()"]``, which the LLM reasonably
    reacts to by asking for another tool call rather than by emitting text.
    F2.8 is covered end-to-end by ``test_add_tool_roundtrip_final_text``.
    """
    async with mcp_client:
        tools = await mcp_client.list_tools()
        t0_tools = convert_tools(tools)

        async with await _gateway_factory() as t0:
            first_turn = await t0.inference(
                function_name="agent",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use the search tool to look up 'tensorzero' "
                                "with a limit of 3 results."
                            ),
                        }
                    ]
                },
                additional_tools=t0_tools,
                tool_choice={"specific": "search"},
                dryrun=True,
            )

            assert isinstance(first_turn, ChatInferenceResponse)
            tool_calls = [b for b in first_turn.content if isinstance(b, ToolCall)]
            assert tool_calls, f"no ToolCall in first turn: {first_turn.content!r}"
            tc = tool_calls[0]
            assert tc.name == "search"
            assert tc.arguments is not None

            # F2.1 — execute the Pydantic-parametered tool
            mcp_result = await mcp_client.call_tool(
                tc.name,
                tc.arguments,
                raise_on_error=False,
            )
            assert not mcp_result.is_error

            # F2.3 + F2.7 — the serialized result is a non-empty string; build
            # produces a well-formed ToolResult with matching id/name.
            t0_result = build_tool_result(tc, mcp_result)
            assert isinstance(t0_result.result, str) and t0_result.result
            assert t0_result.name == "search"
            assert t0_result.id == tc.id

            # F2.7 end-to-end: T0 accepts the ToolResult in a follow-up
            # inference without raising. The exact shape of the LLM's response
            # is not pinned here.
            second_turn = await t0.inference(
                function_name="agent",
                episode_id=first_turn.episode_id,
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Use the search tool to look up 'tensorzero' "
                                "with a limit of 3 results."
                            ),
                        },
                        {"role": "assistant", "content": first_turn.content},
                        {"role": "user", "content": [t0_result]},
                    ]
                },
                additional_tools=t0_tools,
                dryrun=True,
            )

        assert isinstance(second_turn, ChatInferenceResponse)
        assert second_turn.content, (
            "second turn returned empty content — T0 or the LLM rejected the "
            f"ToolResult. first_turn content: {first_turn.content!r}"
        )


# --- Phase 3: agent loop end-to-end ----------------------------------------


async def test_run_agent_single_turn_add(mcp_client: Client[Any]) -> None:
    """F3.1 + F3.2 + F3.3 + F3.4: run_agent end-to-end with the ``add`` tool.

    F3.1 (single-turn user→LLM→tool→result→final) is exercised here. Since the
    add tool requires one execution cycle, the loop naturally runs ``>= 2``
    turns (one to call ``add``, one to deliver the final answer), which also
    satisfies F3.2 (multi-turn). F3.3 (episode_id propagation) is anchored by
    asserting the result has a non-empty episode_id; F3.4 (loop terminates on
    text-only response) is implicit in the loop completing.

    ``dryrun=True`` keeps ClickHouse clean (Phase 1/2 convention).
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the add tool to compute 2 + 3.",
                "agent",
                dryrun=True,
            )

    assert "5" in result.response, (
        f"final answer should mention the sum 5: {result.response!r}"
    )
    assert result.episode_id, "episode_id was not captured from the response"
    assert result.inference_ids, "inference_ids should not be empty"
    assert result.turns >= 2, (
        f"expected at least 2 turns (tool + final), got {result.turns}"
    )
    # F3.3: each turn captures one inference_id, in order.
    assert len(result.inference_ids) == result.turns


async def test_run_agent_no_tools_terminates_immediately(
    mcp_client: Client[Any],
) -> None:
    """F3.4: when the LLM has no tool calls, the loop terminates after 1 turn.

    A pure greeting prompt should never trigger a tool call (none of our test
    tools — add, greet(name), search, echo_no_docs — match a bare "say hi"
    request). The loop should return on the first inference's text-only
    response.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: pong",
                "agent",
                dryrun=True,
            )

    assert result.turns == 1, f"expected exactly 1 turn, got {result.turns}"
    assert result.response, "expected non-empty text response"


async def test_run_agent_continuous_conversation(mcp_client: Client[Any]) -> None:
    """F3.7: after a final response the user can submit another query.

    Two ``run_agent`` calls back-to-back, the second carrying ``episode_id``
    from the first. T0 ties them as one episode (visible in ClickHouse as a
    shared episode_id even with dryrun=True the response.episode_id remains
    consistent). The LLM does not see the prior conversation's messages — that
    is a design choice (run_agent is single-conversation; episode_id is
    analytics-only at this layer) — but the API must still accept the
    second call without error.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            first = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: hello",
                "agent",
                dryrun=True,
            )
            second = await run_agent(
                t0,
                mcp_client,
                "Reply with the single word: world",
                "agent",
                episode_id=first.episode_id,
                dryrun=True,
            )

    assert first.episode_id, "first call should have an episode_id"
    assert second.episode_id == first.episode_id, (
        "T0 should keep the episode_id when we forward it in the second call: "
        f"first={first.episode_id!r}, second={second.episode_id!r}"
    )
    assert second.response, "second call must produce a response"


async def test_run_agent_parallel_tools_flag_works(mcp_client: Client[Any]) -> None:
    """F3.8: ``parallel_tools=True`` does not break the flow.

    Groq's ``gpt-oss-20b`` does not generally emit multiple tool calls in a
    single response (Groq is not on T0's parallel_tool_calls allowlist), so
    this test exercises the parallel path with a single-tool flow. A passing
    run proves that ``execute_tool_calls_parallel`` integrates correctly with
    ``run_agent`` even when there is only one tool to run — the optional
    parallel mode does not regress the sequential happy path.
    """
    async with mcp_client:
        async with await _gateway_factory() as t0:
            result = await run_agent(
                t0,
                mcp_client,
                "Please use the add tool to compute 4 + 6.",
                "agent",
                dryrun=True,
                parallel_tools=True,
            )

    assert "10" in result.response, (
        f"final answer should mention the sum 10: {result.response!r}"
    )
    assert result.turns >= 2
