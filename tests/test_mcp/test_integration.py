"""Integration tests — require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from tensorzero.types import ChatInferenceResponse

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
