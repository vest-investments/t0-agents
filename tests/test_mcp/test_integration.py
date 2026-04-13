"""Integration tests — require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, ToolCall
from tensorzero.types import ChatInferenceResponse

from t0_agents.mcp import convert_tools
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
