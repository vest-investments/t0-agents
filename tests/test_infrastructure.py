"""Phase 0 infrastructure tests.

Validates checklist items P0.2 (T0 inference) and P0.3-P0.5 (FastMCP basics).
P0.1 is verified manually via `docker compose up` + `curl /health`.
P0.6 is verified by pyproject.toml configuration.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text
from tensorzero.types import ChatInferenceResponse


# --- P0.3 ---
async def test_fastmcp_server_has_at_least_two_tools(mcp_client: Client[Any]) -> None:
    """P0.3: Test FastMCP server is instantiated with >= 2 @mcp.tool tools."""
    async with mcp_client:
        tools = await mcp_client.list_tools()
        assert len(tools) >= 2


# --- P0.4 ---
async def test_in_memory_client_lists_tools(mcp_client: Client[Any]) -> None:
    """P0.4: Client(mcp_instance) connects in-memory and list_tools returns tools."""
    async with mcp_client:
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert "add" in names
        assert "greet" in names


# --- P0.5 ---
async def test_call_tool_returns_call_tool_result_with_data(mcp_client: Client[Any]) -> None:
    """P0.5: call_tool() executes and returns CallToolResult with .data."""
    async with mcp_client:
        result = await mcp_client.call_tool("add", {"a": 5, "b": 3})
        assert result.is_error is False
        assert result.data == 8


# --- P0.2 ---
@pytest.mark.integration
async def test_t0_basic_inference_no_tools() -> None:
    """P0.2: Basic inference (no tools) succeeds against T0 via the native SDK."""
    # build_http returns Union[Awaitable[Gateway], Gateway]; with async_setup=True (default)
    # it's the Awaitable branch. Cast so mypy can narrow.
    gateway_factory = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    async with await gateway_factory as t0:
        response = await t0.inference(
            function_name="agent",
            input={"messages": [{"role": "user", "content": "Reply with the word: pong"}]},
            dryrun=True,
        )
        assert isinstance(response, ChatInferenceResponse)
        assert response.content
        # Reasoning models (e.g. gpt-oss on Groq) emit a Thought block before Text.
        # Honor the spirit of M0.4: at least one Text block with non-empty text.
        text_blocks = [b for b in response.content if isinstance(b, Text)]
        assert text_blocks, f"no Text blocks in response.content: {response.content!r}"
        assert any(b.text and b.text.strip() for b in text_blocks)
        assert response.inference_id is not None
        assert response.episode_id is not None
