"""Unit tests for ``block_tools``.

Covers: soft-block short-circuit via ``before_tool``, single ``"ERROR: "``
prefix in the serialised result, and the documented asymmetry that
``list_tools`` is NOT filtered.
"""

from __future__ import annotations

from typing import Any, cast

from mcp.types import Tool as MCPTool

from t0_agents.core.types import Guardrail, GuardrailLogEntry, ToolProvider
from t0_agents.guardrails import block_tools
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.runtime import GuardedToolProvider
from t0_agents.mcp.adapter import serialize_tool_result
from tests._fake_mcp import FakeMCPClient


def _ctx() -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message="hi",
    )


def _wrap(
    inner: ToolProvider,
    guardrail: Guardrail,
) -> tuple[GuardedToolProvider, list[GuardrailLogEntry]]:
    ctx = _ctx()
    log: list[GuardrailLogEntry] = []
    return GuardedToolProvider(inner, [guardrail], ctx, log), log


class TestBlockTools:
    async def test_blocked_tool_short_circuits_inner_call(self) -> None:
        client = FakeMCPClient(call_results={})
        wrapped, log = _wrap(
            cast(ToolProvider, client),
            block_tools(["danger"]),
        )

        result = await wrapped.call_tool("danger", {})

        # Inner call_tool was NOT invoked.
        assert client.call_tool_invocations == []
        # The synthetic CallToolResult is an error.
        assert result.is_error is True
        # Exactly one "ERROR: " prefix after serialisation.
        serialized = serialize_tool_result(result)
        assert serialized.startswith("ERROR: ")
        assert serialized.count("ERROR: ") == 1
        # Log recorded the rejection.
        rejections = [e for e in log if e.kind == "rejection"]
        assert len(rejections) == 1
        assert rejections[0].tool_name == "danger"

    async def test_non_blocked_tool_passes_through(self) -> None:
        from fastmcp.client.client import CallToolResult
        from mcp.types import TextContent

        call = CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content=None,
            meta=None,
            data="ok",
            is_error=False,
        )
        client = FakeMCPClient(call_results={"safe": call})
        wrapped, log = _wrap(
            cast(ToolProvider, client),
            block_tools(["danger"]),
        )

        result = await wrapped.call_tool("safe", {"x": 1})

        assert client.call_tool_invocations == [("safe", {"x": 1}, None)]
        assert result is call
        # The pass path records a "pass" log entry.
        assert any(e.kind == "pass" for e in log)

    async def test_list_tools_not_filtered(self) -> None:
        """Documented asymmetry: block_tools is call-time only."""
        tools = [
            MCPTool(
                name="danger",
                description="x",
                inputSchema={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="safe",
                description="y",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        client = FakeMCPClient(tools=tools)
        wrapped, _ = _wrap(
            cast(ToolProvider, client),
            block_tools(["danger"]),
        )

        advertised = await wrapped.list_tools()
        names = {cast(Any, t).name for t in advertised}
        assert names == {"danger", "safe"}
