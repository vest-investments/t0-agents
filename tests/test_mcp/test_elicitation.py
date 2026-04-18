"""Elicitation test suite.

Covers smoke validation of the in-memory transport with elicitation, pattern
coverage across binary approval / structured input / scalar wrapping /
multi-turn, error paths (missing handler, timeout), and reference-handler
unit tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError
from mcp import McpError
from tensorzero import ToolCall

from t0_agents.mcp import execute_tool_calls
from t0_agents.mcp.elicitation import auto_approve_handler, auto_decline_handler
from tests.server import mcp

# --- Smoke tests -----------------------------------------------------------


async def test_smoke_approve() -> None:
    """Handler accepts approval → tool returns 'Approved: deploy'."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("approve_action", {"action": "deploy"})
        assert result.data == "Approved: deploy"


async def test_smoke_structured_input() -> None:
    """Handler returns structured data → tool returns greeting."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={"name": "Alice", "age": 30})

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("collect_info", {})
        assert result.data == "Hello Alice, age 30"


async def test_smoke_decline() -> None:
    """Handler declines → tool returns rejection message."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    async with Client(mcp, elicitation_handler=handler) as client:
        result = await client.call_tool("approve_action", {"action": "deploy"})
        assert result.data == "Rejected: deploy"


async def test_smoke_no_handler() -> None:
    """No elicitation handler → ToolError raised from call_tool().

    The server-side McpError is wrapped by FastMCP into a
    CallToolResult(isError=True); the client then raises ToolError
    (not McpError) because raise_on_error defaults to True.
    """
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="Elicitation not supported"):
            await client.call_tool("approve_action", {"action": "deploy"})


# --- Elicitation pattern tests ---------------------------------------------


class TestElicitationPatterns:
    """Comprehensive coverage of elicitation patterns."""

    async def test_approval_accept(self, auto_approve_handler: Any) -> None:
        """Handler accepts, tool proceeds."""
        async with Client(mcp, elicitation_handler=auto_approve_handler) as client:
            result = await client.call_tool("approve_action", {"action": "deploy"})
            assert result.data == "Approved: deploy"

    async def test_approval_decline(self, auto_decline_handler: Any) -> None:
        """Handler declines, tool returns rejection."""
        async with Client(mcp, elicitation_handler=auto_decline_handler) as client:
            result = await client.call_tool("approve_action", {"action": "deploy"})
            assert result.data == "Rejected: deploy"

    async def test_approval_cancel(self) -> None:
        """Handler cancels, tool handles gracefully (same else branch)."""

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            return ElicitResult(action="cancel")

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("approve_action", {"action": "deploy"})
            assert result.data == "Rejected: deploy"

    async def test_scalar_str(self) -> None:
        """Scalar string flows through multi_step's first elicitation."""
        call_count = 0

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # "Step 1: name?"
                return ElicitResult(action="accept", content={"value": "Alice"})
            return ElicitResult(action="accept", content={})  # "Step 2: confirm?"

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("multi_step", {})
            assert result.data == "Completed: Alice"

    async def test_scalar_int(self) -> None:
        """Scalar int via ask_age tool."""

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            return ElicitResult(action="accept", content={"value": 42})

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("ask_age", {})
            assert result.data == "Age: 42"

    async def test_structured_input(self) -> None:
        """Structured dataclass input via collect_info."""

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            return ElicitResult(action="accept", content={"name": "Bob", "age": 25})

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("collect_info", {})
            assert result.data == "Hello Bob, age 25"

    async def test_constrained_choice(self) -> None:
        """Constrained choice via pick_priority."""

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            return ElicitResult(action="accept", content={"value": "medium"})

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("pick_priority", {})
            assert result.data == "Priority: medium"

    async def test_multi_turn_accept(self) -> None:
        """Both elicitations accepted in multi_step."""
        call_count = 0

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # "Step 1: name?"
                return ElicitResult(action="accept", content={"value": "Alice"})
            return ElicitResult(action="accept", content={})  # "Step 2: confirm?"

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("multi_step", {})
            assert result.data == "Completed: Alice"
            assert call_count == 2

    async def test_multi_turn_cancel(self) -> None:
        """First elicitation accepted, second declined."""
        call_count = 0

        async def handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # "Step 1: name?"
                return ElicitResult(action="accept", content={"value": "Alice"})
            return ElicitResult(action="decline")  # "Step 2: confirm?" declined

        async with Client(mcp, elicitation_handler=handler) as client:
            result = await client.call_tool("multi_step", {})
            assert result.data == "Rejected at step 2"


# --- Error path tests ------------------------------------------------------


class TestElicitationErrors:
    """Error handling for elicitation edge cases."""

    async def test_no_handler_raises_tool_error(self) -> None:
        """No handler on client → raises ToolError.

        FastMCP's tool-dispatch layer wraps server-side McpError into
        ``CallToolResult(isError=True)``; ``client.call_tool(raise_on_error=True)``
        then surfaces it as ``ToolError``. Documented in docs/error-handling.md.
        """
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="Elicitation not supported"):
                await client.call_tool("approve_action", {"action": "deploy"})

    async def test_no_handler_through_execute(self) -> None:
        """No handler through execute_tool_calls → error ToolResult.

        The ToolError is caught by ``_execute_single_tool_call``'s
        ``except ToolError`` branch and turned into a ``ToolResult`` whose
        ``result`` field starts with ``"ERROR: tool execution failed: "``.
        """
        tc = ToolCall(
            id="c1",
            raw_arguments='{"action": "deploy"}',
            raw_name="approve_action",
            arguments={"action": "deploy"},
            name="approve_action",
        )
        async with Client(mcp) as client:
            results = await execute_tool_calls(client, [tc], timeout=30.0)
            assert len(results) == 1
            assert results[0].name == "approve_action"
            assert results[0].result.startswith("ERROR: tool execution failed:")
            assert "Elicitation not supported" in results[0].result

    async def test_timeout_covers_elicitation(self) -> None:
        """Timeout wraps the elicitation roundtrip.

        A slow handler (simulating human wait time) causes the
        ``call_tool`` timeout to fire, proving that the timeout covers the
        full elicitation exchange — not just the tool execution.
        """

        async def slow_handler(
            message: str,
            response_type: type[Any] | None,
            params: Any,
            context: Any,
        ) -> ElicitResult[Any]:
            await asyncio.sleep(2.0)
            return ElicitResult(action="accept", content={})

        async with Client(mcp, elicitation_handler=slow_handler) as client:
            with pytest.raises((McpError, TimeoutError)):
                await client.call_tool(
                    "approve_action", {"action": "deploy"}, timeout=0.1
                )


# --- Reference handler tests -----------------------------------------------


class TestReferenceHandlers:
    """Unit tests for library reference handlers in t0_agents.mcp.elicitation."""

    async def test_auto_approve_accepts_approval(self) -> None:
        """auto_approve_handler accepts binary approval (response_type=None)."""
        result = await auto_approve_handler("Approve?", None, None, None)
        assert result.action == "accept"
        assert result.content == {}

    async def test_auto_approve_declines_structured(self) -> None:
        """auto_approve_handler declines when structured input is requested."""
        result = await auto_approve_handler("Details?", str, None, None)
        assert result.action == "decline"

    async def test_auto_decline_declines_approval(self) -> None:
        """auto_decline_handler declines binary approval."""
        result = await auto_decline_handler("Approve?", None, None, None)
        assert result.action == "decline"

    async def test_auto_decline_declines_structured(self) -> None:
        """auto_decline_handler declines structured input."""
        result = await auto_decline_handler("Details?", str, None, None)
        assert result.action == "decline"

    async def test_auto_approve_works_with_client(self) -> None:
        """auto_approve_handler works when passed to Client for binary approval."""
        async with Client(mcp, elicitation_handler=auto_approve_handler) as client:
            result = await client.call_tool("approve_action", {"action": "deploy"})
            assert result.data == "Approved: deploy"

    async def test_auto_decline_works_with_client(self) -> None:
        """auto_decline_handler works when passed to Client."""
        async with Client(mcp, elicitation_handler=auto_decline_handler) as client:
            result = await client.call_tool("approve_action", {"action": "deploy"})
            assert result.data == "Rejected: deploy"
