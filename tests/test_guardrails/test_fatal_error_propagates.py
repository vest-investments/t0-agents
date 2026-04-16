"""Verifies the new ``T0AgentsFatalError`` branch in ``_execute_single_tool_call``.

A fatal marker raised from inside ``mcp_client.call_tool`` must NOT be
caught by the belt-and-braces ``except Exception`` clause and stringified
into a ToolResult — it must propagate to the caller so ``Agent.run()``
can surface the policy violation.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import ToolCall

from t0_agents.core.exceptions import GuardrailBlockedError, T0AgentsFatalError
from t0_agents.mcp.loop import _execute_single_tool_call
from tests._fake_mcp import FakeMCPClient


class TestFatalPropagation:
    async def test_t0_agents_fatal_error_propagates(self) -> None:
        client = FakeMCPClient(
            call_results={"x": T0AgentsFatalError("hard stop")},
        )
        block = ToolCall(
            id="c1",
            raw_arguments="{}",
            raw_name="x",
            name="x",
            arguments={},
        )
        with pytest.raises(T0AgentsFatalError, match="hard stop"):
            await _execute_single_tool_call(
                cast(Client[Any], client), block, timeout=30.0,
            )

    async def test_guardrail_blocked_error_propagates(self) -> None:
        """``GuardrailBlockedError`` is a ``T0AgentsFatalError`` subclass."""
        client = FakeMCPClient(
            call_results={"x": GuardrailBlockedError("cost hit")},
        )
        block = ToolCall(
            id="c1",
            raw_arguments="{}",
            raw_name="x",
            name="x",
            arguments={},
        )
        with pytest.raises(GuardrailBlockedError, match="cost hit"):
            await _execute_single_tool_call(
                cast(Client[Any], client), block, timeout=30.0,
            )

    async def test_unrelated_exception_still_stringified(self) -> None:
        """Non-fatal exceptions keep the existing catch-all behaviour."""
        client = FakeMCPClient(
            call_results={"x": RuntimeError("boom")},
        )
        block = ToolCall(
            id="c1",
            raw_arguments="{}",
            raw_name="x",
            name="x",
            arguments={},
        )
        result = await _execute_single_tool_call(
            cast(Client[Any], client), block, timeout=30.0,
        )
        assert result.result.startswith("ERROR: unexpected RuntimeError")
