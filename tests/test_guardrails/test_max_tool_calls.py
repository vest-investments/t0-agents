"""Unit tests for ``max_tool_calls``."""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.guardrails import max_tool_calls
from t0_agents.guardrails.base import GuardrailContext


def _ctx(count: int = 0) -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message="go",
        tool_calls_made=count,
    )


class TestMaxToolCalls:
    async def test_first_call_admitted(self) -> None:
        g = max_tool_calls(per_run=2)
        ctx = _ctx()
        assert await g.before_tool(ctx, type("TC", (), {"name": "x"})()) is None
        assert ctx.tool_calls_made == 1

    async def test_counter_incremented_on_before_tool(self) -> None:
        """Counter lives in ``before_tool`` (pre-dispatch) — race-safe under parallel."""
        g = max_tool_calls(per_run=3)
        ctx = _ctx()
        for _ in range(3):
            await g.before_tool(ctx, type("TC", (), {"name": "x"})())
        assert ctx.tool_calls_made == 3

    async def test_over_cap_raises(self) -> None:
        g = max_tool_calls(per_run=1)
        ctx = _ctx(count=1)
        with pytest.raises(GuardrailBlockedError, match="Tool-call cap"):
            await g.before_tool(ctx, type("TC", (), {"name": "x"})())

    async def test_zero_is_valid_no_calls_allowed(self) -> None:
        """``per_run=0`` is a valid 'read-only agent' configuration."""
        g = max_tool_calls(per_run=0)
        with pytest.raises(GuardrailBlockedError, match="Tool-call cap"):
            await g.before_tool(_ctx(), type("TC", (), {"name": "x"})())

    async def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            max_tool_calls(per_run=-5)
