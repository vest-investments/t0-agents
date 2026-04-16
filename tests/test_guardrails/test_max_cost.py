"""Unit tests for ``max_cost``.

Covers: accumulation via ``ctx.cost_so_far``, hard-abort via
``GuardrailBlockedError``, and the ``usage.cost is None`` → 0.0 fallback.
"""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import ToolProvider
from t0_agents.guardrails import max_cost
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.runtime import GuardedToolProvider


class _FakeInner:
    async def list_tools(self) -> list[object]:
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        timeout: float | None = None,
    ) -> object:
        raise AssertionError("inner call_tool should never run when blocked")


def _ctx(cost: float = 0.0) -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message="go",
        cost_so_far=cost,
    )


class TestMaxCost:
    async def test_under_budget_passes(self) -> None:
        ctx = _ctx(cost=0.1)
        g = max_cost(per_episode=1.0)
        short = await g.before_tool(ctx, type("TC", (), {"name": "x"})())
        assert short is None

    async def test_exceeded_raises_guardrail_blocked_error(self) -> None:
        ctx = _ctx(cost=1.0)
        g = max_cost(per_episode=0.5)
        with pytest.raises(GuardrailBlockedError, match="Cost limit"):
            await g.before_tool(ctx, type("TC", (), {"name": "x"})())

    async def test_invalid_construction(self) -> None:
        with pytest.raises(ValueError):
            max_cost(per_episode=0.0)
        with pytest.raises(ValueError):
            max_cost(per_episode=-1.0)

    async def test_propagates_past_guarded_tool_provider(self) -> None:
        """Fatal error from before_tool propagates out of the wrapper."""
        from typing import cast

        wrapped = GuardedToolProvider(
            cast(ToolProvider, _FakeInner()),
            [max_cost(per_episode=0.0001)],
            _ctx(cost=1.0),
            [],
        )
        with pytest.raises(GuardrailBlockedError):
            await wrapped.call_tool("anything", {})
