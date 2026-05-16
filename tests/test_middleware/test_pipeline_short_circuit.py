"""Short-circuit + double-call + early-return semantics — Agent-integrated.

``test_chain_order.py`` covers the same surface at the
``build_pipeline`` level (no Agent involved).  This file exercises
the same invariants when the chain runs as part of a real
``Agent.invoke()``, ensuring the Run driver / terminal selection /
chain composition all cooperate cleanly.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import T0AgentsError
from t0_agents.core.middleware import (
    CostTrackingMiddleware,
    JsonRetryMiddleware,
)
from t0_agents.core.middleware.base import RunContext
from t0_agents.core.types import AgentResult, ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


class _ShortCircuit:
    """Middleware that returns a canned result without calling ``next_``."""

    name = "short_circuit"

    def __init__(self, response: str) -> None:
        self._response = response
        self.next_called = False

    async def __call__(self, ctx: RunContext, next_: Any) -> AgentResult:
        # Deliberately don't call next_ — the inner chain (and terminal)
        # never runs.  Synthesise a result directly.
        return AgentResult(
            response=self._response,
            episode_id="cached",
            inference_ids=[],
            turns=0,
        )


class _DoubleCaller:
    """Middleware that incorrectly calls ``next_`` twice."""

    name = "double_caller"

    async def __call__(self, ctx: RunContext, next_: Any) -> AgentResult:
        await next_(ctx)
        return await next_(ctx)


class TestShortCircuit:
    async def test_short_circuit_returns_synthesised_result(self) -> None:
        sc = _ShortCircuit("from-cache")
        # When short-circuit is the OUTERMOST layer, the terminal
        # never sees the run.  FakeT0 with no queued responses would
        # raise if reached — the test passes only when the terminal
        # truly never runs.
        agent = Agent(
            AgentConfig(
                name="sc",
                function_name="agent",
                middleware=[sc, CostTrackingMiddleware(), JsonRetryMiddleware()],
            ),
            t0=cast(AsyncTensorZeroGateway, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        result = await agent.invoke("hi").result()

        assert result.response == "from-cache"
        assert result.episode_id == "cached"


class TestDoubleCall:
    async def test_double_next_call_raises_t0agents_error(self) -> None:
        # FakeT0 needs ONE response for the first ``next_(ctx)`` to
        # reach the terminal successfully.  The SECOND ``next_(ctx)``
        # must raise from ``_NextGuard`` before any further inference
        # is attempted.
        agent = Agent(
            AgentConfig(
                name="bad",
                function_name="agent",
                middleware=[
                    _DoubleCaller(),
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(
                AsyncTensorZeroGateway,
                FakeT0(
                    [
                        FakeChatResponse(
                            content=[Text(type="text", text="ok")],
                            usage=FakeUsage(),
                        ),
                    ],
                ),
            ),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        with pytest.raises(T0AgentsError, match="more than once"):
            await agent.invoke("hi").result()


class TestExceptionPropagation:
    """Exceptions raised in middleware propagate out of ``Run.result()``."""

    async def test_middleware_exception_surfaces_to_caller(self) -> None:
        class Raiser:
            name = "raiser"

            async def __call__(self, ctx: RunContext, next_: Any) -> AgentResult:
                raise ValueError("middleware-raise")

        agent = Agent(
            AgentConfig(
                name="ex",
                function_name="agent",
                middleware=[Raiser(), JsonRetryMiddleware()],
            ),
            t0=cast(AsyncTensorZeroGateway, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        with pytest.raises(ValueError, match="middleware-raise"):
            await agent.invoke("hi").result()
