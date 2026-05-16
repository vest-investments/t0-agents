"""``ObservabilityMiddleware`` wires a tracer's span around the run."""

from __future__ import annotations

import contextlib
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.middleware import (
    CostTrackingMiddleware,
    JsonRetryMiddleware,
    ObservabilityMiddleware,
)
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


class _RecordingTracer:
    """Capture every ``start_as_current_span(name)`` call."""

    def __init__(self) -> None:
        self.spans: list[str] = []

    def start_as_current_span(self, name: str) -> Any:
        self.spans.append(name)
        return contextlib.nullcontext()


class TestObservabilityMiddleware:
    async def test_span_named_after_agent_invoke(self) -> None:
        tracer = _RecordingTracer()
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(),
                ),
            ],
        )
        agent = Agent(
            AgentConfig(
                name="traced",
                function_name="agent",
                middleware=[
                    ObservabilityMiddleware(tracer=tracer),
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("hi").result()

        assert tracer.spans == ["agent.invoke/traced"]

    async def test_no_tracer_is_noop(self) -> None:
        # Default constructor (tracer=None) falls back to
        # ``contextlib.nullcontext`` — the middleware is callable but
        # records nothing.  Useful when the user includes the
        # middleware in a default chain without configuring tracing.
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(),
                ),
            ],
        )
        agent = Agent(
            AgentConfig(
                name="no_tracer",
                function_name="agent",
                middleware=[
                    ObservabilityMiddleware(),  # tracer=None
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        # Should not raise.
        result = await agent.invoke("hi").result()
        assert result.response == "ok"

    async def test_span_wraps_inner_chain(self) -> None:
        # Observability is outermost in this chain, so the span name
        # is recorded BEFORE any other middleware runs.
        tracer = _RecordingTracer()
        run_order: list[str] = []

        class Marker:
            name = "marker"

            async def __call__(self, ctx: Any, next_: Any) -> Any:
                run_order.append("marker:before")
                # By now, tracer.spans should already have the span
                # name because ObservabilityMiddleware ran first.
                run_order.append(f"spans:{len(tracer.spans)}")
                result = await next_(ctx)
                run_order.append("marker:after")
                return result

        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(),
                ),
            ],
        )
        agent = Agent(
            AgentConfig(
                name="layered",
                function_name="agent",
                middleware=[
                    ObservabilityMiddleware(tracer=tracer),
                    Marker(),
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("hi").result()

        # Span recorded before Marker — Observability is outermost.
        assert run_order == ["marker:before", "spans:1", "marker:after"]
        assert tracer.spans == ["agent.invoke/layered"]
