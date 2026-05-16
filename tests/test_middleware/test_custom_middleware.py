"""Custom middleware ergonomics — user-authored middlewares run correctly.

Validates that the extension surface works as documented: a third-party
middleware that mutates ``ctx.user_message``, adds an
``on_inference`` observer, or serialises concurrent invocations
behaves end-to-end through ``Agent.invoke``.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.middleware import (
    CostTrackingMiddleware,
    JsonRetryMiddleware,
    Middleware,
    RunContext,
)
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


class _PiiLowercaser:
    """User middleware: lowercase ``ctx.user_message`` before inference."""

    name = "pii_lowercaser"

    async def __call__(self, ctx: RunContext, next_: Any) -> Any:
        ctx.user_message = ctx.user_message.lower()
        return await next_(ctx)


class _InferenceCounter:
    """User middleware: count inferences via ``ctx.chain_on_inference``."""

    def __init__(self) -> None:
        self.name = "counter"
        self.count = 0

    async def __call__(self, ctx: RunContext, next_: Any) -> Any:
        async def _bump(_response: Any) -> None:
            self.count += 1

        ctx.chain_on_inference(_bump)
        return await next_(ctx)


class _SemaphoreSerialiser:
    """User middleware: serialise concurrent invocations via Semaphore."""

    def __init__(self, sem: asyncio.Semaphore, log: list[str]) -> None:
        self.name = "semaphore"
        self._sem = sem
        self._log = log

    async def __call__(self, ctx: RunContext, next_: Any) -> Any:
        async with self._sem:
            self._log.append(f"enter:{ctx.user_message}")
            try:
                return await next_(ctx)
            finally:
                self._log.append(f"exit:{ctx.user_message}")


class TestPiiLowercaser:
    async def test_inference_sees_lowercased_message(self) -> None:
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
                name="pii",
                function_name="agent",
                middleware=[
                    _PiiLowercaser(),
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("HELLO World").result()

        # FakeT0 captured the inference input; the user-turn message
        # in input.messages must be the lowercased form.
        sent_messages = t0.calls[0]["input"]["messages"]
        assert sent_messages[-1]["content"] == "hello world"


class TestChainOnInferenceFromUserMiddleware:
    async def test_observer_fires_for_each_inference(self) -> None:
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(),
                ),
            ],
        )
        counter = _InferenceCounter()
        agent = Agent(
            AgentConfig(
                name="obs",
                function_name="agent",
                middleware=[
                    counter,
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("hi").result()

        assert counter.count == 1

    async def test_user_observer_composes_with_cost_tracker(self) -> None:
        # Both the user's observer and the always-on cost tracker
        # should fire on the same inference.  Validates the
        # ``chain_on_inference`` compose semantics under realistic load.
        t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[Text(type="text", text="ok")],
                    usage=FakeUsage(
                        input_tokens=4, output_tokens=2, cost=0.0001,
                    ),
                ),
            ],
        )
        counter = _InferenceCounter()
        agent = Agent(
            AgentConfig(
                name="dual",
                function_name="agent",
                middleware=[
                    counter,
                    CostTrackingMiddleware(),
                    JsonRetryMiddleware(),
                ],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        run = agent.invoke("hi")
        await run.result()

        assert counter.count == 1
        # Cost tracker also saw the inference.
        assert run.cost() == 0.0001


class TestSemaphoreSerialiser:
    async def test_concurrent_runs_are_serialised(self) -> None:
        sem = asyncio.Semaphore(1)
        log: list[str] = []

        async def _build_run(msg: str) -> Any:
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
                    name=f"sem-{msg}",
                    function_name="agent",
                    middleware=[
                        _SemaphoreSerialiser(sem, log),
                        CostTrackingMiddleware(),
                        JsonRetryMiddleware(),
                    ],
                ),
                t0=cast(AsyncTensorZeroGateway, t0),
                tool_provider=cast(ToolProvider, FakeMCPClient()),
            )
            return await agent.invoke(msg).result()

        # Fire two runs concurrently; the semaphore (capacity=1)
        # should force them to interleave enter/exit cleanly.
        await asyncio.gather(_build_run("first"), _build_run("second"))

        # The two enter-exit pairs must NOT interleave (capacity=1).
        # Either first runs to completion before second starts, or
        # vice versa.
        assert log[0].startswith("enter:")
        assert log[1] == f"exit:{log[0].split(':', 1)[1]}"
        assert log[2].startswith("enter:")
        assert log[2] != log[0]  # different message
        assert log[3] == f"exit:{log[2].split(':', 1)[1]}"


class TestMiddlewareProtocolDuckType:
    """Any object with ``name: str`` + ``__call__`` satisfies the protocol."""

    def test_module_function_is_not_a_middleware(self) -> None:
        # A bare ``async def`` lacks ``.name`` so it doesn't satisfy
        # the Middleware Protocol.  Confirms ``runtime_checkable``
        # detects the missing attribute.
        async def f(ctx: Any, next_: Any) -> Any:
            return await next_(ctx)

        assert not isinstance(f, Middleware)

    def test_dataclass_like_object_works(self) -> None:
        # A class with ``name`` + async ``__call__`` is a valid middleware.
        class MW:
            name = "duck"

            async def __call__(self, ctx: Any, next_: Any) -> Any:
                return await next_(ctx)

        assert isinstance(MW(), Middleware)
