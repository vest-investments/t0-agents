"""Advanced tier: a custom Middleware (timing + cost trace).

Guardrails ARE middleware in v1.0 — the four built-ins on display in
``main.py`` (require_approval, block_tools, max_cost, max_tool_calls)
all run inside the same chain a user-authored ``Middleware`` plugs
into.  This script shows how to write your own.

Two custom middlewares:

- :class:`TimingMiddleware` — wraps ``await next_(ctx)`` to record the
  wall-clock duration of every invocation.
- :class:`CostTraceMiddleware` — chains an ``on_inference`` observer to
  log per-inference usage / cost as the loop runs.

Both implement the :class:`~t0_agents.Middleware` protocol verbatim:
a ``name: str`` attribute and ``async __call__(ctx, next_)``.

Replaces the default middleware chain (because the user supplied an
explicit list); see ``docs/middleware.md`` for how to compose alongside
``AgentConfig.default_middleware_for(...)`` when you need the built-ins
too.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway from a sibling example (06_guardrails has no
       compose of its own; 01_first_agent's chat function is what we
       want)::

           docker compose --env-file .env \\
               -f examples/agent/01_first_agent/docker-compose.yml up -d

    3. Run::

           uv run python examples/agent/06_guardrails/advanced_middleware.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import (
    Agent,
    AgentConfig,
    AgentResult,
    Middleware,
    RunContext,
)


class TimingMiddleware:
    """Record the wall-clock duration of the entire invocation."""

    name = "timing"

    def __init__(self) -> None:
        self.elapsed_seconds: float | None = None

    async def __call__(
        self,
        ctx: RunContext,
        next_: Any,
    ) -> AgentResult[Any]:
        started = time.perf_counter()
        try:
            result = await next_(ctx)
        finally:
            self.elapsed_seconds = time.perf_counter() - started
        return cast(AgentResult[Any], result)


class CostTraceMiddleware:
    """Log per-inference cost as the loop runs."""

    name = "cost_trace"

    def __init__(self) -> None:
        self.inferences: list[tuple[int, int, float | None]] = []

    async def __call__(
        self,
        ctx: RunContext,
        next_: Any,
    ) -> AgentResult[Any]:
        async def _observe(response: Any) -> None:
            usage = getattr(response, "usage", None)
            if usage is None:
                self.inferences.append((0, 0, None))
                return
            self.inferences.append(
                (
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                    getattr(usage, "cost", None),
                )
            )

        ctx.chain_on_inference(_observe)
        return cast(AgentResult[Any], await next_(ctx))


async def main() -> None:
    timing = TimingMiddleware()
    cost = CostTraceMiddleware()

    # Static type check: the two locals must satisfy the Middleware
    # protocol the agent expects.
    _: list[Middleware] = [timing, cost]

    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        agent = Agent(
            AgentConfig(
                name="middleware-demo",
                function_name="agent",
                middleware=[timing, cost],
            ),
            t0=t0,
        )
        result = await agent.invoke(
            "In one sentence, name the largest desert in the world."
        )

    print(f"\nResponse:        {result.response}")
    print(f"Turns:           {result.turns}")
    print(f"Episode ID:      {result.episode_id}")
    print(f"Elapsed:         {timing.elapsed_seconds:.3f}s")
    print(f"Inferences seen: {len(cost.inferences)}")
    for i, (inp, out, c) in enumerate(cost.inferences, start=1):
        cost_str = f"${c:.6f}" if c is not None else "n/a"
        print(f"  #{i}: input_tokens={inp:4d}, output_tokens={out:4d}, cost={cost_str}")

    print("\nSUCCESS: advanced_middleware ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
