"""Advanced tier: the v1.0 Run handle.

``agent.invoke(...)`` always returns a :class:`Run`.  This script
demonstrates the four control-plane methods on that handle that the
streaming demo above ``run.sh`` does not exercise:

- ``Run.events()`` — async iterate typed agent events live.
- ``Run.cancel()`` — cooperatively stop a running invocation.
- ``Run.fork(extra_message)`` — extend the transcript with a follow-up.
- ``Run.replay()`` — re-run the same input (new episode_id by default).

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Boot the gateway (shared with the streaming demo)::

           docker compose --env-file .env \\
               -f examples/agent/07_streaming/docker-compose.yml up -d

    3. Run::

           uv run python examples/agent/07_streaming/advanced_run_handle.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents import (
    Agent,
    AgentConfig,
    RunCancelledError,
    TextDeltaEvent,
)


def _gateway() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def demo_events(agent: Agent) -> None:
    """Iterate typed events as the model generates the answer."""
    print("\n=== 1. Run.events(): token-by-token streaming ===\n")
    run = agent.invoke(
        "In two sentences, what is the difference between a turtle and a tortoise?"
    )
    async for event in run.events():
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
    print()  # trailing newline
    result = await run
    print(f"\n  -> turns={result.turns}, episode_id={result.episode_id}")


async def demo_cancel(agent: Agent) -> None:
    """Start a run, cancel it immediately, observe the state."""
    print("\n=== 2. Run.cancel(): cooperative stop ===\n")
    run = agent.invoke("Write me a long, slow-paced poem about a sunrise.")
    # Chat runs are parked on first-consumer signal — cancel before
    # we touch .events() or await the result.
    await run.cancel()
    print(f"  state after cancel: {run.state!r}")
    try:
        await run.result()
    except RunCancelledError as exc:
        print(f"  awaiting after cancel raises: {type(exc).__name__}")


async def demo_fork(agent: Agent) -> None:
    """Take a completed run, extend it with a follow-up question."""
    print("\n=== 3. Run.fork(extra_message): extend the transcript ===\n")
    original = agent.invoke("Name a famous river in South America.")
    first = await original
    print(f"  original answer: {first.response[:120]}")
    forked = await original.fork(
        "Name another river in the same continent."
    )
    second = await forked
    print(f"  follow-up:       {second.response[:120]}")
    print(
        f"  same episode_id?  {first.episode_id == second.episode_id} "
        f"(fork preserves correlation)"
    )


async def demo_replay(agent: Agent) -> None:
    """Re-run the same input in a fresh episode."""
    print("\n=== 4. Run.replay(): re-run the same prompt ===\n")
    original = agent.invoke("Give me a one-line motivational quote.")
    first = await original
    print(f"  original: {first.response}")
    replayed = await original.replay()
    second = await replayed
    print(f"  replayed: {second.response}")
    print(
        f"  same episode_id?  {first.episode_id == second.episode_id} "
        "(False by default — replay mints a fresh episode)"
    )


async def main() -> None:
    async with await _gateway() as t0:
        agent = Agent(
            AgentConfig(name="run-handle-demo", function_name="agent"),
            t0=t0,
        )
        await demo_events(agent)
        await demo_cancel(agent)
        await demo_fork(agent)
        await demo_replay(agent)

    print("\nSUCCESS: advanced_run_handle ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
