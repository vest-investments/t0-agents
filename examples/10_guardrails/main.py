"""Guardrails example — Milestone M3 "Trusted Agent".

Demonstrates all four built-in guardrails composed on one ``Agent``:

- ``require_approval(tools=["send_email", "delete_task"])`` with a shared
  CLI handler that also handles the server-side ``ctx.elicit()`` of
  ``delete_task`` — one handler, two surfaces, unified UX.
- ``max_cost(per_episode=0.50)`` — cumulative USD ceiling.
- ``max_tool_calls(per_run=8)`` — safety valve.
- ``block_tools(["drop_database"])`` — hard deny-list.

Requirements:
    docker compose up -d          # boots TensorZero gateway + ClickHouse
    GROQ_API_KEY in .env
    uv run python examples/10_guardrails/main.py

Emitting feedback for guardrail outcomes (T0 Q5 resolution — manual path).
The framework does not auto-emit ``t0.feedback(...)`` because metrics must
be pre-declared in ``tensorzero.toml``.  To record approval rejections as a
learnable signal, add to your config:

    [metrics.action_approved]
    type = "boolean"
    level = "inference"
    optimize = "max"

then call ``await t0.feedback(metric_name="action_approved", ...)`` after
inspecting ``result.guardrail_log`` — see ``docs/guardrails.md``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.guardrails import (
    block_tools,
    max_cost,
    max_tool_calls,
    require_approval,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def cli_elicitation_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
    """Unified handler used both by the Client and by ``require_approval``.

    For binary approval (empty schema / ``response_type is None``) reads y/n
    from stdin.  For structured input, prompts each schema property.  The
    function signature matches ``fastmcp.client.elicitation.ElicitationHandler``
    exactly so no adapter is needed.
    """
    print(f"\n{'=' * 60}")
    print("  APPROVAL REQUEST")
    print(f"{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")

    if response_type is None:
        answer = await asyncio.to_thread(input, "  Approve? [y/n]: ")
        if answer.strip().lower() in ("y", "yes"):
            return ElicitResult(action="accept", content={})
        return ElicitResult(action="decline")

    schema = params.requestedSchema
    collected: dict[str, Any] = {}
    for field, spec in schema.get("properties", {}).items():
        raw = await asyncio.to_thread(input, f"  {field} ({spec.get('type')}): ")
        collected[field] = raw
    return ElicitResult(action="accept", content=collected)


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    # Wire the CLI handler both on the Client (server-side ctx.elicit()) and
    # on require_approval (client-side gate).  Same callable, two surfaces.
    client: Client[Any] = Client(
        mcp, elicitation_handler=cli_elicitation_handler,
    )
    config = AgentConfig(
        name="executive_assistant",
        function_name="agent",
        max_turns=10,
        tool_timeout=None,  # humans are slow
        guardrails=[
            require_approval(
                tools=["send_email", "delete_task"],
                handler=cli_elicitation_handler,
            ),
            max_cost(per_episode=0.50),
            max_tool_calls(per_run=8),
            block_tools(["drop_database"]),
        ],
    )

    async with await gateway as t0, client:
        agent = Agent(config, t0=t0, tool_provider=client)

        try:
            result = await agent.run(
                "Read my calendar. Then send a follow-up email to "
                "alice@example.com with subject 'design review recap' and a "
                "two-sentence summary. Do not attempt destructive database "
                "operations."
            )
        except GuardrailBlockedError as exc:
            print(f"\nRun aborted by guardrail: {exc}")
            return

    print(f"\nResponse: {result.response}")
    print(f"Turns: {result.turns}")
    print("\nGuardrail log:")
    for entry in result.guardrail_log:
        print(
            f"  {entry.hook:<11} {entry.kind:<12} "
            f"{entry.guardrail_name}  tool={entry.tool_name!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
