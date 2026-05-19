"""Pydantic-driven structured outputs via :class:`StructuredOutput`.

v1.0 ships :class:`StructuredOutput[M]` as a Pydantic-backed sibling of
:class:`JsonOutput`.  The model derives its own JSON schema (used by
TensorZero to validate the model output) and validates the parsed dict
into a typed instance — so ``result.value`` is statically typed as
``Customer`` (mypy-verified through the per-call overload on
:meth:`Agent.invoke`).

Prerequisites:
    1. ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. ``pip install 't0-agents[pydantic]'`` so :class:`StructuredOutput`
       can import :mod:`pydantic`.
    3. Start the gateway with this example's config:
         docker compose -f examples/03_output_modes/docker-compose.yml \\
             --env-file .env up -d
    4. Run:
         uv run python examples/03_output_modes/structured_pydantic.py

The TZ function ``extract_customer_info`` is the same one
``structured_agent.py`` uses — but here the schema is derived from the
:class:`Customer` Pydantic model at construction time, so updating the
model is the only place a schema change needs to land.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from pydantic import BaseModel, Field
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, StructuredOutput


class Customer(BaseModel):
    """The structured target the model is asked to extract."""

    name: str | None = Field(
        None, description="The customer's full name, or null if not present."
    )
    email: str | None = Field(
        None, description="The customer's email address, or null if not present."
    )


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def main() -> None:
    config = AgentConfig(
        name="customer-extractor",
        function_name="extract_customer_info",
        output=StructuredOutput(Customer),
    )

    messages = [
        "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
        "This is Diego Pérez (diego@example.com) — please follow up.",
    ]

    async with await _gateway_factory() as t0:
        agent = Agent(config, t0=t0)
        for idx, message in enumerate(messages, start=1):
            # Per-call overload narrows ``result.value`` to ``Customer``.
            result = await agent.invoke(
                message, output=StructuredOutput(Customer)
            ).result()
            print(f"--- run {idx} ---")
            print(f"value type:    {type(result.value).__name__}")
            print(f"value:         {result.value}")
            print(f"name:          {result.value.name if result.value else None}")
            print(f"email:         {result.value.email if result.value else None}")
            print(f"parse_strategy:{result.parse_strategy}")
            print(f"episode_id:    {result.episode_id}")


if __name__ == "__main__":
    asyncio.run(main())
