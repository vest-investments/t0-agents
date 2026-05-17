"""Glow structured output: return-type drives the schema.

Demonstrates that a Pydantic return annotation auto-selects
``StructuredOutput`` — the same agent decorator works for both chat
and structured shapes.  The user reads the trip back as a
``TripPlan`` instance, fully typed, with no parsing boilerplate.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/02_structured.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App


class TripPlan(BaseModel):
    destinations: list[str]
    cost_usd: float


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("planner", function_name="plan_trip")
        async def plan(query: str) -> TripPlan:  # type: ignore[empty-body]
            """Plan trips with realistic budgets."""

        trip = await plan("3 days in Japan under $2000")
        print(f"\nDestinations: {trip.destinations}")
        print(f"Estimated cost (USD): {trip.cost_usd}")

    print("\nSUCCESS: structured-output ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
