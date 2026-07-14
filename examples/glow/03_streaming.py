"""Glow streaming: live event selectors over an agent + tool loop.

Demonstrates :meth:`FluentRun.events`: filtered access to the
underlying typed-event stream as the agent runs.  Three selectors
are in flight at once:

- ``Thought`` — yields :class:`ThoughtEvent` (reasoning blocks the
  model emits when running on a thinking-capable variant).
- ``ToolCall.map(lambda tc: f"> calling {tc.name}")`` — yields a
  formatted string when the agent dispatches a tool call.  This is
  the ``.map()`` selector pattern: the underlying event is
  transformed before reaching the consumer.
- ``Text`` — yields :class:`TextEndEvent` markers (one per
  completed assistant text block).  The marker carries the block's
  ``id`` only; for token-by-token streaming, use
  ``Selector(TextDeltaEvent).map(lambda e: e.delta)`` instead.

The agent's loop reaches the tool through the App-owned in-memory
FastMCP server; the ``@app.tool`` decorator registers ``get_weather``
against that server so the LLM can call it without any explicit MCP
plumbing.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/03_streaming.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, Text, Thought, ToolCall, llm
from t0_agents.stream.events import TextEndEvent, ThoughtEvent

# Approximate "current weather" the agent can consult.  Keeping the
# data inline avoids a real HTTP call inside the example so the
# streaming output is the only thing the reader has to follow.
_WEATHER = {
    "Tokyo": "22°C, partly cloudy, light wind",
    "Lisbon": "18°C, sunny, breezy",
    "Reykjavik": "4°C, overcast, drizzle",
}


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.tool
        async def get_weather(city: str) -> str:
            """Return the current weather summary for the given city."""
            return _WEATHER.get(city, f"unknown city: {city}")

        @app.agent("forecaster")
        async def forecaster(query: str) -> str:
            """A weather assistant.  When the user asks about a city,
            call the `get_weather` tool and narrate the result in one
            short, friendly sentence."""
            return llm()

        async with app.lifespan():
            run = forecaster("Quick — what's the weather in Tokyo?")
            print("=== live event stream ===")
            async for event in run.events(
                Thought,
                ToolCall.map(lambda tc: f"> calling {tc.name}"),
                Text,
            ):
                if isinstance(event, ThoughtEvent):
                    print(f"[thought] {event.text}")
                elif isinstance(event, TextEndEvent):
                    print(f"[text block {event.id} complete]")
                else:  # ToolCall.map yielded a formatted string
                    print(event)

    print("\nSUCCESS: streaming ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
