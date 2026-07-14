"""Glow streaming: the text one-liner, then live event selectors.

Part 1 — :meth:`FluentRun.stream_text`: the UI fast path.  Yields the
assistant's text tokens as they arrive; reasoning models that route the
answer through thought summaries (and stream no text deltas) still
yield the final answer once, thanks to the built-in fallback.

Part 2 — :meth:`FluentRun.events`: filtered access to the underlying
typed-event stream.  Three selectors are in flight at once:

- ``Thought`` — yields :class:`ThoughtEvent` (reasoning blocks the
  model emits when running on a thinking-capable variant).
- ``ToolCall.map(lambda tc: f"> calling {tc.name}")`` — yields a
  formatted string when the agent dispatches a tool call.  This is
  the ``.map()`` selector pattern: the underlying event is
  transformed before reaching the consumer.
- ``Text.map(...)`` — ``Text`` selects :class:`TextDeltaEvent`, the
  events that carry the actual tokens (the block-end marker is
  reachable as ``TextEnd``).

Glow streams never raise mid-iteration: a fatal failure arrives as a
single terminal :class:`RunErrorEvent` instead (``on_error="event"`` is
the glow default).

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
from t0_agents.stream.events import ThoughtEvent

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
            print("=== stream_text: tokens as they arrive ===")
            async for token in forecaster(
                "Quick — what's the weather in Lisbon?"
            ).stream_text():
                print(token, end="", flush=True)
            print()

            print("\n=== live event stream ===")
            async for event in forecaster(
                "And in Tokyo?"
            ).events(
                Thought,
                ToolCall.map(lambda tc: f"> calling {tc.name}"),
                Text.map(lambda d: d.delta),
            ):
                if isinstance(event, ThoughtEvent):
                    print(f"[thought] {event.text}")
                else:  # both .map() selectors yielded strings
                    print(event, end="", flush=True)
            print()

    print("\nSUCCESS: streaming ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
