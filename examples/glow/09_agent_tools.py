"""Glow per-agent ``tools=``: restrict which tools an agent sees.

Two agents on one ``App`` declare different ``tools=`` allow-lists and end up
with disjoint tool surfaces, while a third agent (no ``tools=``) keeps the
back-compatible "sees everything" behaviour.  Entries can be ``@app.tool``
callables, ``MCPServer`` classes/instances (matched by namespace prefix), and
glob strings.

The motivating case: ``place_order`` is mounted on the App but kept off the
``analyst``/``trader`` agents via ``tools=`` — a dangerous tool that stays
callable by nobody who wasn't explicitly granted it.

This example does not invoke any agent (no TZ inference happens), so it runs
in well under a second — it builds each agent and inspects the tool provider
the agent loop would see.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project
       root) — required by the docker-compose stack but not by this
       script itself.
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/09_agent_tools.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, llm


async def _visible_tools(wrapper: Any) -> set[str]:
    """The tool names the built agent's loop would advertise to inference."""
    provider = wrapper.agent._tool_provider
    return {tool.name for tool in await provider.list_tools()}


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.tool
        async def get_weather(city: str) -> str:
            """Current weather for a city."""
            return f"sunny in {city}"

        @app.tool
        async def get_quote(symbol: str) -> str:
            """Latest price for a ticker."""
            return f"{symbol}: 100.00"

        @app.tool
        async def place_order(symbol: str, qty: int) -> str:
            """DANGER: places a real order."""
            return f"ordered {qty} {symbol}"

        # Explicit callables — sees exactly these two; never place_order.
        @app.agent("analyst", tools=[get_weather, get_quote])
        async def analyst(q: str) -> str:
            """A read-only market analyst."""
            return llm()

        # Glob — every get_* tool, still no place_order.
        @app.agent("trader", tools=["get_*"])
        async def trader(q: str) -> str:
            """A trader that can read but (here) not place orders."""
            return llm()

        # No tools= — the back-compatible "sees everything" agent.
        @app.agent("generalist")
        async def generalist(q: str) -> str:
            """Sees every composed tool."""
            return llm()

        async with app.lifespan():
            analyst_tools = await _visible_tools(analyst)
            trader_tools = await _visible_tools(trader)
            generalist_tools = await _visible_tools(generalist)

            print(f"analyst    sees: {sorted(analyst_tools)}")
            print(f"trader     sees: {sorted(trader_tools)}")
            print(f"generalist sees: {sorted(generalist_tools)}")

            assert analyst_tools == {"get_weather", "get_quote"}
            assert trader_tools == {"get_weather", "get_quote"}
            assert generalist_tools == {"get_weather", "get_quote", "place_order"}
            assert "place_order" not in analyst_tools, (
                "order-placement tool leaked to the analyst agent"
            )

    print("\nSUCCESS: per-agent tools= filtering ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
