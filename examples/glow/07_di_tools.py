"""Glow Depends + @app.tool: dependency injection for MCP tools.

Demonstrates the FastAPI-style ``Depends(...)`` pattern lifted into
glow's ``@app.tool`` decorator.  ``Depends`` parameters are stripped
from the auto-generated MCP JSON Schema (so the LLM never sees them)
and re-injected at call time via the App's
:class:`~t0_agents.glow.di.DependencyResolver`.

``app.dependency_overrides[provider] = mock`` swaps the resolved
value at call time — no re-instantiation of the decorator needed.

This example does not invoke any agent (no TZ inference happens),
so it executes in well under a second.  The MCP round-trip exercises
the full registration + stripping + injection + override flow.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project
       root) — required by the docker-compose stack but not by this
       script itself.
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/07_di_tools.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from t0_agents.glow import App, Depends


@dataclass
class InMemoryDB:
    """Minimal in-memory store used to demonstrate the override pattern."""

    orders: dict[str, str] = field(default_factory=dict)


async def main() -> None:
    app = App(gateway="http://localhost:3000")

    def get_db() -> InMemoryDB:
        """Real production provider — returns a populated DB."""
        return InMemoryDB(orders={"o1": "Order #1 — A book", "o2": "Order #2 — A pen"})

    @app.tool
    async def lookup_order(
        order_id: str,
        db: Any = Depends(get_db),
    ) -> str:
        """Look up an order by ID."""
        return db.orders.get(order_id, f"<no order: {order_id}>")

    async with app:
        # `app.tools()` is the public inventory; `app._owned_client.call_tool`
        # (glow-internal) exercises the round-trip in isolation below to
        # highlight the Depends injection / override mechanic — in real
        # deployments the LLM calls the tool inside the agent loop.
        tools = await app.tools()
        advertised_params = sorted(
            tools[0].inputSchema.get("properties", {}).keys()
        )
        print(f"Advertised tool params: {advertised_params}")
        assert advertised_params == ["order_id"], (
            "`db` Depends param leaked into the JSON Schema."
        )

        # Real path.
        result = await app._owned_client.call_tool(
            "lookup_order", {"order_id": "o1"}
        )
        real_text = result.content[0].text
        print(f"Real DB:  {real_text}")
        assert real_text == "Order #1 — A book"

        # Install an override and re-call — the tool sees the mock.
        app.dependency_overrides[get_db] = lambda: InMemoryDB(
            orders={"o1": "[MOCKED in test override]"},
        )
        result = await app._owned_client.call_tool(
            "lookup_order", {"order_id": "o1"}
        )
        mock_text = result.content[0].text
        print(f"Mock DB:  {mock_text}")
        assert mock_text == "[MOCKED in test override]"

    print("\nSUCCESS: di-tools ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
