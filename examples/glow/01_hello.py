"""Glow hello-world: the 4-line DX.

Demonstrates the headline ergonomic of ``t0_agents.glow`` — a chat
agent defined entirely by its function signature and docstring, with
no MCP plumbing, no ``AgentConfig`` boilerplate, and no ``Run``
juggling.  The docstring becomes the system prompt; the ``-> str``
return annotation drives ``ChatOutput`` inference; ``await tutor(msg)``
returns the unwrapped string directly.

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/01_hello.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, llm


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with await gateway as t0:
        app = App(t0=t0)

        @app.agent("tutor")
        async def tutor(query: str) -> str:
            """An English tutor that explains grammar like you're 12."""
            return llm()

        async with app.lifespan():
            response = await tutor("How do I say hello in formal Japanese?")
            print(f"\nResponse: {response}")

    print("\nSUCCESS: hello-world ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
