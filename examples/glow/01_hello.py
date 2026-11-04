"""Glow hello-world: the 8-line DX.

A chat agent defined entirely by its function signature and docstring —
no gateway ceremony, no ``AgentConfig`` boilerplate, no ``Run``
juggling.  The docstring becomes the system prompt; the ``-> str``
return annotation drives ``ChatOutput`` inference; ``await tutor(msg)``
returns the unwrapped string directly.

The module-level ``App()`` resolves its gateway at first use — set one
of these (GROQ_API_KEY in the environment or .env either way):

    TENSORZERO_URL=http://localhost:3000
        HTTP gateway, started with:
        docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    TENSORZERO_CONFIG=examples/glow/config/tensorzero.toml
        Embedded gateway: runs in-process — no docker at all.

Run it:

    uv run python examples/glow/01_hello.py

Prefer explicit wiring?  ``App(gateway="http://localhost:3000")``,
``App(embedded="examples/glow/config/tensorzero.toml")``, or the classic
``App(t0=<your AsyncTensorZeroGateway>)`` all work the same way.
"""

from __future__ import annotations

import asyncio

from t0_agents.glow import App, llm

app = App()


@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""
    return llm()


async def main() -> None:
    async with app:
        print(await tutor("How do I say hello in formal Japanese?"))


if __name__ == "__main__":
    asyncio.run(main())
    print("\nSUCCESS: hello-world ran end-to-end.")
