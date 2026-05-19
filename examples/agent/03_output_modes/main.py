"""Single-shot JSON extraction with ``run_structured_inference``.

Prerequisites:
    1. Set ``GROQ_API_KEY`` in ``.env`` at the project root.
    2. Start the gateway with this example's config:
         docker compose -f examples/03_output_modes/docker-compose.yml \\
             --env-file .env up -d
    3. Run:
         uv run python examples/03_output_modes/main.py

The example calls the ``extract_customer_info`` JSON function defined
in ``examples/03_output_modes/config/tensorzero.toml`` and prints the
``StructuredOutput`` fields the helper exposes (``parsed``, ``raw``,
``parse_strategy``, ``usage``).  No MCP server is needed — JSON
functions are single-shot and do not call tools.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.structured import run_structured_inference


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    async with await gateway as t0:
        result = await run_structured_inference(
            t0,
            function_name="extract_customer_info",
            user_message=(
                "Hi, I'm Sarah Johnson and you can reach me at "
                "sarah.j@example.com"
            ),
            json_mode="strict",
        )

    print(f"Parsed: {result.parsed}")
    print(f"Raw: {result.raw}")
    print(f"Parse strategy: {result.parse_strategy}")
    print(f"Episode ID: {result.episode_id}")
    print(f"Inference ID: {result.inference_id}")
    if result.usage is not None:
        print(
            f"Usage: input_tokens={result.usage.input_tokens}, "
            f"output_tokens={result.usage.output_tokens}, "
            f"cost={result.usage.cost}"
        )


if __name__ == "__main__":
    asyncio.run(main())
