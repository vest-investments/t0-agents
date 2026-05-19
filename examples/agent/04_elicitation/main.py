"""Elicitation: human-in-the-loop approval and structured input via MCP.

Demonstrates:
- MCP Elicitation for binary approval before destructive actions
- MCP Elicitation for structured data collection
- CLI-based elicitation handler that reads from stdin
- The handler distinguishes approval from structured input via response_type
- AgentConfig(tool_timeout=None) to disable timeout while waiting for human input

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
         docker compose --env-file ../../../.env \\
             -f examples/agent/04_elicitation/docker-compose.yml up -d
    3. Run: uv run python examples/agent/04_elicitation/main.py
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
from t0_agents.core.types import ToolProvider

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def cli_elicitation_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
    """Prompt the user in the terminal for approval or structured input."""
    print(f"\n{'=' * 60}")
    print("  ELICITATION REQUEST")
    print(f"{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")

    if response_type is None:
        # Binary approval -- no data expected, just y/n
        answer = await asyncio.to_thread(input, "  Approve? [y/n]: ")
        if answer.strip().lower() in ("y", "yes"):
            return ElicitResult(action="accept", content={})
        return ElicitResult(action="decline")

    # Structured input -- read the JSON schema and prompt for each field
    schema = params.requestedSchema
    properties = schema.get("properties", {})
    content: dict[str, Any] = {}

    print("  Please provide the following information:")
    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        raw = await asyncio.to_thread(input, f"    {field_name} ({field_type}): ")
        raw = raw.strip()
        if not raw:
            print("  Cancelled (empty input).")
            return ElicitResult(action="decline")
        if field_type == "integer":
            content[field_name] = int(raw)
        elif field_type == "number":
            content[field_name] = float(raw)
        elif field_type == "boolean":
            content[field_name] = raw.lower() in ("true", "yes", "1")
        else:
            content[field_name] = raw

    return ElicitResult(action="accept", content=content)


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    # Pass elicitation_handler when creating the Client -- this is the key change
    client: Client[Any] = Client(mcp, elicitation_handler=cli_elicitation_handler)

    async with await gateway as t0, client:
        agent = Agent(
            AgentConfig(
                name="task-agent",
                function_name="agent",
                tool_timeout=None,  # Disable timeout -- human will be typing
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, client),
        )
        result = await agent.invoke(
            "List all tasks. Delete task T-103. "
            "Then create a new task — the tool will collect the details."
        )
        print(f"\nResponse: {result.response}")
        print(f"Turns: {result.turns}")
        print(f"Episode ID: {result.episode_id}")
        print(f"Inference IDs: {result.inference_ids}")


if __name__ == "__main__":
    asyncio.run(main())
