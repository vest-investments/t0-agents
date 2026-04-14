"""Robustness sanity-check against vest-mcp-gcp (Cloudflare Access-protected).

Reads CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET from the shell env and
exercises the ToolError, timeout, and max-turns error paths against a real
production MCP server + the local TensorZero gateway at http://localhost:3000.

Usage:
    # make sure the two CF env vars are exported and `docker compose up -d`
    uv run python scripts/robustness_sanity.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import MaxTurnsExceededError, run_agent

CF_ID = os.environ["CF_ACCESS_CLIENT_ID"]
CF_SECRET = os.environ["CF_ACCESS_CLIENT_SECRET"]

# GCP project al que apuntamos las queries de sanity-check.
PROJECT_ID = "northbound-prod"

CONFIG: dict[str, Any] = {
    "mcpServers": {
        "mcp-gcp": {
            "url": "https://vest-mcp-gcp.sproutfi.app/mcp-server-mcp-gcp/mcp",
            "transport": "http",
            "headers": {
                "CF-Access-Client-Id": CF_ID,
                "CF-Access-Client-Secret": CF_SECRET,
            },
        },
    },
}


async def main() -> None:
    mcp_client: Client[Any] = Client(CONFIG)
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )

    async with mcp_client, await gateway as t0:
        # --- 0. Discovery ----------------------------------------------------
        tools = await mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"\n[discovery] {len(tools)} tools disponibles:")
        for name in tool_names:
            print(f"  - {name}")
        if not tools:
            print("  (sin tools — revisa CF Access / URL)")
            return

        # --- [tool-error] Loop sobrevive un tool que falla -------------------
        # Forzamos un execute_sql contra una tabla inexistente: BigQuery
        # responde 404, mcp-gcp lo propaga como ToolError, nuestro loop lo
        # atrapa y lo surface al LLM.
        print("\n[tool-error] execute_sql contra tabla inexistente…")
        try:
            result = await run_agent(
                t0, mcp_client,
                (
                    f"Llamá execute_sql con project_id='{PROJECT_ID}' y "
                    f"query='SELECT * FROM `{PROJECT_ID}."
                    "this_dataset_does_not_exist.fake_table` LIMIT 1'. "
                    "Es un test de error handling — cuéntame exactamente "
                    "qué devuelve BigQuery."
                ),
                "agent", dryrun=True, max_turns=4,
            )
            print(f"  → response: {result.response[:300]!r}")
            print(f"  → turns:    {result.turns}")
        except MaxTurnsExceededError as e:
            print(
                f"  → MaxTurnsExceededError(max_turns={e.max_turns}) — "
                "también tool-error-safe (todos los errores se atraparon)"
            )

        # --- [timeout] Timeout no crashea el loop -----------------------------
        # tool_timeout=0.1s vs. cualquier tool call real contra un server
        # remoto detrás de CF Access: el round-trip siempre excede 100ms.
        # McpError(-32000) del lado cliente → atrapado por nuestro loop.
        print("\n[timeout] tool_timeout=0.1 contra list_dataset_ids real…")
        try:
            result = await run_agent(
                t0, mcp_client,
                (
                    f"Llamá list_dataset_ids con project_id='{PROJECT_ID}' "
                    "y contame cuántos datasets encontrás."
                ),
                "agent", dryrun=True, tool_timeout=0.1, max_turns=4,
            )
            print(f"  → response: {result.response[:300]!r}")
            print(f"  → turns:    {result.turns}")
        except MaxTurnsExceededError as e:
            print(
                f"  → MaxTurnsExceededError(max_turns={e.max_turns}) — "
                "timeout-safe (McpError de timeout atrapado)"
            )

        # --- [max-turns] Max-turn raise ---------------------------------------
        # Prompt directivo: list_dataset_ids contra un project_id real casi
        # siempre produce una tool call en el turno 1, lo que agota el
        # presupuesto max_turns=1 y fuerza el raise.
        print("\n[max-turns] max_turns=1 fuerza el raise tras una tool call…")
        try:
            await run_agent(
                t0, mcp_client,
                (
                    f"Llamá list_dataset_ids con project_id='{PROJECT_ID}'. "
                    "Después resumime los resultados."
                ),
                "agent", dryrun=True, max_turns=1,
            )
            print(
                "  → inesperado: el loop volvió sin raise; el modelo no "
                "emitió una tool call en el turno 1"
            )
        except MaxTurnsExceededError as e:
            print(
                f"  → MaxTurnsExceededError(max_turns={e.max_turns}) — "
                "esperado"
            )


if __name__ == "__main__":
    asyncio.run(main())
