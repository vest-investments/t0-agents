"""FastMCP server for the researcher sub-agent in demo 02.

Single tool ``search_papers`` with a short artificial latency so the
researcher's work overlaps visibly with the writer's when the
orchestrator delegates to both in the same turn.
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP

LATENCY_S = 1.0

mcp = FastMCP("parallel-demo-research")


@mcp.tool
async def search_papers(query: str) -> list[str]:
    """Return canned paper titles for ``query``.  Simulates a slow API."""
    await asyncio.sleep(LATENCY_S)
    return [
        f"[{query}] Paper A: a foundational result",
        f"[{query}] Paper B: a follow-up experiment",
        f"[{query}] Paper C: a recent survey",
    ]
