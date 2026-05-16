"""FastMCP server for the writer sub-agent in demo 02.

Single tool ``format_outline`` with a short artificial latency so the
writer's work overlaps visibly with the researcher's when the
orchestrator delegates to both in the same turn.
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP

LATENCY_S = 1.0

mcp = FastMCP("parallel-demo-writing")


@mcp.tool
async def format_outline(topic: str) -> str:
    """Return a canned markdown outline for ``topic``.  Simulates a slow API."""
    await asyncio.sleep(LATENCY_S)
    return (
        f"# {topic}\n\n"
        "## Background\n- bullet\n\n"
        "## Findings\n- bullet\n\n"
        "## Next steps\n- bullet\n"
    )
