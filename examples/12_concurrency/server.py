"""FastMCP server for the parallel-tools demos.

Exposes two tools that share the same artificial latency budget so the
wall-clock difference between sequential and parallel dispatch is easy
to see:

- ``fetch_url(url)``           — simulates one HTTP round trip (``LATENCY_S``).
- ``fetch_urls_batch(urls)``   — fans out N round trips in one tool call
  via ``asyncio.gather`` so a single-tool-per-turn model can still
  benefit from concurrency (the L2 "batch tool" pattern from
  ``docs/concurrency.md``).

Both tools return a deterministic per-URL string and log when they
start and finish, so the demos can show overlap on stderr without
depending on network access.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastmcp import FastMCP

LATENCY_S = 1.5

logger = logging.getLogger("examples.12_concurrency.server")

mcp = FastMCP("parallel-demo")


async def _simulate_fetch(url: str) -> str:
    start = time.monotonic()
    logger.info("fetch_url START  %s", url)
    await asyncio.sleep(LATENCY_S)
    elapsed = time.monotonic() - start
    logger.info("fetch_url END    %s  (%.2fs)", url, elapsed)
    return f"{url} -> 200 OK, body length 1024 (took {elapsed:.2f}s)"


@mcp.tool
async def fetch_url(url: str) -> str:
    """Fetch a URL.  Simulated latency: ``LATENCY_S`` seconds per call."""
    return await _simulate_fetch(url)


@mcp.tool
async def fetch_urls_batch(urls: list[str]) -> list[str]:
    """Fetch several URLs concurrently inside a single tool call.

    Demonstrates the L2 "batch tool" pattern: even when the LLM emits
    one tool call per turn, the tool itself can parallelise inside via
    ``asyncio.gather``.  Independent of ``parallel_tools``.
    """
    return await asyncio.gather(*[_simulate_fetch(u) for u in urls])
