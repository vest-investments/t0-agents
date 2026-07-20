"""Standalone elicit-survival scenarios for the httpx read-window fuse.

Executed as a SUBPROCESS by ``test_mcp_timeouts.py`` (never imported by
pytest) so each run gets a pristine process: fastmcp 3.2.3's
``create_proxy`` per-call-client machinery has an unbounded connect race
when many client/uvicorn cycles share one process (``client_init_timeout``
defaults to ``None``), and these scenarios deliberately kill streams
mid-elicit — in-process they poison later, unrelated App+uvicorn tests
into order-dependent hangs.

Scenario A — heartbeat survival: a 5 s elicit completes through a 2 s
``MCPServer(timeout=)`` read window because ``ctx.info`` heartbeats every
0.5 s reset the gap timer.

Scenario B — silent death, bounded: without heartbeats the window kills
the wire at ~2 s; the call fails at its 8 s per-call deadline instead of
completing at ~5 s (which is what would happen if the window were still
the 300 s default — the differential with scenario A proves the knob
reaches the wire).  The deadline matters: the mcp SDK swallows the
read-timeout on a request's SSE stream and, absent ``id:``-bearing events
to resume from, never resolves the request — a dead stream plus
``timeout=None`` is an infinite hang, not an ``McpError``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast

import uvicorn
from fastmcp import Context, FastMCP
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError
from mcp import McpError
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, MCPServer


def _slow_approve(delay: float) -> Any:
    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        await asyncio.sleep(delay)
        return ElicitResult(action="accept", content={})

    return handler


def _make_slow_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("slow-hitl-http")

    @server.tool
    async def confirm_hb(ctx: Context) -> str:
        """Elicit while heartbeating ``ctx.info`` every 0.5 s.

        Log notifications carry ``related_request_id`` and ride the call's
        own SSE stream, so each one resets the client's httpx read-gap
        timer.  (``ctx.report_progress`` would silently no-op here: no
        progressToken flows on the proxy's inner call.)
        """

        async def beat() -> None:
            while True:
                await asyncio.sleep(0.5)
                await ctx.info("heartbeat")

        task = asyncio.create_task(beat())
        try:
            result = await ctx.elicit("Proceed?", response_type=None)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return "confirmed" if result.action == "accept" else "declined"

    @server.tool
    async def confirm_silent(ctx: Context) -> str:
        """Elicit with no heartbeat — the stream goes silent while waiting."""
        result = await ctx.elicit("Proceed?", response_type=None)
        return "confirmed" if result.action == "accept" else "declined"

    return server


@asynccontextmanager
async def _ephemeral_server(server: FastMCP[Any]) -> AsyncIterator[str]:
    config = uvicorn.Config(
        server.http_app(path="/mcp"), host="127.0.0.1", port=0, log_level="error"
    )
    srv = uvicorn.Server(config)
    serve_task = asyncio.create_task(srv.serve())
    while not srv.started:
        if serve_task.done():
            await serve_task
            raise RuntimeError("uvicorn exited before startup completed")
        await asyncio.sleep(0.02)
    port = srv.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        srv.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=10)
        except TimeoutError:
            srv.force_exit = True
            await serve_task


async def scenario_a_heartbeat_survival() -> None:
    async with _ephemeral_server(_make_slow_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, object()),
            servers=[MCPServer(url=url, namespace="hitl", timeout=2.0)],
            elicitation_handler=_slow_approve(5.0),
        )
        async with app.lifespan():
            client = app._owned_client
            assert client is not None
            result = await client.call_tool("hitl_confirm_hb", {}, timeout=None)
    assert result.data == "confirmed", f"expected completion, got {result.data!r}"
    print("SURVIVAL-OK A")


async def scenario_b_silent_death_bounded() -> None:
    async with _ephemeral_server(_make_slow_server()) as url:
        app = App(
            t0=cast(AsyncTensorZeroGateway, object()),
            servers=[MCPServer(url=url, namespace="hitl", timeout=2.0)],
            elicitation_handler=_slow_approve(5.0),
        )
        async with app.lifespan():
            client = app._owned_client
            assert client is not None
            start = time.monotonic()
            try:
                result = await client.call_tool(
                    "hitl_confirm_silent", {}, timeout=8.0
                )
            except (ToolError, McpError) as e:
                elapsed = time.monotonic() - start
                assert "timed out" in str(e).lower(), f"unexpected error: {e}"
                assert 5.0 < elapsed < 15.0, f"deadline did not bound: {elapsed:.1f}s"
            else:
                raise AssertionError(
                    f"call completed ({result.data!r}) — the 2s read window "
                    f"never reached the wire (300s default still in effect?)"
                )
    print("SURVIVAL-OK B")


async def main() -> None:
    await scenario_a_heartbeat_survival()
    await scenario_b_silent_death_bounded()


if __name__ == "__main__":
    asyncio.run(main())
