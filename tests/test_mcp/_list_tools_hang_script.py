"""Never-returns list_tools scenario for the inventory fan-out deadline.

Executed as a SUBPROCESS by ``test_runtime.py`` (never imported by
pytest), following ``tests/test_glow/_timeout_survival_script.py``: the
scenario wedges one server's POST handler so its request stream simply
never answers.  Run in-process, a regression here would not fail but
hang, poisoning later App+uvicorn tests into order-dependent hangs; as
a subprocess the driver's hard timeout converts the hang into a loud
failure.

This is the regression proof for the 2026-08-04 vest-ai incident: four
servers answered the inventory fan-out, mcp-fred did not, and the turn
hung forever.  Against a pre-fix runtime this script never exits.

Prints ``LIST-TOOLS-OK <names> <elapsed>s`` and exits 0 when the merged
inventory comes back within the deadline despite the wedged server.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

import uvicorn
from fastmcp import FastMCP

from t0_agents.glow import MCPServer
from t0_agents.mcp.runtime import MCPRuntime


def _wedge_all_posts(app: Any) -> Any:
    """Hang every POST until the client hangs up — a dead request
    stream, the exact shape of a NotReady server behind a k8s Service."""

    async def wrapper(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["method"] == "POST":
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
        await app(scope, receive, send)

    return wrapper


@asynccontextmanager
async def _ephemeral(
    server: FastMCP[Any], wrap: Callable[[Any], Any] | None = None
) -> AsyncIterator[str]:
    app: Any = server.http_app(path="/mcp")
    if wrap is not None:
        app = wrap(app)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="error", lifespan="on"
    )
    http = uvicorn.Server(config)
    serve_task = asyncio.create_task(http.serve())
    try:
        while not http.started:
            if serve_task.done():
                await serve_task
                raise RuntimeError("uvicorn exited before startup completed")
            await asyncio.sleep(0.02)
        port = http.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        http.should_exit = True
        http.force_exit = True
        await serve_task


def _named_server(server_name: str, tool_name: str) -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP(server_name)

    async def impl() -> str:
        return f"{server_name}:{tool_name}"

    impl.__name__ = tool_name
    impl.__doc__ = f"Return the {tool_name} marker."
    server.tool(impl)
    return server


async def main() -> int:
    async with (
        _ephemeral(_named_server("healthy", "alive")) as healthy_url,
        _ephemeral(_named_server("wedged", "dark"), wrap=_wedge_all_posts) as wedged_url,
    ):
        specs = [
            MCPServer(url=healthy_url, namespace="healthy")._to_spec(),
            MCPServer(url=wedged_url, namespace="wedged")._to_spec(),
        ]
        try:
            runtime = MCPRuntime(specs, tool_list_timeout=2.0)
        except TypeError:
            # Pre-fix runtime (the against-main regression proof): no
            # knob exists and the fan-out below never returns.
            runtime = MCPRuntime(specs)
        started = asyncio.get_running_loop().time()
        tools = await runtime.list_tools()
        elapsed = asyncio.get_running_loop().time() - started
        names = sorted(tool.name for tool in tools)
        assert names == ["healthy_alive"], names
        assert elapsed < 30, elapsed
        with suppress(Exception):
            await runtime.aclose()
    print(f"LIST-TOOLS-OK {names} {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
