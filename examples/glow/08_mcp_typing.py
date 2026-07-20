"""Glow ``MCPServer``: typed MCP integration.

Two patterns for declaring an MCP server a glow agent should see:

1. **Class-based** (used here): ``class MathMCP(MCPServer, server=...)``.
   The spec is recorded on the class via ``__init_subclass__``; pass the
   class as a typed parameter of ``@app.agent`` (or via the
   ``servers=`` kwarg on ``App``) to register the mount.
2. **Annotated** (commented out below): ``Annotated[Any,
   MCPServer(url=...)]`` for per-parameter overrides.

This example builds an in-memory FastMCP server with one tool and
mounts it on the App-owned MCP runtime.  The classifier agent can then
call the tool just like any other ``@app.tool``.

To unit-test a mount like this in-memory (no gateway, no sockets), see
the ``glow_tools`` helper in the testing guide (``docs/testing.md`` §
Glow MCP servers).

Prerequisites:
    1. Set GROQ_API_KEY in your environment (or in .env at the project root).
    2. Start the gateway:
       docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
    3. Run: uv run python examples/glow/08_mcp_typing.py

Remote MCP servers (HTTP/SSE) work too — uncomment the ``GithubMCP``
block below.  A ``url=`` mount takes first-class ``headers=`` (static)
and ``auth=`` (an ``httpx.Auth``, applied per request), so an
authenticated endpoint like ``api.githubcopilot.com/mcp`` needs no manual
``create_proxy`` bridge; ``prefix=False`` presents its tools unprefixed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from t0_agents.glow import App, MCPServer, llm


def _build_math_server() -> FastMCP[Any]:
    """One-tool in-memory FastMCP server used as the example's backend."""
    server: FastMCP[Any] = FastMCP("math-tools")

    @server.tool
    async def add(a: int, b: int) -> int:
        """Return the integer sum of ``a`` and ``b``."""
        return a + b

    return server


_MATH_SERVER = _build_math_server()


class MathMCP(MCPServer, server=_MATH_SERVER, namespace="math"):
    """Typed declaration that mounts the math server on the App."""


# Authenticated remote mount (illustrative — a real ``url=`` endpoint and
# token are required, so it stays commented).  ``headers=`` is static and
# ``auth=`` takes any ``httpx.Auth`` for per-request credentials; these
# apply only to a remote ``url=`` mount, never to an in-memory ``server=``.
# ``prefix=False`` presents the tools unprefixed (single-server case).
# class GithubMCP(
#     MCPServer,
#     url="https://api.githubcopilot.com/mcp/",
#     namespace="github",
#     headers={"Authorization": "Bearer <token>"},
#     # auth=MyHttpxAuth(),   # per-request credentials
#     # prefix=False,         # present tools unprefixed
# ): ...
#
# The authenticated HTTP path is exercised end-to-end against a live local
# server in tests/test_glow/test_mcp_authed_mounts.py, and documented in
# docs/glow/typed-mcp-and-guardrails.md § Authenticated remote mounts.


async def main() -> None:
    # ``servers=[MathMCP]`` seeds the App's spec list at construction
    # time.  The same effect can be obtained by mentioning ``MathMCP``
    # as a typed parameter on any ``@app.agent`` function; both
    # paths converge in ``app._mcp_specs``.
    app = App(gateway="http://localhost:3000", servers=[MathMCP])

    @app.agent("calc")
    async def calc(query: str) -> str:
        """Math assistant.  Use the available `math_add` tool to
        compute sums.  Reply with the integer result only."""
        return llm()

    async with app:
        # Verify the mounted tool is visible through the App-owned runtime.
        # `app._owned_client` is glow-internal; a public `app.tools()`
        # inventory API lands in a later release.
        tools = await app._owned_client.list_tools()
        tool_names = sorted(t.name for t in tools)
        print(f"Visible tools: {tool_names}")
        assert "math_add" in tool_names, (
            "math_add not mounted; check namespace handling"
        )

        response = await calc("What is 2 + 2?  Use math_add.")
        print(f"\nResponse: {response}")

    print("\nSUCCESS: mcp-typing ran end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
