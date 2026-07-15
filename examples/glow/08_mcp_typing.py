"""Glow ``MCPServer``: typed MCP integration.

Two patterns for declaring an MCP server a glow agent should see:

1. **Class-based** (used here): ``class MathMCP(MCPServer, server=...)``.
   The spec is recorded on the class via ``__init_subclass__``; pass the
   class as a typed parameter of ``@app.agent`` (or via the
   ``servers=`` kwarg on ``App``) to register the mount.
2. **Annotated** (commented out below): ``Annotated[Any,
   MCPServer(url=...)]`` for per-parameter overrides.

This example builds an in-memory FastMCP server with one tool and
mounts it onto the App-owned ``glow-app`` composed server.  The
classifier agent can then call the tool through ``app._owned_client``
just like any other ``@app.tool``.

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


# Uncomment to demonstrate the authenticated HTTP path (requires a
# reachable MCP endpoint + a real token).  ``headers=`` is static;
# ``auth=`` takes any ``httpx.Auth`` for per-request tokens/signatures:
# class GithubMCP(
#     MCPServer,
#     url="https://api.githubcopilot.com/mcp/",
#     namespace="github",
#     headers={"Authorization": "Bearer <token>"},
# ): ...
#
# Add ``prefix=False`` to present its tools verbatim (no ``github_``
# prefix) — the single-server / legacy-name case.


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
        # Verify the mounted tool is visible to the composed Client.
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
