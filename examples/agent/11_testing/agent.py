"""The agent under test.

Three tools, one agent — a compact but realistic shape for both the
unit test (MockGateway + MockToolProvider) and the embedded test
(real inference against ``build_embedded``).

The agent is exposed via :func:`build_agent` which takes the gateway
and the tool provider as arguments so the tests can swap them out
without touching the agent's configuration.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider


def build_mcp_server() -> FastMCP[Any]:
    """Register the three demo tools on a fresh FastMCP server."""
    mcp: FastMCP[Any] = FastMCP("testing-demo")

    notes: dict[str, str] = {}

    @mcp.tool
    def search_notes(query: str) -> list[str]:
        """Return the titles of notes whose body contains *query*."""
        return sorted(
            title for title, body in notes.items() if query.lower() in body.lower()
        )

    @mcp.tool
    def save_note(title: str, body: str) -> str:
        """Persist a note and return a confirmation."""
        notes[title] = body
        return f"Saved {title!r} ({len(body)} chars)"

    @mcp.tool
    def read_note(title: str) -> str:
        """Return the body of the note with *title*, or a not-found string."""
        return notes.get(title, "<not found>")

    return mcp


def build_agent(
    t0: AsyncTensorZeroGateway,
    tool_provider: ToolProvider,
    *,
    function_name: str = "assistant",
    max_turns: int = 5,
) -> Agent:
    """Build the agent under test.

    Keeping the construction trivial — every knob has a default —
    lets the tests focus on scripting *behaviour* instead of
    boilerplate configuration.
    """
    return Agent(
        AgentConfig(
            name="notes-assistant",
            function_name=function_name,
            max_turns=max_turns,
        ),
        t0=t0,
        tool_provider=tool_provider,
    )
