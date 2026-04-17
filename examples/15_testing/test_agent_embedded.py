"""Embedded-gateway integration tests — real inference, no Docker.

Opt-in: these tests run only when ``RUN_EMBEDDED_TESTS=1`` and a
provider API key for the configured model are present in the
environment.  ``tensorzero.toml`` (next to this file) declares a
tiny Groq-backed chat function; ``GROQ_API_KEY`` is therefore the
credential you need.

Unlike the unit tests, these assertions DO touch real model behaviour
— expect minor variance in wording.  Keep the checks permissive
(substring matches, not regex equality) and the timeouts generous.

Run locally::

    GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest examples/15_testing -m embedded
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from agent import build_agent, build_mcp_server
from fastmcp import Client

from t0_agents.core.types import ToolProvider
from t0_agents.testing import EmbeddedGatewayHelper

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDED_TESTS") != "1",
        reason="Set RUN_EMBEDDED_TESTS=1 (plus GROQ_API_KEY) to run",
    ),
]

_CONFIG = str(Path(__file__).parent / "tensorzero.toml")


async def test_real_agent_against_provider() -> None:
    """The agent completes a simple turn against real Groq inference."""
    mcp = build_mcp_server()
    async with (
        EmbeddedGatewayHelper.from_config(_CONFIG) as gw,
        Client(mcp) as client,
    ):
        agent = build_agent(gw, cast(ToolProvider, client))
        result = await agent.run("Say 'ready' and nothing else.")
        assert result.response
        # Permissive check — exact wording is model-dependent.
        assert "ready" in result.response.lower()


async def test_real_agent_invokes_tool() -> None:
    """When asked, the model picks a tool and the loop executes it."""
    mcp = build_mcp_server()
    async with (
        EmbeddedGatewayHelper.from_config(_CONFIG) as gw,
        Client(mcp) as client,
    ):
        agent = build_agent(gw, cast(ToolProvider, client))
        result = await agent.run(
            "Save a note titled 'hello' with body 'world', "
            "then confirm in one sentence."
        )
        # We can't assert exact wording, but we expect >= 1 turn with a
        # tool call — the response came after a tool result round-trip.
        assert result.turns >= 2
