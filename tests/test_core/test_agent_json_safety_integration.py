"""Integration test for the JSON streaming guard.

Requires ``docker compose up -d`` from the repo root with
``GROQ_API_KEY`` populated in ``.env``.  Confirms that
``Agent.stream(...)`` on a JSON-flavoured agent raises
:class:`ConfigurationError` at the first ``__anext__`` — no provider
round-trip, no SSE setup, just a fail-fast guard.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import JsonOutput
from t0_agents.core.types import ToolProvider

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_stream_guard_against_real_gateway(
    mcp_client: Client[Any],
) -> None:
    """``Agent.stream`` on a JSON agent raises at first ``__anext__``."""
    config = AgentConfig(
        name="customer-extractor",
        function_name="extract_customer_info",
        output=JsonOutput(json_mode="strict"),
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        gen = agent.invoke("Hi, I'm Sarah at sarah@example.com").events()
        with pytest.raises(ConfigurationError, match=r"streaming"):
            await gen.__anext__()
