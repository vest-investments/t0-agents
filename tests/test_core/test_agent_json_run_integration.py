"""Integration tests for ``Agent.run`` JSON dispatch.

Requires ``docker compose up -d`` from the repo root with
``GROQ_API_KEY`` populated in ``.env``.  The gateway config at
``config/tensorzero.toml`` declares the ``extract_customer_info`` JSON
function used here — the same function the structured-inference helper
integration test exercises.

The model output is not asserted exactly (Groq's gpt-oss-20b may produce
slight variations).  Assertions cover the dispatch contract: the agent
returns an ``AgentResult`` with ``function_kind == "json"``, a non-None
``parsed`` carrying the schema's two keys, ``response`` equal to the
raw JSON string, ``turns == 1``, and a single inference id.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.outputs import JsonOutput
from t0_agents.core.types import ToolProvider

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_agent_run_json_end_to_end(mcp_client: Client[Any]) -> None:
    """``Agent.run`` dispatches to the JSON path against the live gateway."""
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
        result = await agent.invoke(
            "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",
        )

    assert result.function_kind == "json"
    assert result.parsed is not None
    assert isinstance(result.parsed, dict)
    assert set(result.parsed.keys()) == {"name", "email"}
    assert isinstance(result.parsed["name"], str | type(None))
    assert isinstance(result.parsed["email"], str | type(None))
    assert result.response  # raw JSON string is non-empty
    assert result.turns == 1
    assert result.episode_id is not None
    assert len(result.inference_ids) == 1
    assert result.response_source == "text"
