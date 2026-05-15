"""Integration test for the JSON retry orchestration against a real provider.

Reuses the ``extract_customer_info`` JSON function configured in
``config/tensorzero.toml``, but overrides ``json_mode="off"`` per call so
the TensorZero gateway does *not* enforce schema-compliant output.  That
lets the underlying model occasionally emit prose-wrapped or
fenced-block JSON, which is exactly the recovery surface the
orchestration targets.

This test is informative rather than strict: the model's behaviour is
non-deterministic, so the assertions only require that the agent
returns a sensible :class:`AgentResult`.  When a recovery path was
exercised, the test prints the strategy and attempt count to ``stderr``
so a flake-watcher can confirm the test is doing something useful.

Requires ``docker compose up -d`` from the repo root with
``GROQ_API_KEY`` populated in ``.env``.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Narrow ``build_http``'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_agent_json_retry_against_real_provider(
    mcp_client: Client[Any],
) -> None:
    """End-to-end smoke for permissive + retry on a real provider.

    With ``json_mode="off"`` and a deliberately loose prompt, the model
    *may* emit prose-wrapped JSON.  The agent's permissive layer or
    retry layer should still produce a non-None ``parsed`` in the vast
    majority of runs.  Either of the four ``parse_strategy`` values is
    acceptable — we are only asserting that the orchestration completes,
    that every attempt's id was recorded, and that ``turns`` matches the
    attempt count.
    """
    config = AgentConfig(
        name="customer-extractor-loose",
        function_name="extract_customer_info",
        function_kind="json",
        json_mode="strict",  # config baseline; overridden per call below
        permissive_json_parsing=True,
        max_json_retries=2,
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        result = await agent.run(
            "Tell me about Sarah Johnson, sarah.j@example.com. "
            "Feel free to wrap your reply in a brief comment.",
            json_mode="off",
        )

    sys.stderr.write(
        f"[m3-integration] parse_strategy={result.parse_strategy!r} "
        f"attempts={result.json_parse_attempts} "
        f"turns={result.turns} "
        f"inference_ids={len(result.inference_ids)}\n"
    )

    assert result.function_kind == "json"
    assert result.episode_id is not None
    assert result.parse_strategy in {"strict", "permissive", "retry", "none"}
    assert result.json_parse_attempts >= 0
    assert result.turns == 1 + result.json_parse_attempts
    assert len(result.inference_ids) == result.turns
