"""Integration tests for Agent -- require T0 Gateway running."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider
from tests.server import mcp

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    """Helper to narrow `build_http`'s Union return to the async branch."""
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_agent_run_end_to_end(mcp_client: Client[Any]) -> None:
    """Agent.run() works end-to-end against the live gateway."""
    config = AgentConfig(
        name="integration-test",
        function_name="agent",
        dryrun=True,
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        result = await agent.run("Say hi briefly.")

    assert result.response
    assert result.episode_id is not None
    assert len(result.inference_ids) >= 1
    assert result.turns >= 1


async def test_agent_episode_continuation(mcp_client: Client[Any]) -> None:
    """Agent can continue an episode by passing episode_id."""
    config = AgentConfig(
        name="continuation-test",
        function_name="agent",
        dryrun=True,
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        r1 = await agent.run("Say hi briefly.")
        r2 = await agent.run("What did I just say?", episode_id=r1.episode_id)

    assert r1.episode_id is not None
    assert r2.episode_id is not None
    assert r2.response
