"""Integration tests for Agent + memory — require T0 Gateway running."""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider

pytestmark = pytest.mark.integration

aiosqlite = pytest.importorskip("aiosqlite")
from t0_agents.memory.sqlite import SqliteMemory  # noqa: E402


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_agent_remembers_across_runs(mcp_client: Client[Any]) -> None:
    """Two successive ``agent.invoke()`` calls; second recalls the first."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        mem = SqliteMemory(tmp.name)
        config = AgentConfig(
            name="memory-test",
            function_name="agent",
            memory=mem,
            dryrun=True,
        )
        async with await _gateway_factory() as t0, mcp_client:
            agent = Agent(
                config,
                t0=t0,
                tool_provider=cast(ToolProvider, mcp_client),
            )
            r1 = await agent.invoke(
                "Please remember: my favourite colour is blue."
            )
            assert r1.episode_id is not None

            r2 = await agent.invoke(
                "What is my favourite colour?",
                episode_id=r1.episode_id,
            )

        # The second response should reference the colour
        assert r2.response
        # Memory should have two records
        records = await mem.load(r1.episode_id)
        assert len(records) == 2
