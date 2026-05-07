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
from t0_agents.stream.events import DoneEvent
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


async def test_agent_run_system_arguments_against_strict_schema(
    mcp_client: Client[Any],
) -> None:
    """`system_arguments` is sent in the shape TensorZero validates.

    Regression guard for the wrapper bug: the function `agent_with_schema`
    declares a system schema with ``additionalProperties: false`` and a
    single required ``max_turns`` integer.  The gateway will reject any
    payload that does not match that exact shape, so this test is
    sufficient to catch any future regression that re-introduces an
    ``arguments`` envelope (or any other shape mismatch) on the wire.
    """
    config = AgentConfig(
        name="system-args-schema-test",
        function_name="agent_with_schema",
        dryrun=True,
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        result = await agent.run(
            "Say hi briefly.",
            system_arguments={"max_turns": 5},
        )

    assert result.response
    assert result.episode_id is not None


async def test_agent_stream_system_arguments_against_strict_schema(
    mcp_client: Client[Any],
) -> None:
    """Streaming counterpart to the schema-validation regression guard."""
    config = AgentConfig(
        name="system-args-schema-stream-test",
        function_name="agent_with_schema",
        dryrun=True,
    )
    async with await _gateway_factory() as t0, mcp_client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, mcp_client),
        )
        terminal: DoneEvent | None = None
        async for event in agent.stream(
            "Say hi briefly.",
            system_arguments={"max_turns": 5},
        ):
            if isinstance(event, DoneEvent):
                terminal = event
                break

    assert terminal is not None
    assert terminal.result.response
    assert terminal.result.episode_id is not None
