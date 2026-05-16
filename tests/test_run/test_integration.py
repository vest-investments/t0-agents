"""Integration smoke for :class:`Run` against a live TensorZero gateway.

Requires the gateway to be running (``docker compose up -d`` at the
repo root with ``--env-file .env`` for ``GROQ_API_KEY``).  These tests
exercise the canonical entry point :meth:`Agent.invoke` end-to-end so
the lazy-dispatch model is validated against a real provider.

Mirrors the assertion style of ``tests/test_infrastructure.py`` —
inference IDs are real, episode IDs are real, the response shape
matches v0.12.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def gateway_url() -> str:
    return os.environ.get(
        "TENSORZERO_GATEWAY_URL", "http://localhost:3000",
    )


def _make_test_server() -> FastMCP[Any]:
    mcp: FastMCP[Any] = FastMCP("test-run-integration")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return mcp


class TestInvokeEndToEnd:
    async def test_chat_invoke_returns_run_with_result(
        self,
        gateway_url: str,
    ) -> None:
        gateway = cast(
            Awaitable[AsyncTensorZeroGateway],
            AsyncTensorZeroGateway.build_http(gateway_url=gateway_url),
        )
        mcp = _make_test_server()
        client: Client[Any] = Client(mcp)
        async with await gateway as t0, client:
            agent = Agent(
                AgentConfig(
                    name="integ",
                    function_name="agent",
                    dryrun=True,
                    max_turns=4,
                ),
                t0=t0,
                tool_provider=client,
            )
            run = agent.invoke("What is 2 + 3?")
            result = await run.result()
            # AgentResult is populated end-to-end against the real
            # gateway — episode and inference ids are present.
            assert result.response
            assert result.episode_id
            assert len(result.inference_ids) >= 1
            assert result.turns >= 1
            # Run.state is terminal.
            assert run.state == "succeeded"
            # Read-side properties mirror the AgentResult.
            assert run.episode_id == result.episode_id
            assert run.inference_ids == tuple(result.inference_ids)

    async def test_invoke_then_stream_via_events(
        self,
        gateway_url: str,
    ) -> None:
        gateway = cast(
            Awaitable[AsyncTensorZeroGateway],
            AsyncTensorZeroGateway.build_http(gateway_url=gateway_url),
        )
        mcp = _make_test_server()
        client: Client[Any] = Client(mcp)
        async with await gateway as t0, client:
            agent = Agent(
                AgentConfig(
                    name="integ-stream",
                    function_name="agent",
                    dryrun=True,
                    max_turns=4,
                ),
                t0=t0,
                tool_provider=client,
            )
            run = agent.invoke("Say hello.")
            collected: list[Any] = []
            async for ev in run.events():
                collected.append(ev)
            # At least the terminal DoneEvent fires.
            assert any(
                getattr(ev, "type", None) == "done" for ev in collected
            )
            assert run.state == "succeeded"
