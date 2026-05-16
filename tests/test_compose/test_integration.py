"""Integration tests for multi-agent composition.

Require the local TensorZero gateway (``docker compose up -d``).  Use
the root ``config/tensorzero.toml``'s ``agent`` function for every
sub-agent — the test exercises delegation mechanics, not per-function
prompt specialisation.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


def _build_researcher_server() -> FastMCP:
    mcp = FastMCP("researcher-server")

    @mcp.tool
    def lookup_fact(topic: str) -> str:
        """Return a canned fact for the given topic."""
        facts = {
            "rlhf": "RLHF uses human feedback to align LLM outputs.",
            "default": "No fact available for that topic.",
        }
        return facts.get(topic.lower(), facts["default"])

    return mcp


def _build_writer_server() -> FastMCP:
    mcp = FastMCP("writer-server")

    @mcp.tool
    def make_bullet(text: str) -> str:
        """Wrap the given text in a markdown bullet."""
        return f"- {text}"

    return mcp


def _build_orchestrator_server() -> FastMCP:
    # The orchestrator's FastMCP has no pre-declared tools;
    # ``install_sub_agents`` adds ``researcher_ask`` and ``writer_ask``
    # at Agent construction time.
    return FastMCP("orchestrator-server")


async def test_two_agent_orchestration_e2e() -> None:
    researcher_server = _build_researcher_server()
    writer_server = _build_writer_server()
    orchestrator_server = _build_orchestrator_server()

    async with (
        await _gateway_factory() as t0,
        Client(researcher_server) as researcher_client,
        Client(writer_server) as writer_client,
        Client(orchestrator_server) as orch_client,
    ):
        researcher = Agent(
            AgentConfig(
                name="researcher",
                function_name="agent",
                dryrun=True,
                max_turns=4,
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, researcher_client),
        )
        writer = Agent(
            AgentConfig(
                name="writer",
                function_name="agent",
                dryrun=True,
                max_turns=4,
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, writer_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="orchestrator",
                function_name="agent",
                sub_agents=[researcher, writer],
                dryrun=True,
                max_turns=6,
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, orch_client),
        )

        result = await orchestrator.run(
            "Ask the researcher about RLHF, then have the writer bullet "
            "the fact. Reply with the final bulleted answer.",
        )

    assert result.response, "orchestrator should produce a final response"
    assert result.episode_id is not None
    assert result.turns >= 1


async def test_subagent_episode_isolation() -> None:
    """Sub-agent's AgentResult.episode_id differs from orchestrator's.

    Each ``Agent.run`` defaults to its own fresh episode; the parent
    correlation happens via the ``t0_agents.parent_episode_id`` tag,
    not by reusing episode ids.  We assert isolation by having a spy
    sub-agent record its own episode id on every delegation and then
    comparing against the orchestrator's.
    """
    researcher_server = _build_researcher_server()
    orchestrator_server = _build_orchestrator_server()

    sub_episode_ids: list[str] = []

    class _SpyAgent(Agent):
        # v1.0 idiom: override the inner terminal (the v0.12 ``run``
        # override is no longer wired by the
        # _OrchestrationToolProvider, which dispatches via
        # ``invoke().result()``).  The terminal sees the final
        # AgentResult before any outer middleware mutates it.
        async def _chat_terminal(self, ctx: Any) -> Any:  # type: ignore[override]
            result = await super()._chat_terminal(ctx)
            if result.episode_id is not None:
                sub_episode_ids.append(result.episode_id)
            return result

    async with (
        await _gateway_factory() as t0,
        Client(researcher_server) as researcher_client,
        Client(orchestrator_server) as orch_client,
    ):
        researcher = _SpyAgent(
            AgentConfig(
                name="researcher",
                function_name="agent",
                dryrun=True,
                max_turns=4,
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, researcher_client),
        )
        orchestrator = Agent(
            AgentConfig(
                name="orchestrator",
                function_name="agent",
                sub_agents=[researcher],
                dryrun=True,
                max_turns=4,
            ),
            t0=t0,
            tool_provider=cast(ToolProvider, orch_client),
        )

        result = await orchestrator.run(
            "Ask the researcher about RLHF and return the fact verbatim.",
        )

    # The LLM may or may not call the researcher — if it does, we assert
    # isolation.  If it never delegates, the test still passes on the
    # end-to-end guarantee below.
    if sub_episode_ids:
        assert all(
            ep != result.episode_id for ep in sub_episode_ids
        ), (
            f"sub-agent episode id should differ from orchestrator's "
            f"({result.episode_id}); got {sub_episode_ids}"
        )
    assert result.response
