"""Unit tests for ``t0_agents.compose.orchestrator``.

Covers:

- ``bind_parent_episode_id`` / ``get_parent_episode_id`` — ContextVar
  semantics (set, nest, reset, None default).
- ``install_sub_agents`` error branches — non-Client tool provider,
  non-in-process transport, duplicate sub-agent names, collisions
  with existing tools.
- ``Agent.__init__`` integration — the happy-path install registers
  the expected ``{name}_ask`` tools on the orchestrator's in-process
  FastMCP server.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp import Client, FastMCP
from fastmcp.tools import Tool as FastMCPTool
from tensorzero import AsyncTensorZeroGateway

from t0_agents.compose.orchestrator import (
    bind_parent_episode_id,
    get_parent_episode_id,
    install_sub_agents,
)
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0


def _build_agent(
    name: str,
    *,
    tool_provider: ToolProvider | None = None,
    sub_agents: list[Agent] | None = None,
) -> Agent:
    config = AgentConfig(
        name=name,
        function_name=f"{name}_fn",
        sub_agents=sub_agents or [],
    )
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        tool_provider=tool_provider or cast(ToolProvider, FakeMCPClient()),
    )


def _list_tool_names(server: FastMCP) -> set[str]:
    components = server.local_provider._components
    return {c.name for c in components.values() if isinstance(c, FastMCPTool)}


# ---------------------------------------------------------------------------
# ContextVar machinery
# ---------------------------------------------------------------------------


class TestParentEpisodeContextVar:
    def test_default_is_none(self) -> None:
        assert get_parent_episode_id() is None

    def test_bind_sets_and_resets(self) -> None:
        with bind_parent_episode_id("ep-parent"):
            assert get_parent_episode_id() == "ep-parent"
        assert get_parent_episode_id() is None

    def test_bind_none_is_valid(self) -> None:
        # Agent.run passes episode_id verbatim; None is the common path
        # for fresh orchestrator runs and must not raise here.
        with bind_parent_episode_id(None):
            assert get_parent_episode_id() is None

    def test_nested_bindings_shadow_and_restore(self) -> None:
        with bind_parent_episode_id("outer"):
            assert get_parent_episode_id() == "outer"
            with bind_parent_episode_id("inner"):
                assert get_parent_episode_id() == "inner"
            assert get_parent_episode_id() == "outer"
        assert get_parent_episode_id() is None


# ---------------------------------------------------------------------------
# install_sub_agents error branches
# ---------------------------------------------------------------------------


class TestInstallSubAgentsErrors:
    def test_non_fastmcp_client_provider_raises(self) -> None:
        # The sub_agents list is populated by the caller; build the
        # orchestrator Agent with a ``FakeMCPClient`` (not a real
        # fastmcp.Client) to trigger the Client-isinstance guard.
        researcher = _build_agent("researcher")
        with pytest.raises(ConfigurationError, match=r"fastmcp\.Client"):
            _build_agent(
                "orch",
                tool_provider=cast(ToolProvider, FakeMCPClient()),
                sub_agents=[researcher],
            )

    def test_non_inprocess_transport_raises(self) -> None:
        # Build a real ``Client`` but with a transport that is not a
        # ``FastMCPTransport``.  Easiest way: a Client that resolves to
        # an ``mcp://`` transport via infer_transport would require
        # network calls; instead, stub ``client.transport`` with an
        # arbitrary object after construction.
        server = FastMCP("parent-server")
        client: Client[Any] = Client(server)

        class _FakeTransport:
            def __repr__(self) -> str:
                return "fake"

        client.transport = _FakeTransport()  # type: ignore[assignment]

        researcher = _build_agent("researcher")
        with pytest.raises(ConfigurationError, match="in-process FastMCPTransport"):
            _build_agent(
                "orch",
                tool_provider=cast(ToolProvider, client),
                sub_agents=[researcher],
            )

    def test_duplicate_sub_agent_names_raise(self) -> None:
        server = FastMCP("parent-server")
        client: Client[Any] = Client(server)

        researcher_a = _build_agent("researcher")
        researcher_b = _build_agent("researcher")

        with pytest.raises(ConfigurationError, match="Duplicate sub-agent names"):
            _build_agent(
                "orch",
                tool_provider=cast(ToolProvider, client),
                sub_agents=[researcher_a, researcher_b],
            )

    def test_collision_with_existing_tool_raises(self) -> None:
        server = FastMCP("parent-server")

        @server.tool
        def researcher_ask(message: str) -> str:  # pragma: no cover - registered only
            return "pre-existing"

        client: Client[Any] = Client(server)
        researcher = _build_agent("researcher")

        with pytest.raises(ConfigurationError, match="collides with an existing tool"):
            _build_agent(
                "orch",
                tool_provider=cast(ToolProvider, client),
                sub_agents=[researcher],
            )

    def test_sub_agents_field_rejects_non_agent_entries(self) -> None:
        # AgentConfig.__post_init__ catches this before Agent.__init__,
        # so installation never starts.
        with pytest.raises(
            ConfigurationError, match="sub_agents\\[0\\] must be an Agent"
        ):
            AgentConfig(
                name="orch",
                function_name="orch_fn",
                sub_agents=[cast(Agent, "not-an-agent")],
            )


# ---------------------------------------------------------------------------
# install_sub_agents happy path
# ---------------------------------------------------------------------------


class TestInstallSubAgentsHappyPath:
    def test_registers_one_tool_per_sub_agent(self) -> None:
        server = FastMCP("parent-server")
        client: Client[Any] = Client(server)

        researcher = _build_agent("researcher")
        writer = _build_agent("writer")

        orchestrator = _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, client),
            sub_agents=[researcher, writer],
        )

        assert isinstance(orchestrator, Agent)
        registered = _list_tool_names(server)
        assert {"researcher_ask", "writer_ask"} <= registered

    def test_noop_when_sub_agents_empty(self) -> None:
        # Direct invocation of install_sub_agents with no sub-agents is
        # explicitly the no-op path; the Agent need not even have a
        # fastmcp.Client tool provider in that case.
        orch = _build_agent("orch")  # FakeMCPClient — invalid if sub_agents set
        install_sub_agents(orch)  # must not raise

    def test_does_not_pollute_subsequent_servers(self) -> None:
        # Each construction uses its own FastMCP instance; verify that
        # installing on one server does not leak to a second.
        server_a = FastMCP("a")
        server_b = FastMCP("b")
        client_a: Client[Any] = Client(server_a)
        client_b: Client[Any] = Client(server_b)

        researcher = _build_agent("researcher")
        _build_agent(
            "orch_a",
            tool_provider=cast(ToolProvider, client_a),
            sub_agents=[researcher],
        )

        writer = _build_agent("writer")
        _build_agent(
            "orch_b",
            tool_provider=cast(ToolProvider, client_b),
            sub_agents=[writer],
        )

        assert _list_tool_names(server_a) == {"researcher_ask"}
        assert _list_tool_names(server_b) == {"writer_ask"}
