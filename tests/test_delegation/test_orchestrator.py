"""Unit tests for ``t0_agents.delegation.orchestrator``.

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

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.tools import Tool as FastMCPTool
from mcp.types import TextContent
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
    SubAgentError,
)
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.delegation.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    _OrchestrationToolProvider,
    bind_parent_episode_id,
    get_parent_episode_id,
    install_sub_agents,
    update_parent_episode_id,
)
from t0_agents.stream.events import AgentEvent, DoneEvent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


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

    def test_update_overwrites_bound_value(self) -> None:
        # Agent.run / Agent.stream enter bind_parent_episode_id(None) for
        # fresh orchestrator runs; update_parent_episode_id replaces the
        # stored value once the gateway allocates a real episode id.  The
        # outer reset-on-exit still restores the default.
        with bind_parent_episode_id(None):
            assert get_parent_episode_id() is None
            update_parent_episode_id("ep-refreshed")
            assert get_parent_episode_id() == "ep-refreshed"
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


# ---------------------------------------------------------------------------
# Agent.run / Agent.stream refresh the parent-episode ContextVar
# ---------------------------------------------------------------------------


class TestParentEpisodeIdRefresh:
    """Regression: when the orchestrator starts a fresh episode
    (``episode_id=None``), the ``t0_agents.parent_episode_id`` tag on
    sub-agent inferences must reflect the gateway-allocated episode id
    as soon as it is known — not stay on the ``"unknown"`` fallback for
    the duration of the run.

    Covered by wrapping ``on_inference`` in :meth:`Agent.run` and
    :meth:`Agent.stream` so the first response refreshes the ContextVar
    via :func:`update_parent_episode_id`.  Flagged by Gemini on PR #9.
    """

    _EPISODE = UUID("11111111-2222-3333-4444-555555555555")

    @staticmethod
    def _orchestrator_with_sub_agent() -> Agent:
        server = FastMCP("orch")
        client: Client[Any] = Client(server)
        researcher = _build_agent("researcher")
        return _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, client),
            sub_agents=[researcher],
        )

    async def test_run_refreshes_context_var_on_first_inference(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[str | None] = []

        async def fake_run_agent(*args: Any, **kwargs: Any) -> AgentResult:
            on_inf = kwargs.get("on_inference")
            assert on_inf is not None, (
                "Agent.run must install a refresh wrapper when "
                "sub_agents are configured and episode_id is None"
            )
            await on_inf(
                FakeChatResponse(
                    content=[],
                    episode_id=TestParentEpisodeIdRefresh._EPISODE,
                ),
            )
            observed.append(get_parent_episode_id())
            return AgentResult(
                response="done",
                episode_id=str(TestParentEpisodeIdRefresh._EPISODE),
                inference_ids=["i-1"],
                turns=1,
            )

        monkeypatch.setattr("t0_agents.core.agent.run_agent", fake_run_agent)

        orch = self._orchestrator_with_sub_agent()
        result = await orch.invoke("delegate", episode_id=None)

        assert observed == [str(self._EPISODE)]
        assert result.episode_id == str(self._EPISODE)

    async def test_run_does_not_refresh_parent_episode_when_episode_id_provided(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An explicit episode_id means the initial bind already holds
        # the right value; the SubAgentMiddleware refresh wrapper must
        # not call ``update_parent_episode_id``.
        #
        # v0.12 asserted ``on_inference is None`` because nothing else
        # wired a callback when guardrails were absent.  v1.0
        # always-on ``CostTrackingMiddleware`` populates on_inference
        # regardless, so the v1.0 invariant is now behavioural:
        # invoking the callback must not mutate the parent_episode_id
        # ContextVar.
        observed: list[str | None] = []

        async def fake_run_agent(*args: Any, **kwargs: Any) -> AgentResult:
            on_inf = kwargs.get("on_inference")
            if on_inf is not None:
                await on_inf(
                    FakeChatResponse(
                        content=[],
                        episode_id=TestParentEpisodeIdRefresh._EPISODE,
                    ),
                )
            observed.append(get_parent_episode_id())
            return AgentResult(
                response="done",
                episode_id="ep-explicit",
                inference_ids=["i-1"],
                turns=1,
            )

        monkeypatch.setattr("t0_agents.core.agent.run_agent", fake_run_agent)

        orch = self._orchestrator_with_sub_agent()
        await orch.invoke("delegate", episode_id="ep-explicit")

        # The refresh wrapper was not installed, so the ContextVar
        # remained at the initial bind ("ep-explicit") even after the
        # cost-tracker observed the first inference.
        assert observed == ["ep-explicit"]

    async def test_stream_refreshes_context_var_on_first_inference(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[str | None] = []

        async def fake_stream(
            _self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            on_inf = kwargs.get("on_inference")
            assert on_inf is not None, (
                "Agent.stream must install a refresh wrapper when "
                "sub_agents are configured and episode_id is None"
            )
            await on_inf(
                FakeChatResponse(
                    content=[],
                    episode_id=TestParentEpisodeIdRefresh._EPISODE,
                ),
            )
            observed.append(get_parent_episode_id())
            yield DoneEvent(
                result=AgentResult(
                    response="done",
                    episode_id=str(TestParentEpisodeIdRefresh._EPISODE),
                    inference_ids=["i-1"],
                    turns=1,
                ),
            )

        monkeypatch.setattr(
            "t0_agents.stream.runner.StreamingAgentLoop.stream",
            fake_stream,
        )

        orch = self._orchestrator_with_sub_agent()
        events: list[AgentEvent] = []
        async for event in orch.invoke("delegate", episode_id=None).events():
            events.append(event)

        assert observed == [str(self._EPISODE)]
        assert any(isinstance(e, DoneEvent) for e in events)

    async def test_stream_does_not_refresh_parent_episode_when_episode_id_provided(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same v1.0 invariant as the non-streaming variant: the refresh
        # wrapper is not installed when episode_id is explicit, so the
        # ContextVar remains at the initial bind even after on_inference
        # observes the first response.  v1.0 ``CostTrackingMiddleware``
        # always wires on_inference, so the v0.12 ``captured == [None]``
        # invariant is replaced by the behavioural check below.
        observed: list[str | None] = []

        async def fake_stream(
            _self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            on_inf = kwargs.get("on_inference")
            if on_inf is not None:
                await on_inf(
                    FakeChatResponse(
                        content=[],
                        episode_id=TestParentEpisodeIdRefresh._EPISODE,
                    ),
                )
            observed.append(get_parent_episode_id())
            yield DoneEvent(
                result=AgentResult(
                    response="done",
                    episode_id="ep-explicit",
                    inference_ids=["i-1"],
                    turns=1,
                ),
            )

        monkeypatch.setattr(
            "t0_agents.stream.runner.StreamingAgentLoop.stream",
            fake_stream,
        )

        orch = self._orchestrator_with_sub_agent()
        async for _ in orch.invoke("delegate", episode_id="ep-explicit").events():
            pass

        assert observed == ["ep-explicit"]


# ---------------------------------------------------------------------------
# _OrchestrationToolProvider — inline sub-agent dispatch
# ---------------------------------------------------------------------------


class _StubInnerProvider:
    """Minimal ToolProvider stand-in that records call_tool invocations."""

    def __init__(self, result: CallToolResult) -> None:
        self._result = result
        self.list_calls = 0
        self.tool_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def list_tools(self) -> list[Any]:
        self.list_calls += 1
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **_: Any,
    ) -> CallToolResult:
        self.tool_calls.append((name, arguments))
        return self._result


def _ok_call_result(text: str = "inner") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        meta=None,
        data=None,
        is_error=False,
    )


class TestOrchestrationToolProvider:
    """Unit coverage for the in-process sub-agent dispatcher.

    Regression layer for the bug surfaced against PR #9 where sub-agent
    rows in ClickHouse were tagged ``parent_episode_id='unknown'``:
    FastMCP's in-process transport runs the server handler in an anyio
    sub-task with a ContextVar snapshot taken before Agent.run could
    bind anything, so the ContextVar-based handoff alone never worked
    end-to-end.  The wrapper dispatches sub-agents inline instead.
    """

    @staticmethod
    def _spy_sub_agent(name: str = "researcher") -> Agent:
        # v1.0: _OrchestrationToolProvider.call_tool dispatches via
        # ``sub_agent.invoke(msg, tags=...).result()``.  Spy the invoke
        # callable (sync) returning a Run-shaped MagicMock with an
        # async ``.result()``.
        sub = _build_agent(name)
        mock_run = MagicMock()
        mock_run.result = AsyncMock(
            return_value=AgentResult(
                response="sub-answer",
                episode_id="ep-sub",
                inference_ids=["i-sub"],
                turns=1,
            ),
        )
        sub.invoke = MagicMock(return_value=mock_run)  # type: ignore[method-assign]
        return sub

    async def test_forwards_non_subagent_calls_to_inner(self) -> None:
        inner_result = _ok_call_result("from-inner")
        inner = _StubInnerProvider(inner_result)
        sub = self._spy_sub_agent("researcher")
        provider = _OrchestrationToolProvider(
            cast(ToolProvider, inner),
            {"researcher_ask": sub},
        )

        result = await provider.call_tool(
            "add", {"a": 1, "b": 2}, timeout=5.0,
        )

        assert result is inner_result
        assert inner.tool_calls == [("add", {"a": 1, "b": 2})]
        sub.invoke.assert_not_called()

    async def test_intercepts_subagent_calls_with_live_parent_id(
        self,
    ) -> None:
        inner = _StubInnerProvider(_ok_call_result())
        sub = self._spy_sub_agent("researcher")
        provider = _OrchestrationToolProvider(
            cast(ToolProvider, inner),
            {"researcher_ask": sub},
        )

        with bind_parent_episode_id("ep-orch-123"):
            result = await provider.call_tool(
                "researcher_ask", {"message": "hi"},
            )

        # Inner was NOT touched — dispatch bypassed FastMCP.
        assert inner.tool_calls == []
        # Sub-agent saw the bound parent id, not 'unknown'.
        sub.invoke.assert_called_once_with(
            "hi", tags={PARENT_EPISODE_ID_TAG: "ep-orch-123"},
        )
        assert result.is_error is False
        assert result.content[0].text == "sub-answer"  # type: ignore[union-attr]

    async def test_falls_back_to_unknown_when_no_parent_bound(self) -> None:
        inner = _StubInnerProvider(_ok_call_result())
        sub = self._spy_sub_agent()
        provider = _OrchestrationToolProvider(
            cast(ToolProvider, inner),
            {"researcher_ask": sub},
        )

        await provider.call_tool("researcher_ask", {"message": "hi"})

        sub.invoke.assert_called_once_with(
            "hi", tags={PARENT_EPISODE_ID_TAG: "unknown"},
        )

    async def test_returns_is_error_when_message_missing(self) -> None:
        sub = self._spy_sub_agent()
        provider = _OrchestrationToolProvider(
            cast(ToolProvider, _StubInnerProvider(_ok_call_result())),
            {"researcher_ask": sub},
        )

        result = await provider.call_tool("researcher_ask", {})

        assert result.is_error is True
        assert "message" in result.content[0].text  # type: ignore[union-attr]
        sub.invoke.assert_not_called()

    async def test_wraps_non_fatal_exception_in_sub_agent_error(self) -> None:
        sub = self._spy_sub_agent("researcher")
        sub.invoke.return_value.result.side_effect = ValueError("boom")

        provider = _OrchestrationToolProvider(
            cast(ToolProvider, _StubInnerProvider(_ok_call_result())),
            {"researcher_ask": sub},
        )

        with pytest.raises(SubAgentError) as exc_info:
            await provider.call_tool("researcher_ask", {"message": "hi"})

        assert "researcher" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    async def test_reraises_fatal_error_unchanged(self) -> None:
        sub = self._spy_sub_agent("researcher")
        sub.invoke.return_value.result.side_effect = GuardrailBlockedError(
            "hard stop",
        )

        provider = _OrchestrationToolProvider(
            cast(ToolProvider, _StubInnerProvider(_ok_call_result())),
            {"researcher_ask": sub},
        )

        with pytest.raises(GuardrailBlockedError):
            await provider.call_tool("researcher_ask", {"message": "hi"})


# ---------------------------------------------------------------------------
# End-to-end regression: reproduce the user's ClickHouse spot-check bug
# ---------------------------------------------------------------------------


class TestParentEpisodeIdEndToEnd:
    """Scripted-T0 regression: an orchestrator with a real FastMCP
    client must dispatch a scripted sub-agent tool call and tag the
    sub-agent's ``run`` call with the orchestrator's allocated episode
    id — not the ``"unknown"`` fallback that surfaced in the user's
    ClickHouse query pre-fix.
    """

    _ORCH_EPISODE = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    async def test_subagent_receives_orchestrator_episode_id_in_tags(
        self,
    ) -> None:
        recorded_tags: list[dict[str, str] | None] = []

        sub = _build_agent("researcher")

        def spy_invoke(
            message: str,
            *,
            episode_id: str | None = None,
            tags: dict[str, str] | None = None,
            **_extra: Any,
        ) -> Any:
            # v1.0: _OrchestrationToolProvider dispatches via
            # ``sub.invoke(msg, tags=...).result()``.  The spy records
            # the kwargs at invoke time and returns a Run-shaped mock
            # whose ``.result()`` resolves to the stub AgentResult.
            recorded_tags.append(tags)
            mock_run = MagicMock()
            mock_run.result = AsyncMock(
                return_value=AgentResult(
                    response="research-done",
                    episode_id="ep-sub",
                    inference_ids=["i-sub-1"],
                    turns=1,
                ),
            )
            return mock_run

        sub.invoke = spy_invoke  # type: ignore[method-assign]

        tool_call = ToolCall(
            id="tc-1",
            raw_name="researcher_ask",
            name="researcher_ask",
            raw_arguments='{"message": "what is rlhf?"}',
            arguments={"message": "what is rlhf?"},
        )
        fake_t0 = FakeT0(
            [
                FakeChatResponse(
                    content=[tool_call],
                    episode_id=TestParentEpisodeIdEndToEnd._ORCH_EPISODE,
                ),
                FakeChatResponse(
                    content=[Text(type="text", text="final")],
                    episode_id=TestParentEpisodeIdEndToEnd._ORCH_EPISODE,
                ),
            ],
        )

        server = FastMCP("orch")
        client: Client[Any] = Client(server)
        orch = _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, client),
            sub_agents=[sub],
        )
        # Inject the scripted T0 — _build_agent's default FakeT0 has an
        # empty queue and would assert on the first inference.
        orch._t0 = cast(AsyncTensorZeroGateway, fake_t0)

        async with client:
            result = await orch.invoke("go", episode_id=None)

        assert result.response == "final"
        assert result.episode_id == str(self._ORCH_EPISODE)
        assert recorded_tags == [
            {PARENT_EPISODE_ID_TAG: str(self._ORCH_EPISODE)},
        ]
