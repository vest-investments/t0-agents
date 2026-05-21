"""Unit tests for ``t0_agents.delegation.orchestrator``.

Covers the surfaces that survived the SubAgentToolProvider refactor:

- ``bind_parent_episode_id`` / ``get_parent_episode_id`` /
  ``update_parent_episode_id`` — ContextVar semantics (set, nest,
  reset, None default, in-place refresh).
- ``Agent.__init__`` integration — when ``config.sub_agents`` is set,
  ``agent._tool_provider`` is auto-allocated as a
  :class:`SubAgentToolProvider` wrapping any user-supplied inner.
- ``SubAgentMiddleware`` parent-episode refresh — when the orchestrator
  starts a fresh episode, the first inference's response refreshes the
  ContextVar so sub-agent dispatch tags the orchestrator's id.

Tool-call dispatch behaviour (synthesis + inline ``call_tool``) is
covered exhaustively by :mod:`tests.test_delegation.test_provider`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.delegation.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    bind_parent_episode_id,
    get_parent_episode_id,
    update_parent_episode_id,
)
from t0_agents.delegation.provider import SubAgentToolProvider
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
# Agent.__init__ auto-allocates SubAgentToolProvider
# ---------------------------------------------------------------------------


class TestSubAgentAutoAllocation:
    """Construction-time wiring of the new provider on the orchestrator."""

    def test_provider_synthesises_one_tool_per_sub_agent(self) -> None:
        researcher = _build_agent("researcher")
        writer = _build_agent("writer")
        orchestrator = _build_agent(
            "orch",
            sub_agents=[researcher, writer],
        )

        provider = orchestrator._tool_provider
        assert isinstance(provider, SubAgentToolProvider)
        assert set(provider._by_tool_name) == {"researcher_ask", "writer_ask"}

    def test_provider_carries_inner_when_caller_passes_tool_provider(
        self,
    ) -> None:
        # When the orchestrator also wants real MCP tools, the
        # caller-supplied tool_provider becomes the provider's ``inner``
        # so ``list_tools()`` composes both surfaces and ``call_tool``
        # delegates unknown names to the inner.
        researcher = _build_agent("researcher")
        inner = FakeMCPClient()
        orchestrator = _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, inner),
            sub_agents=[researcher],
        )

        provider = orchestrator._tool_provider
        assert isinstance(provider, SubAgentToolProvider)
        assert provider._inner is inner

    def test_no_wrap_when_sub_agents_empty(self) -> None:
        # Empty sub-agent list pays no cost: tool_provider is the bare
        # one the caller supplied (no SubAgentToolProvider in front).
        inner = FakeMCPClient()
        agent = _build_agent("plain", tool_provider=cast(ToolProvider, inner))

        assert agent._tool_provider is inner

    def test_duplicate_sub_agent_names_raise(self) -> None:
        # assert_unique_names runs inside SubAgentToolProvider.__init__,
        # so the duplicate is caught at orchestrator construction time.
        a = _build_agent("researcher")
        b = _build_agent("researcher")
        with pytest.raises(ConfigurationError, match="Duplicate sub-agent names"):
            _build_agent("orch", sub_agents=[a, b])

    def test_sub_agents_field_rejects_non_agent_entries(self) -> None:
        # AgentConfig.__post_init__ catches this before Agent.__init__,
        # so SubAgentToolProvider never sees a non-Agent value.
        with pytest.raises(
            ConfigurationError, match=r"sub_agents\[0\] must be an Agent",
        ):
            AgentConfig(
                name="orch",
                function_name="orch_fn",
                sub_agents=[cast(Agent, "not-an-agent")],
            )

    def test_inner_can_be_arbitrary_tool_provider(self) -> None:
        # No fastmcp.Client constraint anymore: any object satisfying
        # the ToolProvider protocol (FakeMCPClient here, real Client in
        # other tests) is accepted as the inner provider.
        researcher = _build_agent("researcher")
        for inner in (FakeMCPClient(), Client(FastMCP("just-a-server"))):
            orchestrator = _build_agent(
                "orch",
                tool_provider=cast(ToolProvider, inner),
                sub_agents=[researcher],
            )
            provider = orchestrator._tool_provider
            assert isinstance(provider, SubAgentToolProvider)
            assert provider._inner is inner


# ---------------------------------------------------------------------------
# SubAgentMiddleware refreshes the parent-episode ContextVar
# ---------------------------------------------------------------------------


class TestParentEpisodeIdRefresh:
    """Regression: when the orchestrator starts a fresh episode
    (``episode_id=None``), the ``t0_agents.parent_episode_id`` tag on
    sub-agent inferences must reflect the gateway-allocated episode id
    as soon as it is known — not stay on the ``"unknown"`` fallback for
    the duration of the run.

    Covered by wrapping ``on_inference`` in
    :class:`SubAgentMiddleware` so the first response refreshes the
    ContextVar via :func:`update_parent_episode_id`.  Originally
    flagged by Gemini on PR #9.
    """

    _EPISODE = UUID("11111111-2222-3333-4444-555555555555")

    @staticmethod
    def _orchestrator_with_sub_agent() -> Agent:
        researcher = _build_agent("researcher")
        return _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, FakeMCPClient()),
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
# End-to-end regression: reproduce the user's ClickHouse spot-check bug
# ---------------------------------------------------------------------------


class TestParentEpisodeIdEndToEnd:
    """Scripted-T0 regression: an orchestrator must dispatch a scripted
    sub-agent tool call and tag the sub-agent's ``run`` call with the
    orchestrator's allocated episode id — not the ``"unknown"`` fallback
    that surfaced in the user's ClickHouse query pre-fix.

    The orchestrator no longer needs a real ``fastmcp.Client``: the
    SubAgentToolProvider synthesises the delegation tool surface and
    dispatches inline.  This test exercises the loop end-to-end against
    a scripted gateway with a bare ``FakeMCPClient`` as inner.
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

        orch = _build_agent(
            "orch",
            tool_provider=cast(ToolProvider, FakeMCPClient()),
            sub_agents=[sub],
        )
        # Inject the scripted T0 — _build_agent's default FakeT0 has an
        # empty queue and would assert on the first inference.
        orch._t0 = cast(AsyncTensorZeroGateway, fake_t0)

        result = await orch.invoke("go", episode_id=None)

        assert result.response == "final"
        assert result.episode_id == str(self._ORCH_EPISODE)
        assert recorded_tags == [
            {PARENT_EPISODE_ID_TAG: str(self._ORCH_EPISODE)},
        ]
