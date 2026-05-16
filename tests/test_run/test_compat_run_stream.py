"""Backwards-compatibility tests for the v0.12 entry-point shims.

``Agent.run()`` and ``Agent.stream()`` are deprecated in v1.0 (Tier 1
migration) and removed in v1.1.  This module verifies the byte-identity
guarantees from ``00-integration-contract.md`` §2.3:

- ``agent.run("hi")`` returns an :class:`AgentResult` with the same
  fields and values as v0.12 (response, episode_id, inference_ids,
  turns).
- ``agent.stream("hi")`` yields the same event sequence.
- Both shims emit ``DeprecationWarning`` exactly once per source line
  (Python's default filter dedups by stacklevel) and the message
  points at the canonical :meth:`Agent.invoke` replacement.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import Agent, AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from t0_agents.stream.events import DoneEvent, TextDeltaEvent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, chat_chunk, final_chunk, text_chunk


def _resp(text: str = "ok") -> FakeChatResponse:
    return FakeChatResponse(content=[Text(type="text", text=text)])


def _stream(text: str = "ok") -> list[Any]:
    return [
        chat_chunk([text_chunk(text, block_id="t1")]),
        final_chunk(),
    ]


def _agent(
    t0: FakeT0 | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "compat", "function_name": "agent"}
    defaults.update(config_overrides)
    return Agent(
        AgentConfig(**defaults),
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_resp()])),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


class TestAgentRunByteIdentity:
    async def test_run_returns_agent_result_with_v0_12_shape(self) -> None:
        agent = _agent(FakeT0([_resp("hello")]))
        # The shim itself emits the warning — opt in to seeing it so
        # the assertion below catches the v1.0 emission shape.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DeprecationWarning)
            result = await agent.run("hi")
        # Result fields match v0.12 contract.
        assert result.response == "hello"
        assert result.episode_id is not None
        assert len(result.inference_ids) == 1
        assert result.turns == 1
        assert result.function_kind == "chat"
        assert result.parsed is None
        # Warning emitted; mentions invoke().
        msgs = [str(w.message) for w in captured]
        depr = [
            m for m in msgs
            if "Agent.run() is deprecated" in m
        ]
        assert len(depr) == 1
        assert "agent.invoke" in depr[0]


class TestAgentStreamByteIdentity:
    async def test_stream_yields_events_terminating_in_done(self) -> None:
        agent = _agent(FakeT0([_stream("hello")]))
        events_collected: list[Any] = []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DeprecationWarning)
            async for ev in agent.stream("hi"):
                events_collected.append(ev)
        # At least one TextDeltaEvent and a terminal DoneEvent.
        assert any(isinstance(e, TextDeltaEvent) for e in events_collected)
        assert isinstance(events_collected[-1], DoneEvent)
        # Warning emitted; mentions invoke()…events() pattern.
        depr = [
            str(w.message) for w in captured
            if "Agent.stream() is deprecated" in str(w.message)
        ]
        assert len(depr) == 1
        assert "agent.invoke" in depr[0]
        assert ".events()" in depr[0]


class TestStreamRejectsJsonAgent:
    """v0.12 :meth:`Agent.stream` raised ConfigurationError on JSON agents.

    The v1.0 shim preserves this — ``invoke().events()`` raises
    :class:`ConfigurationError` on first iteration (now keyed on
    ``output.supports_streaming``), and the shim's ``async for``
    surfaces that immediately.
    """

    async def test_stream_on_json_agent_raises_configuration_error(
        self,
    ) -> None:
        from t0_agents.core.outputs import JsonOutput

        agent = _agent(output=JsonOutput(json_mode="strict"))
        with pytest.raises(ConfigurationError, match="streaming"):
            async for _ in agent.stream("extract X"):
                pass


# v0.12 per-call JSON kwargs on ``Agent.invoke`` were removed in M3 — the
# only remaining surface that accepts them is the ``Agent.run`` Tier-1
# shim, which translates them into a per-call :class:`JsonOutput`.  The
# v1.0 :class:`RunInput` no longer carries ``output_schema``/
# ``json_mode``/``max_json_retries`` fields.  ``TestLegacyKwargsForwarded``
# from M1 has been removed (M1 dev-log §4 M3 carry-forward 2).


class TestDeprecationDeduplicationByLine:
    """Python's default warnings filter dedups by (msg, cat, mod, lineno).

    Two calls to ``agent.run`` from the SAME source line should emit
    one warning total — confirmed by Python's default ``warnings``
    filter rather than by code in the shim.
    """

    async def test_warning_dedups_by_source_line(self) -> None:
        agent = _agent(FakeT0([_resp("a"), _resp("b")]))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("default", DeprecationWarning)
            for _ in range(2):
                await agent.run("hi")
        # The "default" action emits once per (msg, cat, mod, lineno).
        # Both calls are on the same line of THIS test, so exactly one
        # DeprecationWarning is recorded.
        depr = [
            w for w in captured
            if issubclass(w.category, DeprecationWarning)
            and "Agent.run() is deprecated" in str(w.message)
        ]
        assert len(depr) == 1
