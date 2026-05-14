"""Unit tests for the M4 safety surfaces on ``Agent``.

Covers four concerns:

- The runtime check for ``json_mode="tool"`` against a tool provider
  that advertises tools (raised lazily on the first ``Agent.run``).
- The ``Agent.stream`` guard for JSON functions (raised at first
  ``__anext__`` before any event is yielded).
- The once-per-instance Anthropic extended-thinking compatibility
  warning (TZ guide §11.4).
- The ``after_run`` exception-handler realignment: fatal hooks (e.g.
  :func:`require_parsed`) must propagate past the broad ``except
  Exception`` belt-and-braces trap so policy aborts surface to the
  caller, matching the documented contract at ``docs/guardrails.md``.

Local ``FakeJsonT0`` / ``_fake_json_response`` helpers mirror M3 — M5
will consolidate the JSON fakes into ``MockGateway``.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from mcp.types import Tool as MCPTool
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    Text,
    Usage,
)
from tensorzero.types import JsonInferenceOutput

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
)
from t0_agents.core.types import ToolProvider
from t0_agents.stream.events import DoneEvent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, chat_chunk, final_chunk, text_chunk

# ---------------------------------------------------------------------------
# JSON fakes — local copy, will be folded into MockGateway in M5.
# ---------------------------------------------------------------------------


def _fake_json_response(
    *,
    raw: str | None,
    parsed: dict[str, Any] | None,
    inference_id: UUID | None = None,
    episode_id: UUID | None = None,
) -> JsonInferenceResponse:
    return JsonInferenceResponse(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name="test-variant",
        output=JsonInferenceOutput(raw=raw, parsed=parsed),
        usage=Usage(input_tokens=10, output_tokens=20),
    )


class FakeJsonT0:
    """Scriptable async gateway — pops one response per ``inference()`` call."""

    def __init__(
        self, responses: list[JsonInferenceResponse | Exception]
    ) -> None:
        self._queue: list[JsonInferenceResponse | Exception] = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> JsonInferenceResponse:
        self.calls.append(dict(kwargs))
        if not self._queue:
            raise AssertionError("FakeJsonT0 ran out of canned responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _CountingFakeMCPClient(FakeMCPClient):
    """``FakeMCPClient`` that records every ``list_tools()`` call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.list_tools_calls = 0

    async def list_tools(self) -> list[MCPTool]:
        self.list_tools_calls += 1
        return await super().list_tools()


def _json_agent(
    *,
    t0: FakeJsonT0,
    tool_provider: ToolProvider | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "extractor",
        "function_name": "extract_customer_info",
        "function_kind": "json",
        "json_mode": "strict",
    }
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    provider = tool_provider or cast(ToolProvider, FakeMCPClient(tools=[]))
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=provider,
    )


# ---------------------------------------------------------------------------
# Tool-mode + tools incompatibility
# ---------------------------------------------------------------------------


class TestToolModeCheck:
    async def test_tool_mode_with_tools_errors(self) -> None:
        """``json_mode='tool'`` + non-empty tool provider → ConfigurationError."""
        t0 = FakeJsonT0([_fake_json_response(raw='{"a": 1}', parsed={"a": 1})])
        client = FakeMCPClient(
            tools=[
                MCPTool(
                    name="add",
                    description="add tool",
                    inputSchema={"type": "object", "properties": {}},
                )
            ],
        )
        agent = _json_agent(
            t0=t0,
            tool_provider=cast(ToolProvider, client),
            json_mode="tool",
        )

        with pytest.raises(ConfigurationError, match="§11.1"):
            await agent.run("go")

    async def test_tool_mode_with_empty_tools_works(self) -> None:
        """``json_mode='tool'`` + empty tool provider → run succeeds."""
        t0 = FakeJsonT0([_fake_json_response(raw='{"a": 1}', parsed={"a": 1})])
        client = FakeMCPClient(tools=[])
        agent = _json_agent(
            t0=t0,
            tool_provider=cast(ToolProvider, client),
            json_mode="tool",
        )

        result = await agent.run("go")
        assert result.parsed == {"a": 1}
        assert result.function_kind == "json"

    async def test_tool_mode_check_caches_after_first_run(self) -> None:
        """The list_tools() round-trip happens exactly once across runs."""
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw='{"a": 1}', parsed={"a": 1}),
                _fake_json_response(raw='{"a": 2}', parsed={"a": 2}),
            ]
        )
        client = _CountingFakeMCPClient(tools=[])
        agent = _json_agent(
            t0=t0,
            tool_provider=cast(ToolProvider, client),
            json_mode="tool",
        )

        await agent.run("first")
        await agent.run("second")

        assert client.list_tools_calls == 1


# ---------------------------------------------------------------------------
# Streaming guard
# ---------------------------------------------------------------------------


class TestStreamGuard:
    async def test_stream_on_json_agent_errors(self) -> None:
        """``Agent.stream`` on a JSON agent raises at first __anext__."""
        t0 = FakeJsonT0([_fake_json_response(raw='{"a": 1}', parsed={"a": 1})])
        agent = _json_agent(t0=t0)
        gen = agent.stream("go")

        with pytest.raises(ConfigurationError, match="function_kind='json'"):
            await gen.__anext__()

    async def test_stream_on_chat_agent_byte_identical(self) -> None:
        """``Agent.stream`` on a chat agent still produces a DoneEvent."""
        fake_t0 = FakeT0(
            responses=[
                [
                    chat_chunk([text_chunk("hello", block_id="t1")]),
                    final_chunk(),
                ]
            ]
        )
        client = FakeMCPClient(tools=[])
        agent = Agent(
            AgentConfig(name="a", function_name="agent"),
            t0=cast(AsyncTensorZeroGateway, fake_t0),
            tool_provider=cast(ToolProvider, client),
        )

        events = [ev async for ev in agent.stream("hi")]

        assert isinstance(events[-1], DoneEvent)
        assert events[-1].result.response == "hello"


# ---------------------------------------------------------------------------
# Anthropic warning
# ---------------------------------------------------------------------------


class TestAnthropicWarning:
    async def test_warning_fires_once_for_on(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``json_mode='on'`` emits exactly one warning across two runs."""
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw='{"a": 1}', parsed={"a": 1}),
                _fake_json_response(raw='{"a": 2}', parsed={"a": 2}),
            ]
        )
        agent = _json_agent(t0=t0, json_mode="on")

        with caplog.at_level(logging.WARNING, logger="t0_agents.core.agent"):
            await agent.run("first")
            await agent.run("second")

        warnings = [
            rec
            for rec in caplog.records
            if "Anthropic extended-thinking" in rec.getMessage()
        ]
        assert len(warnings) == 1

    async def test_warning_fires_once_for_tool(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``json_mode='tool'`` (with an empty provider) also warns once."""
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw='{"a": 1}', parsed={"a": 1}),
                _fake_json_response(raw='{"a": 2}', parsed={"a": 2}),
            ]
        )
        agent = _json_agent(
            t0=t0,
            tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
            json_mode="tool",
        )

        with caplog.at_level(logging.WARNING, logger="t0_agents.core.agent"):
            await agent.run("first")
            await agent.run("second")

        warnings = [
            rec
            for rec in caplog.records
            if "Anthropic extended-thinking" in rec.getMessage()
        ]
        assert len(warnings) == 1

    async def test_warning_not_emitted_for_strict(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``json_mode='strict'`` is the safe path — no warning."""
        t0 = FakeJsonT0([_fake_json_response(raw='{"a": 1}', parsed={"a": 1})])
        agent = _json_agent(t0=t0, json_mode="strict")

        with caplog.at_level(logging.WARNING, logger="t0_agents.core.agent"):
            await agent.run("go")

        warnings = [
            rec
            for rec in caplog.records
            if "Anthropic extended-thinking" in rec.getMessage()
        ]
        assert warnings == []


# ---------------------------------------------------------------------------
# after_run fatal-error propagation (pre-existing bug fix shipped with M4)
# ---------------------------------------------------------------------------


class _AfterRunFatalGuardrail:
    """Test double whose ``after_run`` always raises GuardrailBlockedError."""

    name = "after_run_fatal"

    async def before_run(self, ctx: Any) -> None:
        return None

    async def before_tool(self, ctx: Any, tool_call: Any) -> str | None:
        return None

    async def after_tool(
        self, ctx: Any, tool_call: Any, result_str: str
    ) -> None:
        return None

    async def after_run(self, ctx: Any, result: Any) -> None:
        raise GuardrailBlockedError("after_run hard abort")


class TestAfterRunFatalPropagation:
    async def test_propagates_on_chat_run(self) -> None:
        """``after_run`` raising ``GuardrailBlockedError`` aborts a chat run."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")])])
        client = FakeMCPClient(tools=[])
        agent = Agent(
            AgentConfig(
                name="a",
                function_name="agent",
                guardrails=[cast(Any, _AfterRunFatalGuardrail())],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, client),
        )

        with pytest.raises(GuardrailBlockedError, match="after_run hard abort"):
            await agent.run("go")

    async def test_propagates_on_json_run(self) -> None:
        """``after_run`` raising ``GuardrailBlockedError`` aborts a JSON run."""
        t0 = FakeJsonT0([_fake_json_response(raw='{"a": 1}', parsed={"a": 1})])
        agent = _json_agent(
            t0=t0,
            guardrails=[cast(Any, _AfterRunFatalGuardrail())],
        )

        with pytest.raises(GuardrailBlockedError, match="after_run hard abort"):
            await agent.run("go")
