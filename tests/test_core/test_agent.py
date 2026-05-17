"""Unit tests for Agent and AgentConfig.

Tests configuration validation, field forwarding to ``run_agent``, and
error propagation. Uses hand-written fakes (``FakeT0``, ``FakeMCPClient``)
to stay mypy-strict-clean.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

# ---------------------------------------------------------------------------
# Fixture builders (local to this test file)
# ---------------------------------------------------------------------------


def _mcp_tool(name: str = "add") -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def _tool_call(
    *,
    name: str | None = "add",
    raw_name: str = "add",
    call_id: str = "call-1",
    arguments: dict[str, Any] | None = None,
    raw_arguments: str = '{"a": 2, "b": 3}',
) -> ToolCall:
    args = arguments if arguments is not None else {"a": 2, "b": 3}
    return ToolCall(
        id=call_id,
        raw_arguments=raw_arguments,
        raw_name=raw_name,
        arguments=args,
        name=name,
    )


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    client: FakeMCPClient | None = None,
    **config_overrides: Any,
) -> Agent:
    """Build an Agent with sensible defaults for testing."""
    defaults: dict[str, Any] = {"name": "test", "function_name": "agent"}
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([])),
        tool_provider=cast(ToolProvider, client or FakeMCPClient()),
    )


# ---------------------------------------------------------------------------
# TestAgentConfig — validation
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_valid_minimal_config(self) -> None:
        cfg = AgentConfig(name="a", function_name="f")
        assert cfg.name == "a"
        assert cfg.function_name == "f"
        assert cfg.max_turns == 20
        assert cfg.tool_timeout == 30.0
        assert cfg.parallel_tools is False
        assert cfg.dryrun is False
        assert cfg.tool_choice is None
        assert cfg.tags is None
        assert cfg.namespace is None

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="name"):
            AgentConfig(name="", function_name="f")

    def test_empty_function_name_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="function_name"):
            AgentConfig(name="a", function_name="")

    def test_negative_max_turns_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_turns"):
            AgentConfig(name="a", function_name="f", max_turns=-1)

    def test_zero_max_turns_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_turns"):
            AgentConfig(name="a", function_name="f", max_turns=0)

    def test_negative_tool_timeout_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="tool_timeout"):
            AgentConfig(name="a", function_name="f", tool_timeout=-5.0)

    def test_zero_tool_timeout_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="tool_timeout"):
            AgentConfig(name="a", function_name="f", tool_timeout=0.0)

    def test_none_tool_timeout_accepted(self) -> None:
        cfg = AgentConfig(name="a", function_name="f", tool_timeout=None)
        assert cfg.tool_timeout is None


# ---------------------------------------------------------------------------
# TestAgent — construction and field forwarding
# ---------------------------------------------------------------------------


class TestAgent:
    def test_config_property(self) -> None:
        agent = _make_agent()
        assert agent.config.name == "test"
        assert agent.config.function_name == "agent"

    def test_invalid_tool_provider_raises(self) -> None:
        config = AgentConfig(name="a", function_name="f")
        with pytest.raises(ConfigurationError, match="ToolProvider"):
            Agent(
                config,
                t0=cast(AsyncTensorZeroGateway, FakeT0([])),
                tool_provider=cast(ToolProvider, "not a provider"),
            )

    def test_chat_agent_without_tool_provider_raises(self) -> None:
        """Chat-mode (the default ``function_kind``) requires a provider.

        The constructor must reject the omission eagerly with a message
        that names ``output='chat'`` and points at the single-shot
        escape hatch — otherwise the failure surfaces deep inside
        ``run_agent`` on the first ``Agent.run`` call.
        """
        config = AgentConfig(name="a", function_name="f")
        with pytest.raises(ConfigurationError, match="output='chat'"):
            Agent(
                config,
                t0=cast(AsyncTensorZeroGateway, FakeT0([])),
            )

    def test_json_agent_without_tool_provider_constructs(self) -> None:
        """Single-shot agents may omit the provider entirely (v1.0)."""
        from t0_agents.core.outputs import JsonOutput

        config = AgentConfig(
            name="extractor",
            function_name="extract",
            output=JsonOutput(json_mode="strict"),
        )
        agent = Agent(
            config,
            t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        )
        assert agent.config.output.name == "json"
        # Internal attribute is exposed for the serve/share_client check;
        # documenting the contract here keeps that coupling deliberate.
        assert agent._tool_provider is None

    async def test_defaults_forwarded(self) -> None:
        """Minimal config → run_agent called with expected default kwargs."""
        episode = uuid4()
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")], episode_id=episode)])
        client = FakeMCPClient()
        agent = _make_agent(t0=t0, client=client)

        result = await agent.invoke("hello")

        assert result.response == "hi"
        assert result.episode_id == str(episode)
        call = t0.calls[0]
        assert call["function_name"] == "agent"
        assert call["dryrun"] is False
        assert call["tool_choice"] is None
        assert call["tags"] is None
        assert call["namespace"] is None

    async def test_forwards_every_config_field(self) -> None:
        """Every AgentConfig field reaches the underlying t0.inference call."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        client = FakeMCPClient()
        agent = _make_agent(
            t0=t0,
            client=client,
            dryrun=True,
            tool_choice="required",
            tags={"env": "test"},
            namespace="staging",
        )
        await agent.invoke("go")

        call = t0.calls[0]
        assert call["dryrun"] is True
        assert call["tool_choice"] == "required"
        assert call["tags"] == {"env": "test"}
        assert call["namespace"] == "staging"

    async def test_episode_id_plumbing(self) -> None:
        """episode_id from ``agent.invoke()`` reaches t0.inference on first call."""
        ep = str(uuid4())
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)

        await agent.invoke("go", episode_id=ep)
        assert t0.calls[0]["episode_id"] == ep

    async def test_max_turns_exceeded_propagates(self) -> None:
        """MaxTurnsExceededError from run_agent bubbles through ``agent.invoke()``."""
        from fastmcp.client.client import CallToolResult

        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call(call_id="c1")]),
            FakeChatResponse(content=[_tool_call(call_id="c2")]),
        ])
        client = FakeMCPClient(
            call_results={
                "add": CallToolResult(
                    content=[], structured_content=None, meta=None, data=5
                ),
            },
        )
        agent = _make_agent(t0=t0, client=client, max_turns=2)

        with pytest.raises(MaxTurnsExceededError) as exc_info:
            await agent.invoke("loop forever")
        assert exc_info.value.max_turns == 2

    async def test_max_turns_exceeded_is_t0_agents_error(self) -> None:
        """MaxTurnsExceededError is a subclass of T0AgentsError."""
        assert issubclass(MaxTurnsExceededError, T0AgentsError)

    async def test_tags_forwarded(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0, tags={"k": "v"})
        await agent.invoke("go")
        assert t0.calls[0]["tags"] == {"k": "v"}

    async def test_run_tags_override_merges_with_config(self) -> None:
        """Per-call tags shallow-merge on top of AgentConfig.tags."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0, tags={"env": "prod", "app": "api"})
        await agent.invoke("go", tags={"request_id": "r-1"})
        assert t0.calls[0]["tags"] == {
            "env": "prod",
            "app": "api",
            "request_id": "r-1",
        }

    async def test_run_tags_override_wins_on_collision(self) -> None:
        """Override values replace base config values on matching keys."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0, tags={"env": "prod"})
        await agent.invoke("go", tags={"env": "staging"})
        assert t0.calls[0]["tags"] == {"env": "staging"}

    async def test_run_tags_none_preserves_config_only(self) -> None:
        """Default tags=None leaves config.tags untouched (byte-identity)."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0, tags={"env": "prod"})
        await agent.invoke("go")
        assert t0.calls[0]["tags"] == {"env": "prod"}

    async def test_run_tags_empty_config_and_override_forwards_none(self) -> None:
        """No config.tags and no override → tags=None reaches run_agent."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)  # no tags in config
        await agent.invoke("go")
        assert t0.calls[0]["tags"] is None

    async def test_namespace_forwarded(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0, namespace="prod")
        await agent.invoke("go")
        assert t0.calls[0]["namespace"] == "prod"

    async def test_run_system_arguments_forwarded(self) -> None:
        """system_arguments reaches t0.inference as input.system directly.

        Per the TensorZero gateway API reference, ``input.system`` IS the
        schema-described arguments dict — no ``arguments`` wrapper.
        """
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)
        await agent.invoke("go", system_arguments={"max_turns": 5})
        assert t0.calls[0]["input"]["system"] == {"max_turns": 5}

    async def test_run_system_arguments_omitted_no_op(self) -> None:
        """Default system_arguments=None leaves the input.system key absent."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)
        await agent.invoke("go")
        assert "system" not in t0.calls[0]["input"]

    async def test_run_system_arguments_empty_dict_no_op(self) -> None:
        """Empty system_arguments={} collapses to no `system` key (guard)."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)
        await agent.invoke("go", system_arguments={})
        assert "system" not in t0.calls[0]["input"]

    async def test_run_system_string_forwarded(self) -> None:
        """``system="..."`` reaches t0.inference as a raw string in input.system."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(t0=t0)
        await agent.invoke("go", system="Reply only in haiku.")
        assert t0.calls[0]["input"]["system"] == "Reply only in haiku."

    async def test_run_system_and_system_arguments_rejected(self) -> None:
        """Caller passing both kwargs is rejected up front with ``ValueError``."""
        t0 = FakeT0([])
        agent = _make_agent(t0=t0)
        with pytest.raises(ValueError, match="Pass either `system`"):
            await agent.invoke(
                "go", system="raw", system_arguments={"foo": "bar"},
            )
        assert t0.calls == []

    async def test_run_system_with_system_template_strategy_rejected(self) -> None:
        """``system=`` is incompatible with ``memory_strategy='system_template'``."""
        t0 = FakeT0([])
        agent = _make_agent(t0=t0, memory_strategy="system_template")
        with pytest.raises(
            ValueError, match="memory_strategy='system_template'"
        ):
            await agent.invoke("go", system="raw")
        assert t0.calls == []

    async def test_run_system_with_prepended_message_strategy_ok(self) -> None:
        """``memory_strategy='prepended_message'`` leaves ``input.system`` alone,
        so ``system=`` is allowed."""
        t0 = FakeT0([FakeChatResponse(content=[Text(text="ok")])])
        agent = _make_agent(
            t0=t0, memory_strategy="prepended_message",
        )
        await agent.invoke("go", system="raw prompt")
        assert t0.calls[0]["input"]["system"] == "raw prompt"

    async def test_none_timeout_uses_default(self) -> None:
        """tool_timeout=None on config → 30.0 passed to run_agent."""
        from fastmcp.client.client import CallToolResult

        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call()]),
            FakeChatResponse(content=[Text(text="ok")]),
        ])
        client = FakeMCPClient(
            call_results={
                "add": CallToolResult(
                    content=[], structured_content=None, meta=None, data=5
                ),
            },
        )
        agent = _make_agent(t0=t0, client=client, tool_timeout=None)
        await agent.invoke("go")
        # FakeMCPClient records (name, args, timeout) per call_tool invocation.
        assert client.call_tool_invocations[0][2] == 30.0

    async def test_custom_timeout_forwarded(self) -> None:
        """Explicit tool_timeout on config reaches call_tool."""
        from fastmcp.client.client import CallToolResult

        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call()]),
            FakeChatResponse(content=[Text(text="ok")]),
        ])
        client = FakeMCPClient(
            call_results={
                "add": CallToolResult(
                    content=[], structured_content=None, meta=None, data=5
                ),
            },
        )
        agent = _make_agent(t0=t0, client=client, tool_timeout=99.0)
        await agent.invoke("go")
        assert client.call_tool_invocations[0][2] == 99.0

    async def test_concurrent_runs_serial(self) -> None:
        """Two sequential ``invoke()`` calls produce independent results."""
        ep1 = uuid4()
        ep2 = uuid4()
        t0 = FakeT0([
            FakeChatResponse(content=[Text(text="first")], episode_id=ep1),
            FakeChatResponse(content=[Text(text="second")], episode_id=ep2),
        ])
        agent = _make_agent(t0=t0)

        r1 = await agent.invoke("one")
        r2 = await agent.invoke("two")

        assert r1.response == "first"
        assert r2.response == "second"
        assert r1.episode_id != r2.episode_id
