"""Unit tests for ``Agent.run`` JSON-function dispatch.

Drives ``Agent.run`` against ``MockGateway.script_json`` for the JSON path
and the shared ``tests/_fake_t0`` for the chat-byte-identity assertions.
``MockGateway.script_json`` is the JSON-flavoured sibling of
``MockGateway.script`` — see ``tests/test_testing/test_json_helpers.py``
for the dict-shape semantic (auto-derive ``parsed`` when the key is
absent; honour explicit ``parsed`` when present).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from t0_agents.memory.in_memory import InMemoryMemory
from t0_agents.testing import MockGateway
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

# Default raw payload used across the happy-path tests below.
_OK_RAW = '{"name": "Sarah", "email": "sarah@example.com"}'


def _json_agent(
    *,
    t0: MockGateway,
    client: FakeMCPClient | None = None,
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
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client or FakeMCPClient()),
    )


def _chat_agent(
    *,
    t0: FakeT0,
    client: FakeMCPClient | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "chat", "function_name": "agent"}
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client or FakeMCPClient()),
    )


def _mcp_tool(name: str = "add") -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonDispatch:
    async def test_run_dispatches_to_json_when_function_kind_is_json(self) -> None:
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(t0=t0)

        result = await agent.run("Hi, I'm Sarah at sarah@example.com")

        assert result.function_kind == "json"
        assert result.parsed == {
            "name": "Sarah",
            "email": "sarah@example.com",
        }
        assert result.response == _OK_RAW
        assert result.turns == 1
        assert len(result.inference_ids) == 1
        # Chat-only field default still preserved.
        assert result.response_source == "text"

    async def test_json_agent_runs_without_tool_provider(self) -> None:
        """End-to-end: a JSON agent built without a provider can still run.

        Mirrors the canonical ``examples/17_structured/structured_agent.py``
        post-0.12.0 shape — no ``fastmcp.Client``, no noop server, just
        ``Agent(config, t0=t0)``.
        """
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = Agent(
            AgentConfig(
                name="extractor",
                function_name="extract_customer_info",
                function_kind="json",
                json_mode="strict",
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
        )

        result = await agent.run("Hi, I'm Sarah at sarah@example.com")

        assert result.function_kind == "json"
        assert result.parsed == {
            "name": "Sarah",
            "email": "sarah@example.com",
        }
        assert result.response == _OK_RAW
        assert result.turns == 1

    async def test_chat_path_is_byte_identical(self) -> None:
        # Same setup as a minimal chat test: the new fields default and
        # nothing else changes for a chat-flavoured agent.
        t0 = FakeT0(
            [FakeChatResponse(content=[Text(text="hello", type="text")])],
        )
        agent = _chat_agent(t0=t0)

        result = await agent.run("hi")

        assert result.function_kind == "chat"
        assert result.parsed is None
        assert result.response == "hello"
        assert result.response_source == "text"


class TestPerCallOverrides:
    async def test_per_call_output_schema_override(self) -> None:
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(
            t0=t0,
            output_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        )

        per_call_schema = {
            "type": "object",
            "properties": {"y": {"type": "string"}},
        }
        await agent.run("hello", output_schema=per_call_schema)

        assert len(t0.calls) == 1
        kwargs = t0.calls[0]
        # Per-call override beats the config-level schema.
        assert kwargs["output_schema"] == per_call_schema

    async def test_per_call_json_mode_override(self) -> None:
        # ``json_mode="tool"`` requires an empty tool provider per the
        # JSON safety surface (TZ guide §11.1).  The test's intent is
        # to confirm the per-call override is threaded into ``params``,
        # not to exercise the (forbidden) tool-mode + advertised-tools
        # combo.
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(
            t0=t0,
            json_mode="strict",
            client=FakeMCPClient(tools=[]),
        )

        await agent.run("hello", json_mode="tool")

        kwargs = t0.calls[0]
        assert kwargs["params"] == {"chat_completion": {"json_mode": "tool"}}

    async def test_config_schema_used_when_per_call_omits_override(self) -> None:
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        config_schema = {"type": "object", "properties": {"z": {"type": "boolean"}}}
        agent = _json_agent(t0=t0, output_schema=config_schema)

        await agent.run("hello")

        kwargs = t0.calls[0]
        assert kwargs["output_schema"] == config_schema

    async def test_per_call_kwargs_forbidden_on_chat_agent(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi", type="text")])])
        agent = _chat_agent(t0=t0)

        with pytest.raises(ConfigurationError, match="function_kind"):
            await agent.run("hi", output_schema={"type": "object"})

    async def test_per_call_json_mode_kwarg_forbidden_on_chat_agent(self) -> None:
        t0 = FakeT0([FakeChatResponse(content=[Text(text="hi", type="text")])])
        agent = _chat_agent(t0=t0)

        with pytest.raises(ConfigurationError, match="function_kind"):
            await agent.run("hi", json_mode="strict")


class TestGuardrails:
    async def test_before_run_after_run_fire_on_json(self) -> None:
        class _SpyGuardrail:
            name = "spy"

            def __init__(self) -> None:
                self.before_run_calls = 0
                self.after_run_calls = 0
                self.last_result: Any = "uninitialised"

            async def before_run(self, ctx: Any) -> None:
                self.before_run_calls += 1

            async def before_tool(self, ctx: Any, tc: Any) -> str | None:
                return None

            async def after_tool(
                self, ctx: Any, tc: Any, result_str: str,
            ) -> None: ...

            async def after_run(self, ctx: Any, result: Any) -> None:
                self.after_run_calls += 1
                self.last_result = result

        spy = _SpyGuardrail()
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(t0=t0, guardrails=[cast(Any, spy)])

        result = await agent.run("hi")

        assert spy.before_run_calls == 1
        assert spy.after_run_calls == 1
        # after_run sees the populated AgentResult, not None.
        assert spy.last_result is result
        # guardrail_log carries two pass entries (one per hook).
        assert len(result.guardrail_log) == 2
        assert {entry.hook for entry in result.guardrail_log} == {
            "before_run",
            "after_run",
        }
        assert all(entry.kind == "pass" for entry in result.guardrail_log)

    async def test_before_tool_after_tool_do_not_fire_on_json(self) -> None:
        # JSON has no tool loop — before_tool/after_tool must never fire.
        class _ToolSpy:
            name = "tool-spy"

            def __init__(self) -> None:
                self.before_tool_calls = 0
                self.after_tool_calls = 0

            async def before_run(self, ctx: Any) -> None: ...

            async def before_tool(self, ctx: Any, tc: Any) -> str | None:
                self.before_tool_calls += 1
                return None

            async def after_tool(
                self, ctx: Any, tc: Any, result_str: str,
            ) -> None:
                self.after_tool_calls += 1

            async def after_run(self, ctx: Any, result: Any) -> None: ...

        spy = _ToolSpy()
        client = FakeMCPClient(tools=[_mcp_tool("add")])
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(t0=t0, client=client, guardrails=[cast(Any, spy)])

        await agent.run("hi")

        assert spy.before_tool_calls == 0
        assert spy.after_tool_calls == 0


class TestMemory:
    async def test_memory_save_persists_raw_json(self) -> None:
        memory = InMemoryMemory()
        episode = uuid4()
        t0 = MockGateway.script_json(
            [
                {
                    "raw": '{"x": 1}',
                    "parsed": {"x": 1},
                    "episode_id": episode,
                }
            ]
        )
        agent = _json_agent(t0=t0, memory=memory)

        result = await agent.run("hi", episode_id=str(episode))

        records = await memory.load(str(episode))
        assert len(records) == 1
        record = records[0]
        assert record.agent_response == '{"x": 1}'
        assert record.agent_response == result.response
        assert record.turns == 1
        assert record.inference_ids == result.inference_ids


class TestEpisodePropagation:
    async def test_episode_id_propagation(self) -> None:
        episode = uuid4()
        t0 = MockGateway.script_json(
            [{"raw": _OK_RAW, "episode_id": episode}]
        )
        agent = _json_agent(t0=t0)

        result = await agent.run("hi", episode_id=str(episode))

        # The gateway received the episode_id we passed in.
        assert t0.calls[0]["episode_id"] == str(episode)
        # The result echoes back the gateway's reported episode_id.
        assert result.episode_id == str(episode)


class TestMessagesMemoryStrategy:
    """Reject ``memory_strategy='messages'`` for JSON runs uniformly.

    The rejection is keyed off the resolved strategy rather than the
    presence of loaded history, so an episode with no prior messages
    (or no ``episode_id`` at all) still surfaces the misconfiguration
    instead of running silently.
    """

    @staticmethod
    def _messages_memory(
        *, history: list[dict[str, str]],
    ) -> Any:
        """Build a minimal memory provider that implements load_history."""

        class _MessagesMemory:
            async def load(self, episode_id: str) -> list[Any]:
                return []

            async def save(self, record: Any) -> None: ...

            async def search(
                self, query: str, *, limit: int = 5,
            ) -> list[Any]:
                return []

            async def load_history(self, episode_id: str) -> list[Any]:
                return list(history)

        return _MessagesMemory()

    async def test_rejected_with_non_empty_history(self) -> None:
        memory = self._messages_memory(
            history=[{"role": "user", "content": "prior"}],
        )
        episode = uuid4()
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(
            t0=t0,
            memory=cast(Any, memory),
            memory_strategy="messages",
        )

        with pytest.raises(ConfigurationError, match="messages"):
            await agent.run("hi", episode_id=str(episode))

    async def test_rejected_with_empty_history(self) -> None:
        # Before applying the strategy-level check, an empty history
        # would have let the misconfigured run proceed silently because
        # ``extra_kwargs`` never picked up ``prior_messages``.
        memory = self._messages_memory(history=[])
        episode = uuid4()
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(
            t0=t0,
            memory=cast(Any, memory),
            memory_strategy="messages",
        )

        with pytest.raises(ConfigurationError, match="messages"):
            await agent.run("hi", episode_id=str(episode))

    async def test_rejected_without_episode_id(self) -> None:
        # No ``episode_id`` ⇒ no memory load happens at all; the check
        # still fires because it keys off the configured strategy.
        memory = self._messages_memory(history=[])
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(
            t0=t0,
            memory=cast(Any, memory),
            memory_strategy="messages",
        )

        with pytest.raises(ConfigurationError, match="messages"):
            await agent.run("hi")
