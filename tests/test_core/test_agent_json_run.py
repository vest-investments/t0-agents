"""Unit tests for ``Agent.invoke`` single-shot dispatch.

Drives ``Agent.invoke`` with a ``JsonOutput`` against
``MockGateway.script_json`` for the JSON path and the shared
``tests/_fake_t0`` for the chat-byte-identity assertions.  Covers the
config-level ``output=JsonOutput(...)`` shape as well as per-call
overrides via ``Agent.invoke(msg, output=JsonOutput(...))``.

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
from t0_agents.core.outputs import JsonOutput
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
    output: JsonOutput | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "extractor",
        "function_name": "extract_customer_info",
        "output": output if output is not None else JsonOutput(json_mode="strict"),
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
    async def test_run_dispatches_to_json_when_output_is_jsonoutput(self) -> None:
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(t0=t0)

        result = await agent.invoke("Hi, I'm Sarah at sarah@example.com")

        assert result.function_kind == "json"
        assert result.parsed == {
            "name": "Sarah",
            "email": "sarah@example.com",
        }
        assert result.response == _OK_RAW
        assert result.turns == 1
        assert len(result.inference_ids) == 1

    async def test_json_agent_runs_without_tool_provider(self) -> None:
        """End-to-end: a JSON agent built without a provider can still run.

        Mirrors the canonical ``examples/agent/03_output_modes/structured_agent.py``
        v1.0 shape — no ``fastmcp.Client``, no noop server, just
        ``Agent(config, t0=t0)``.
        """
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = Agent(
            AgentConfig(
                name="extractor",
                function_name="extract_customer_info",
                output=JsonOutput(json_mode="strict"),
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
        )

        result = await agent.invoke("Hi, I'm Sarah at sarah@example.com")

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

        result = await agent.invoke("hi")

        assert result.function_kind == "chat"
        assert result.parsed is None
        assert result.response == "hello"
        assert result.response_source == "text"

    async def test_json_run_system_string_forwarded(self) -> None:
        """``system="..."`` flows through the single-shot terminal to
        ``input.system``."""
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _json_agent(t0=t0)

        await agent.invoke("hi", system="Extract structured fields from prose.")

        kwargs = t0.calls[0]
        assert kwargs["input"]["system"] == (
            "Extract structured fields from prose."
        )

    async def test_json_run_system_and_system_arguments_rejected(self) -> None:
        """Mutex applies to the single-shot dispatch path too — checked
        before the first inference is issued."""
        t0 = MockGateway.script_json([])
        agent = _json_agent(t0=t0)

        with pytest.raises(ValueError, match="Pass either `system`"):
            await agent.invoke(
                "hi",
                system="raw",
                system_arguments={"who": "world"},
            )
        assert t0.calls == []


class TestPerCallOverrides:
    async def test_per_call_output_schema_override(self) -> None:
        # Config-level schema is overridden when the caller passes a
        # per-call ``output_schema=`` (translated by the v0.12 shim into
        # a synthesised per-call :class:`JsonOutput`).
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        config_schema = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
        }
        agent = _json_agent(
            t0=t0,
            output=JsonOutput(
                schema=config_schema, json_mode="strict"
            ),
        )

        per_call_schema = {
            "type": "object",
            "properties": {"y": {"type": "string"}},
        }
        await agent.invoke(
            "hello",
            output=JsonOutput(schema=per_call_schema, json_mode="strict"),
        )

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
            output=JsonOutput(json_mode="strict"),
            client=FakeMCPClient(tools=[]),
        )

        await agent.invoke("hello", output=JsonOutput(json_mode="tool"))

        kwargs = t0.calls[0]
        assert kwargs["params"] == {"chat_completion": {"json_mode": "tool"}}

    async def test_config_schema_used_when_per_call_omits_override(self) -> None:
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        config_schema = {
            "type": "object",
            "properties": {"z": {"type": "boolean"}},
        }
        agent = _json_agent(
            t0=t0,
            output=JsonOutput(schema=config_schema, json_mode="strict"),
        )

        await agent.invoke("hello")

        kwargs = t0.calls[0]
        assert kwargs["output_schema"] == config_schema

    async def test_per_call_jsonoutput_overrides_chat_config(self) -> None:
        # v1.0 behaviour: passing a per-call output= overrides the
        # config-level ``ChatOutput`` for the duration of this invocation.
        # v0.12 raised ConfigurationError here — the v1.0 surface is
        # additive instead (the per-call mode wins).
        t0 = MockGateway.script_json([{"raw": _OK_RAW}])
        agent = _chat_agent(t0=t0)

        # Direct ``invoke`` with a per-call JsonOutput.  The shim
        # translation in ``agent.run`` would also work; we use
        # ``invoke`` here to keep the test free of deprecation warnings.
        result = await agent.invoke(
            "hello", output=JsonOutput(json_mode="strict"),
        ).result()

        assert result.function_kind == "json"


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

        result = await agent.invoke("hi")

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

        await agent.invoke("hi")

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

        result = await agent.invoke("hi", episode_id=str(episode))

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

        result = await agent.invoke("hi", episode_id=str(episode))

        # The gateway received the episode_id we passed in.
        assert t0.calls[0]["episode_id"] == str(episode)
        # The result echoes back the gateway's reported episode_id.
        assert result.episode_id == str(episode)


class TestMessagesMemoryStrategy:
    """Reject ``memory_strategy='messages'`` for single-shot runs uniformly.

    Single-shot output modes have no multi-turn message history to
    thread, so the strategy is rejected at config construction (when the
    output is single-shot) — keying off the resolved strategy rather
    than the presence of loaded history.
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

    def test_rejected_at_config_time(self) -> None:
        memory = self._messages_memory(
            history=[{"role": "user", "content": "prior"}],
        )

        with pytest.raises(ConfigurationError, match="messages"):
            AgentConfig(
                name="extractor",
                function_name="extract_customer_info",
                output=JsonOutput(json_mode="strict"),
                memory=cast(Any, memory),
                memory_strategy="messages",
            )
