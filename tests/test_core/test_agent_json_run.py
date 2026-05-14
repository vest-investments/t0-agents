"""Unit tests for ``Agent.run`` JSON-function dispatch (M2).

Uses a hand-rolled ``FakeJsonT0`` that mirrors
``tests/test_core/test_structured.py``'s fake.  ``MockGateway`` is
chat-only in v0.9.0 — M5 will fold the JSON branch in and these local
fakes will go away.  Chat-path byte-identity is exercised by reusing
the same shared ``FakeT0`` / ``FakeMCPClient`` fakes the rest of the
suite uses.
"""

from __future__ import annotations

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
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from t0_agents.memory.base import MemoryRecord
from t0_agents.memory.in_memory import InMemoryMemory
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

# ---------------------------------------------------------------------------
# JSON fakes — local until M5 generalises MockGateway.
# ---------------------------------------------------------------------------


def _fake_json_response(
    *,
    raw: str | None = '{"name": "Sarah", "email": "sarah@example.com"}',
    parsed: dict[str, Any] | None = None,
    inference_id: UUID | None = None,
    episode_id: UUID | None = None,
) -> JsonInferenceResponse:
    """Build a real ``JsonInferenceResponse`` from SDK dataclasses."""
    if parsed is None and raw is not None and raw.strip().startswith("{"):
        import json as _json

        try:
            parsed_value = _json.loads(raw)
        except _json.JSONDecodeError:
            parsed_value = None
    else:
        parsed_value = parsed
    return JsonInferenceResponse(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name="test-variant",
        output=JsonInferenceOutput(raw=raw, parsed=parsed_value),
        usage=Usage(input_tokens=10, output_tokens=20),  # type: ignore[call-arg]
    )


class FakeJsonT0:
    """Records every ``inference()`` call; returns scripted JSON responses."""

    def __init__(self, responses: list[JsonInferenceResponse | Exception]) -> None:
        self._queue: list[JsonInferenceResponse | Exception] = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> JsonInferenceResponse:
        captured: dict[str, Any] = dict(kwargs)
        if isinstance(captured.get("input"), dict):
            captured["input"] = dict(captured["input"])
            if isinstance(captured["input"].get("messages"), list):
                captured["input"]["messages"] = [
                    dict(m) if isinstance(m, dict) else m
                    for m in captured["input"]["messages"]
                ]
        self.calls.append(captured)
        if not self._queue:
            raise AssertionError("FakeJsonT0 ran out of canned responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _json_agent(
    *,
    t0: FakeJsonT0,
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
        t0 = FakeJsonT0([_fake_json_response()])
        agent = _json_agent(t0=t0)

        result = await agent.run("Hi, I'm Sarah at sarah@example.com")

        assert result.function_kind == "json"
        assert result.parsed == {
            "name": "Sarah",
            "email": "sarah@example.com",
        }
        assert result.response == '{"name": "Sarah", "email": "sarah@example.com"}'
        assert result.turns == 1
        assert len(result.inference_ids) == 1
        # Chat-only field default still preserved.
        assert result.response_source == "text"

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
        t0 = FakeJsonT0([_fake_json_response()])
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
        t0 = FakeJsonT0([_fake_json_response()])
        agent = _json_agent(t0=t0, json_mode="strict")

        await agent.run("hello", json_mode="tool")

        kwargs = t0.calls[0]
        assert kwargs["params"] == {"chat_completion": {"json_mode": "tool"}}

    async def test_config_schema_used_when_per_call_omits_override(self) -> None:
        t0 = FakeJsonT0([_fake_json_response()])
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
        t0 = FakeJsonT0([_fake_json_response()])
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
        t0 = FakeJsonT0([_fake_json_response()])
        agent = _json_agent(t0=t0, client=client, guardrails=[cast(Any, spy)])

        await agent.run("hi")

        assert spy.before_tool_calls == 0
        assert spy.after_tool_calls == 0


class TestMemory:
    async def test_memory_save_persists_raw_json(self) -> None:
        memory = InMemoryMemory()
        episode = uuid4()
        canned = _fake_json_response(
            raw='{"x": 1}',
            parsed={"x": 1},
            episode_id=episode,
        )
        t0 = FakeJsonT0([canned])
        agent = _json_agent(t0=t0, memory=memory)

        result = await agent.run("hi", episode_id=str(episode))

        records = await memory.load(str(episode))
        assert len(records) == 1
        record = cast(MemoryRecord, records[0])
        assert record.agent_response == '{"x": 1}'
        assert record.agent_response == result.response
        assert record.turns == 1
        assert record.inference_ids == result.inference_ids


class TestEpisodePropagation:
    async def test_episode_id_propagation(self) -> None:
        episode = uuid4()
        t0 = FakeJsonT0(
            [_fake_json_response(episode_id=episode)],
        )
        agent = _json_agent(t0=t0)

        result = await agent.run("hi", episode_id=str(episode))

        # The gateway received the episode_id we passed in.
        assert t0.calls[0]["episode_id"] == str(episode)
        # The result echoes back the gateway's reported episode_id.
        assert result.episode_id == str(episode)


class TestMessagesMemoryStrategy:
    async def test_messages_strategy_rejected_for_json_runs(self) -> None:
        # ``memory_strategy='messages'`` requires a ``load_history`` method
        # on the memory provider; build a minimal stub that satisfies it.
        class _MessagesMemory:
            async def load(self, episode_id: str) -> list[Any]:
                return []

            async def save(self, record: Any) -> None: ...

            async def search(
                self, query: str, *, limit: int = 5,
            ) -> list[Any]:
                return []

            async def load_history(self, episode_id: str) -> list[Any]:
                return [{"role": "user", "content": "prior"}]

        memory = _MessagesMemory()
        episode = uuid4()
        t0 = FakeJsonT0([_fake_json_response()])
        agent = _json_agent(
            t0=t0,
            memory=cast(Any, memory),
            memory_strategy="messages",
        )

        with pytest.raises(ConfigurationError, match="messages"):
            await agent.run("hi", episode_id=str(episode))
