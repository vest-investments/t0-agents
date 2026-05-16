"""v0.12 backwards-compat — ``AgentConfig(memory=, guardrails=, sub_agents=)``
produces functionally equivalent runs to the v0.12 inline code.

These tests run the full ``Agent.invoke`` path with the default
middleware chain materialised by ``AgentConfig.__post_init__``.  The
chain must preserve the v0.12 cross-cutting behaviour (memory load
+ inject + save, guardrail before_run/after_run, ``GuardedToolProvider``
wrapping, cost-tracking, sub-agent dispatch via
``_OrchestrationToolProvider``).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import (
    AgentResult,
    Guardrail,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.memory.base import MemoryRecord
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


class _RecordingMemory:
    """Test-only :class:`MemoryProvider` that records ``load`` / ``save`` calls."""

    def __init__(self) -> None:
        self.loaded_for: list[str] = []
        self.saved: list[MemoryRecord] = []
        self._store: dict[str, list[MemoryRecord]] = {}

    async def load(self, episode_id: str) -> list[Any]:
        self.loaded_for.append(episode_id)
        return list(self._store.get(episode_id, []))

    async def save(self, record: Any) -> None:
        assert isinstance(record, MemoryRecord)
        self.saved.append(record)
        self._store.setdefault(record.episode_id, []).append(record)

    async def search(self, query: str, *, limit: int = 5) -> list[Any]:
        return []


class _RecordingGuardrail:
    """Test-only guardrail capturing every hook invocation."""

    name = "recorder"

    def __init__(self) -> None:
        self.before_run_seen: list[Any] = []
        self.after_run_seen: list[Any] = []
        self.before_tool_seen: list[Any] = []
        self.after_tool_seen: list[Any] = []

    async def before_run(self, ctx: Any) -> None:
        self.before_run_seen.append(ctx)

    async def before_tool(self, ctx: Any, tool_call: Any) -> str | None:
        self.before_tool_seen.append(tool_call)
        return None

    async def after_tool(self, ctx: Any, tool_call: Any, result_str: str) -> None:
        self.after_tool_seen.append((tool_call, result_str))

    async def after_run(self, ctx: Any, result: Any) -> None:
        self.after_run_seen.append(result)


def _ok_response(text: str = "ok") -> FakeChatResponse:
    return FakeChatResponse(
        content=[Text(type="text", text=text)],
        episode_id=UUID("11111111-2222-3333-4444-555555555555"),
        inference_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        usage=FakeUsage(input_tokens=5, output_tokens=3, cost=0.001),
    )


class TestBareChat:
    async def test_chat_result_shape_matches_v0_12(self) -> None:
        t0 = FakeT0([_ok_response("hello back")])
        agent = Agent(
            AgentConfig(name="bare", function_name="agent"),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        result = await agent.invoke("hi").result()

        assert isinstance(result, AgentResult)
        assert result.response == "hello back"
        assert result.turns == 1
        assert result.function_kind == "chat"
        assert result.parsed is None
        # No guardrails configured → guardrail_log is empty.
        assert result.guardrail_log == []


class TestChatWithMemory:
    async def test_memory_load_called_with_episode_id(self) -> None:
        memory = _RecordingMemory()
        t0 = FakeT0([_ok_response()])
        agent = Agent(
            AgentConfig(
                name="m",
                function_name="agent",
                memory=cast(MemoryProvider, memory),
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("hi", episode_id="ep-mem").result()

        assert memory.loaded_for == ["ep-mem"]

    async def test_memory_save_called_after_run(self) -> None:
        memory = _RecordingMemory()
        t0 = FakeT0([_ok_response("agent reply")])
        agent = Agent(
            AgentConfig(
                name="m",
                function_name="agent",
                memory=cast(MemoryProvider, memory),
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("user input").result()

        assert len(memory.saved) == 1
        record = memory.saved[0]
        # MemoryRecord must preserve the ORIGINAL user_message (not
        # the memory-injected effective form).
        assert record.user_message == "user input"
        assert record.agent_response == "agent reply"


class TestChatWithGuardrails:
    async def test_before_run_and_after_run_fire(self) -> None:
        guard = _RecordingGuardrail()
        t0 = FakeT0([_ok_response()])
        agent = Agent(
            AgentConfig(
                name="g",
                function_name="agent",
                guardrails=[cast(Guardrail, guard)],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        result = await agent.invoke("hi").result()

        assert len(guard.before_run_seen) == 1
        assert len(guard.after_run_seen) == 1
        # AgentResult.guardrail_log is populated with the before_run +
        # after_run "pass" entries.
        assert {e.hook for e in result.guardrail_log} == {
            "before_run", "after_run",
        }

    async def test_guard_ctx_user_message_is_original(self) -> None:
        # GuardrailContext.user_message must be the ORIGINAL user
        # message, not the memory-injected effective form.  Parity with
        # v0.12.
        guard = _RecordingGuardrail()
        memory = _RecordingMemory()
        # Pre-load memory so the next run has a prepended summary.
        await memory.save(
            MemoryRecord(
                episode_id="ep-mem",
                user_message="prior turn",
                agent_response="prior reply",
                turns=1,
                inference_ids=["prior-i-1"],
            ),
        )

        t0 = FakeT0([_ok_response()])
        agent = Agent(
            AgentConfig(
                name="gm",
                function_name="agent",
                guardrails=[cast(Guardrail, guard)],
                memory=cast(MemoryProvider, memory),
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        await agent.invoke("current turn", episode_id="ep-mem").result()

        # The before_run hook saw the original user_message, not the
        # memory-prepended form.
        assert len(guard.before_run_seen) == 1
        ctx = guard.before_run_seen[0]
        assert ctx.user_message == "current turn"


class TestEndToEndChain:
    async def test_memory_guardrails_chain_together(self) -> None:
        guard = _RecordingGuardrail()
        memory = _RecordingMemory()
        t0 = FakeT0([_ok_response("response")])
        agent = Agent(
            AgentConfig(
                name="combo",
                function_name="agent",
                memory=cast(MemoryProvider, memory),
                guardrails=[cast(Guardrail, guard)],
            ),
            t0=cast(AsyncTensorZeroGateway, t0),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        result = await agent.invoke("hi", episode_id="ep-combo").result()

        # Memory and guardrails both fired.
        assert memory.loaded_for == ["ep-combo"]
        assert len(memory.saved) == 1
        assert len(guard.before_run_seen) == 1
        assert len(guard.after_run_seen) == 1
        assert result.episode_id == str(
            UUID("11111111-2222-3333-4444-555555555555"),
        )
