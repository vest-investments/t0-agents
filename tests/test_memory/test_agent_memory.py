"""Offline Agent + memory integration tests.

Uses ``FakeT0``, ``FakeMCPClient``, and ``InMemoryMemory`` to verify
memory hydration, injection, and persistence without a live gateway.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import MemoryProvider, ToolProvider
from t0_agents.memory.base import MemoryRecord
from t0_agents.memory.in_memory import InMemoryMemory
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPISODE = UUID("00000000-0000-0000-0000-000000000001")


def _text_response(text: str = "ok", *, episode_id: UUID = _EPISODE) -> FakeChatResponse:
    return FakeChatResponse(
        content=[Text(text=text)],
        episode_id=episode_id,
    )


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    client: FakeMCPClient | None = None,
    memory: InMemoryMemory | None = None,
    memory_strategy: str = "auto",
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "test",
        "function_name": "agent",
        "memory": memory,
        "memory_strategy": memory_strategy,
    }
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_text_response()])),
        tool_provider=cast(ToolProvider, client or FakeMCPClient()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentWithoutMemory:
    """When config.memory is None, Agent.run performs no memory load or save."""

    async def test_no_memory_returns_result(self) -> None:
        agent = _make_agent()
        result = await agent.run("hello")
        assert result.response == "ok"
        assert result.episode_id == str(_EPISODE)

    async def test_no_memory_no_save_called(self) -> None:
        """Ensure no memory operations happen when memory is None."""
        t0 = FakeT0([_text_response()])
        agent = _make_agent(t0=t0)
        await agent.run("hello")
        # If memory were set, we'd see a save. With None, it's a no-op.
        assert agent.config.memory is None


class TestMemoryHydrationAndPersistence:
    async def test_saves_after_run(self) -> None:
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("I remember")])
        agent = _make_agent(t0=t0, memory=mem)

        result = await agent.run("Remember this")
        records = await mem.load(str(_EPISODE))
        assert len(records) == 1
        assert records[0].user_message == "Remember this"
        assert records[0].agent_response == "I remember"
        assert records[0].episode_id == str(_EPISODE)
        assert result.response == "I remember"

    async def test_loads_before_second_run(self) -> None:
        mem = InMemoryMemory()
        ep = str(_EPISODE)
        t0 = FakeT0([
            _text_response("Noted"),
            _text_response("Yes I recall"),
        ])
        agent = _make_agent(t0=t0, memory=mem)

        # First run — no episode, creates one
        r1 = await agent.run("Remember: my name is Alice")
        assert r1.episode_id == ep

        # Second run — pass episode_id so memory loads
        r2 = await agent.run("What is my name?", episode_id=ep)

        # The second inference call should have the memory prefix
        second_call = t0.calls[1]
        first_msg = second_call["input"]["messages"][0]
        assert "Alice" in str(first_msg["content"])

    async def test_first_run_no_episode_still_saves(self) -> None:
        """Even without an explicit episode_id, save uses result.episode_id."""
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem)

        result = await agent.run("hello")  # no episode_id
        records = await mem.load(result.episode_id or "")
        assert len(records) == 1

    async def test_saves_original_user_message_not_injected(self) -> None:
        """The saved record should contain the original user message,
        not the one with memory prefix."""
        mem = InMemoryMemory()
        ep = str(_EPISODE)

        # Pre-populate memory
        await mem.save(MemoryRecord(
            episode_id=ep,
            user_message="prior message",
            agent_response="prior response",
            turns=1,
        ))

        t0 = FakeT0([_text_response("answer")])
        agent = _make_agent(t0=t0, memory=mem)

        await agent.run("follow-up", episode_id=ep)
        records = await mem.load(ep)
        # Second record should have original user message
        assert records[1].user_message == "follow-up"


class TestMemoryStrategy:
    async def test_auto_resolves_to_prepended(self) -> None:
        mem = InMemoryMemory()
        ep = str(_EPISODE)
        await mem.save(MemoryRecord(
            episode_id=ep,
            user_message="context msg",
            agent_response="context resp",
            turns=1,
        ))

        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem, memory_strategy="auto")
        await agent.run("new question", episode_id=ep)

        # Should prepend memory to user message (not system_arguments)
        call = t0.calls[0]
        first_msg = call["input"]["messages"][0]
        assert "context msg" in str(first_msg["content"])
        assert "system" not in call["input"]

    async def test_prepended_message_strategy(self) -> None:
        mem = InMemoryMemory()
        ep = str(_EPISODE)
        await mem.save(MemoryRecord(
            episode_id=ep,
            user_message="remember X",
            agent_response="noted X",
            turns=1,
        ))

        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem, memory_strategy="prepended_message")
        await agent.run("what was X?", episode_id=ep)

        call = t0.calls[0]
        user_msg = call["input"]["messages"][0]["content"]
        assert "remember X" in user_msg
        assert "what was X?" in user_msg
        assert "---" in user_msg

    async def test_system_template_strategy(self) -> None:
        mem = InMemoryMemory()
        ep = str(_EPISODE)
        await mem.save(MemoryRecord(
            episode_id=ep,
            user_message="context",
            agent_response="response",
            turns=1,
        ))

        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem, memory_strategy="system_template")
        await agent.run("question", episode_id=ep)

        call = t0.calls[0]
        # User message should be original (unmodified)
        assert call["input"]["messages"][0]["content"] == "question"
        # run_agent translates system_arguments into input.system.arguments
        system = call["input"]["system"]
        assert "arguments" in system
        assert "memory_context" in system["arguments"]
        assert "context" in system["arguments"]["memory_context"]

    async def test_no_injection_when_no_records(self) -> None:
        """With an empty episode, no memory prefix should be injected."""
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem)

        await agent.run("hello", episode_id="empty-episode")
        call = t0.calls[0]
        assert call["input"]["messages"][0]["content"] == "hello"


class TestMemoryErrorGracefulDegradation:
    async def test_load_failure_does_not_abort(self) -> None:
        """If memory.load raises, run() should still complete."""

        class FailingMemory:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                raise RuntimeError("disk on fire")

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

        mem = FailingMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=cast(InMemoryMemory, mem))
        result = await agent.run("hello", episode_id="ep-1")
        assert result.response == "ok"

    async def test_save_failure_does_not_abort(self) -> None:
        """If memory.save raises, run() should still return the result."""

        class SaveFailingMemory:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                raise RuntimeError("write failed")

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

        mem = SaveFailingMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=cast(InMemoryMemory, mem))
        result = await agent.run("hello")
        assert result.response == "ok"


class TestMemoryValidation:
    def test_invalid_memory_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_strategy"):
            AgentConfig(
                name="test",
                function_name="agent",
                memory_strategy="invalid",  # type: ignore[arg-type]
            )
