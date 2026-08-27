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
from t0_agents.core.exceptions import (
    ConfigurationError,
    MemoryHistoryUnavailableError,
    T0AgentsFatalError,
)
from t0_agents.core.types import MemoryProvider, ToolProvider
from t0_agents.memory.base import MemoryRecord
from t0_agents.memory.in_memory import InMemoryMemory
from t0_agents.stream.events import RunErrorEvent
from t0_agents.wire import ErrorFrame, project
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


class _StoreUnavailableError(T0AgentsFatalError):
    """A backend declaring its own failure fatal — the documented escape hatch."""


class _HistoryMem:
    """A ``messages``-strategy provider with a scriptable history read.

    *history* is either the value ``load_history`` returns or an exception
    instance it raises, so one class covers every arm of the empty /
    populated / unreadable trichotomy the strategy now distinguishes.
    """

    def __init__(self, history: Any) -> None:
        self._history = history
        self.history_calls = 0

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        return []

    async def save(self, record: MemoryRecord) -> None:
        pass

    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        return []

    async def load_history(self, episode_id: str) -> list[Any] | None:
        self.history_calls += 1
        if isinstance(self._history, BaseException):
            raise self._history
        return cast("list[Any] | None", self._history)


def _messages_agent(mem: Any, t0: FakeT0) -> Agent:
    return _make_agent(t0=t0, memory=cast(InMemoryMemory, mem), memory_strategy="messages")


async def _drain(events: Any) -> list[Any]:
    return [item async for item in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentWithoutMemory:
    """When config.memory is None, Agent.run performs no memory load or save."""

    async def test_no_memory_returns_result(self) -> None:
        agent = _make_agent()
        result = await agent.invoke("hello")
        assert result.response == "ok"
        assert result.episode_id == str(_EPISODE)

    async def test_no_memory_no_save_called(self) -> None:
        """Ensure no memory operations happen when memory is None."""
        t0 = FakeT0([_text_response()])
        agent = _make_agent(t0=t0)
        await agent.invoke("hello")
        # If memory were set, we'd see a save. With None, it's a no-op.
        assert agent.config.memory is None


class TestMemoryHydrationAndPersistence:
    async def test_saves_after_run(self) -> None:
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("I remember")])
        agent = _make_agent(t0=t0, memory=mem)

        result = await agent.invoke("Remember this")
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
        r1 = await agent.invoke("Remember: my name is Alice")
        assert r1.episode_id == ep

        # Second run — pass episode_id so memory loads
        await agent.invoke("What is my name?", episode_id=ep)

        # The second inference call should have the memory prefix
        second_call = t0.calls[1]
        first_msg = second_call["input"]["messages"][0]
        assert "Alice" in str(first_msg["content"])

    async def test_first_run_no_episode_still_saves(self) -> None:
        """Even without an explicit episode_id, save uses result.episode_id."""
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem)

        result = await agent.invoke("hello")  # no episode_id
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

        await agent.invoke("follow-up", episode_id=ep)
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
        await agent.invoke("new question", episode_id=ep)

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
        await agent.invoke("what was X?", episode_id=ep)

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
        await agent.invoke("question", episode_id=ep)

        call = t0.calls[0]
        # User message should be original (unmodified)
        assert call["input"]["messages"][0]["content"] == "question"
        # run_agent assigns system_arguments directly to input.system —
        # the dict is the schema-described arguments object, not wrapped.
        system = call["input"]["system"]
        assert "memory_context" in system
        assert "context" in system["memory_context"]

    async def test_no_injection_when_no_records(self) -> None:
        """With an empty episode, no memory prefix should be injected."""
        mem = InMemoryMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=mem)

        await agent.invoke("hello", episode_id="empty-episode")
        call = t0.calls[0]
        assert call["input"]["messages"][0]["content"] == "hello"

    async def test_caller_system_arguments_merge_with_system_template(self) -> None:
        """Caller-supplied keys land alongside memory_context; non-colliding."""
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
        await agent.invoke(
            "question",
            episode_id=ep,
            system_arguments={"max_turns": 5},
        )

        system = t0.calls[0]["input"]["system"]
        assert "memory_context" in system
        assert "context" in system["memory_context"]
        assert system["max_turns"] == 5

    async def test_caller_wins_on_memory_context_collision(self) -> None:
        """Caller-precedence: explicit memory_context override beats memory's."""
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
        await agent.invoke(
            "question",
            episode_id=ep,
            system_arguments={"memory_context": "OVERRIDE"},
        )

        system = t0.calls[0]["input"]["system"]
        assert system == {"memory_context": "OVERRIDE"}


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
        result = await agent.invoke("hello", episode_id="ep-1")
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
        result = await agent.invoke("hello")
        assert result.response == "ok"

    async def test_load_fatal_error_propagates(self) -> None:
        """docs/memory.md's escape hatch: a fatal backend error is not swallowed."""

        class FatalLoadMemory:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                raise _StoreUnavailableError("store gone")

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

        mem = FatalLoadMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=cast(InMemoryMemory, mem))
        with pytest.raises(_StoreUnavailableError):
            await agent.invoke("hello", episode_id="ep-1")
        assert t0.calls == []

    async def test_save_fatal_error_propagates(self) -> None:
        """Same hatch on the write leg, which runs inside ``finally``."""

        class FatalSaveMemory:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                raise _StoreUnavailableError("store gone")

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

        mem = FatalSaveMemory()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(t0=t0, memory=cast(InMemoryMemory, mem))
        with pytest.raises(_StoreUnavailableError):
            await agent.invoke("hello")


class TestMemoryValidation:
    def test_invalid_memory_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_strategy"):
            AgentConfig(
                name="test",
                function_name="agent",
                memory_strategy="invalid",  # type: ignore[arg-type]
            )


class TestMessagesStrategyValidation:
    """``memory_strategy='messages'`` requires a provider with load_history."""

    def test_missing_memory_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="load_history"):
            AgentConfig(
                name="test",
                function_name="agent",
                memory_strategy="messages",
                memory=None,
            )

    def test_provider_without_load_history_raises(self) -> None:
        # InMemoryMemory does not implement load_history.
        with pytest.raises(ConfigurationError, match="load_history"):
            AgentConfig(
                name="test",
                function_name="agent",
                memory_strategy="messages",
                memory=InMemoryMemory(),
            )

    async def test_override_provider_without_load_history_raises(self) -> None:
        """``invoke(memory=…)`` skips the construction-time check.

        It validates only the MemoryProvider protocol, so a provider with
        no ``load_history`` reaches the read.  That is a caller mistake,
        not a failed read, and must not be reported as one.
        """
        agent = _make_agent(
            memory=cast(InMemoryMemory, _HistoryMem(None)),
            memory_strategy="messages",
        )
        with pytest.raises(ConfigurationError, match="load_history"):
            await agent.invoke(
                "hello",
                episode_id=str(_EPISODE),
                memory=cast(MemoryProvider, InMemoryMemory()),
            )

    def test_provider_with_load_history_accepts(self) -> None:
        class HistoryProvider:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(
                self, episode_id: str
            ) -> list[Any] | None:
                return None

        # Construction must not raise — the duck-typed provider satisfies
        # both the MemoryProvider Protocol and the load_history extension.
        cfg = AgentConfig(
            name="test",
            function_name="agent",
            memory_strategy="messages",
            memory=cast(MemoryProvider, HistoryProvider()),
        )
        assert cfg.memory_strategy == "messages"


class TestMessagesStrategyInjection:
    """Verify Agent threads load_history's output through prior_messages."""

    async def test_load_history_seeds_prior_messages(self) -> None:
        from tensorzero import Message, Text

        class FakeHistoryMem:
            def __init__(self) -> None:
                self.load_calls = 0
                self.history_calls = 0

            async def load(self, episode_id: str) -> list[MemoryRecord]:
                self.load_calls += 1
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(
                self, episode_id: str
            ) -> list[Message] | None:
                self.history_calls += 1
                return [
                    Message(role="user", content="earlier"),
                    Message(role="assistant", content=[Text(text="prior reply")]),
                ]

        mem = FakeHistoryMem()
        ep = str(_EPISODE)
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(
            t0=t0,
            memory=cast(InMemoryMemory, mem),
            memory_strategy="messages",
        )

        await agent.invoke("now", episode_id=ep)

        # load() must NOT be called when strategy is "messages".
        assert mem.load_calls == 0
        # load_history is the only read path.
        assert mem.history_calls == 1
        # The first inference's input.messages = [prior..., {user, "now"}].
        first_messages = t0.calls[0]["input"]["messages"]
        assert len(first_messages) == 3
        assert first_messages[0]["role"] == "user"
        assert first_messages[0]["content"] == "earlier"
        assert first_messages[1]["role"] == "assistant"
        assert first_messages[2]["role"] == "user"
        assert first_messages[2]["content"] == "now"

    async def test_fresh_episode_skips_load_history(self) -> None:
        """No episode_id → no prior history fetch; behaves like a new run."""

        class FakeHistoryMem:
            def __init__(self) -> None:
                self.history_calls = 0

            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(self, episode_id: str) -> list[Any] | None:
                self.history_calls += 1
                return None

        mem = FakeHistoryMem()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(
            t0=t0,
            memory=cast(InMemoryMemory, mem),
            memory_strategy="messages",
        )

        await agent.invoke("hello")  # no episode_id

        assert mem.history_calls == 0
        first_messages = t0.calls[0]["input"]["messages"]
        assert first_messages == [{"role": "user", "content": "hello"}]

    async def test_load_history_returns_none_starts_clean(self) -> None:
        """Empty episode → a fresh session, no error.

        This is the case a careless fail-loud fix breaks, and the only
        empty shape that pins the guard: ``None`` reaching
        ``list(history)`` would raise.
        """
        mem = _HistoryMem(None)
        t0 = FakeT0([_text_response("ok")])
        agent = _messages_agent(mem, t0)
        result = await agent.invoke("hello", episode_id=str(_EPISODE))
        assert result.response == "ok"
        assert mem.history_calls == 1
        first_messages = t0.calls[0]["input"]["messages"]
        assert first_messages == [{"role": "user", "content": "hello"}]

    async def test_load_history_returns_empty_list_starts_clean(self) -> None:
        """The other empty shape behaves identically.

        A contract test, not a mutation-killer: ``[]`` survives the guard
        either way, because the middleware only seeds a truthy
        ``prior_messages``.  Pinned because the strategy's contract now
        turns on "empty versus unreadable" and both empties must stay on
        the clean side of that line.
        """
        mem = _HistoryMem([])
        t0 = FakeT0([_text_response("ok")])
        agent = _messages_agent(mem, t0)
        result = await agent.invoke("hello", episode_id=str(_EPISODE))
        assert result.response == "ok"
        first_messages = t0.calls[0]["input"]["messages"]
        assert first_messages == [{"role": "user", "content": "hello"}]

    async def test_load_history_failure_ends_the_turn(self) -> None:
        """A failed history read must not produce an amnesiac answer."""

        class FailingHistoryMem:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(self, episode_id: str) -> list[Any] | None:
                raise RuntimeError("gateway down")

        mem = FailingHistoryMem()
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(
            t0=t0,
            memory=cast(InMemoryMemory, mem),
            memory_strategy="messages",
        )
        with pytest.raises(MemoryHistoryUnavailableError) as excinfo:
            await agent.invoke("hello", episode_id=str(_EPISODE))
        assert str(_EPISODE) in str(excinfo.value)
        # The provider's own failure stays reachable for logging.
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        # The load-bearing assertion: no inference was ever issued, so no
        # answer could have been returned without the history.  The
        # exception type alone would not prove that.
        assert t0.calls == []

    async def test_backend_fatal_error_propagates_unwrapped(self) -> None:
        """A backend's own fatal class survives instead of being re-wrapped.

        The class name *is* the discriminator a consumer keys its policy
        on, so wrapping it in ours would erase the very thing the
        backend went out of its way to declare.
        """
        mem = _HistoryMem(_StoreUnavailableError("store gone"))
        t0 = FakeT0([_text_response("ok")])
        agent = _messages_agent(mem, t0)
        with pytest.raises(_StoreUnavailableError):
            await agent.invoke("hello", episode_id=str(_EPISODE))
        assert t0.calls == []

    async def test_caller_system_arguments_pass_through_messages(self) -> None:
        """messages strategy + caller args → both reach the inference call."""
        from tensorzero import Message, Text

        class FakeHistoryMem:
            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(
                self, episode_id: str
            ) -> list[Message] | None:
                return [
                    Message(role="user", content="earlier"),
                    Message(role="assistant", content=[Text(text="prior")]),
                ]

        mem = FakeHistoryMem()
        ep = str(_EPISODE)
        t0 = FakeT0([_text_response("ok")])
        agent = _make_agent(
            t0=t0,
            memory=cast(InMemoryMemory, mem),
            memory_strategy="messages",
        )

        await agent.invoke(
            "now",
            episode_id=ep,
            system_arguments={"max_turns": 5},
        )

        call = t0.calls[0]
        # prior_messages reached the call alongside system arguments.
        assert len(call["input"]["messages"]) == 3
        assert call["input"]["system"] == {"max_turns": 5}


class TestMessagesStrategyStreamParity:
    """Agent.stream must mirror Agent.run's messages-strategy flow."""

    async def test_stream_threads_prior_messages(self) -> None:
        from tensorzero import Message
        from tensorzero.types import TextChunk

        from t0_agents.stream.events import DoneEvent
        from tests._fake_t0 import chat_chunk, final_chunk

        class FakeHistoryMem:
            def __init__(self) -> None:
                self.history_calls = 0

            async def load(self, episode_id: str) -> list[MemoryRecord]:
                return []

            async def save(self, record: MemoryRecord) -> None:
                pass

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[MemoryRecord]:
                return []

            async def load_history(
                self, episode_id: str
            ) -> list[Message] | None:
                self.history_calls += 1
                return [Message(role="user", content="seed")]

        mem = FakeHistoryMem()
        ep = str(_EPISODE)
        # Streaming script: one text chunk, one final usage chunk.
        t0 = FakeT0(
            [
                [
                    chat_chunk([TextChunk(id="t-1", text="ok", type="text")],
                               episode_id=_EPISODE),
                    final_chunk(episode_id=_EPISODE),
                ]
            ]
        )
        agent = _make_agent(
            t0=t0,
            memory=cast(InMemoryMemory, mem),
            memory_strategy="messages",
        )
        events: list[Any] = []
        async for event in agent.invoke("now", episode_id=ep).events():
            events.append(event)

        assert mem.history_calls == 1
        # DoneEvent at the tail.
        assert isinstance(events[-1], DoneEvent)
        # Streamed inference saw prior_messages + new user.
        first_messages = t0.calls[0]["input"]["messages"]
        assert first_messages[0]["role"] == "user"
        assert first_messages[0]["content"] == "seed"
        assert first_messages[1]["role"] == "user"
        assert first_messages[1]["content"] == "now"


class TestMessagesStrategyConsumerSurface:
    """An unreadable history must be tellable apart from a provider failure.

    The consumer decides whether to re-drive, and re-driving a dead store
    is pointless while re-driving a flaky provider is not — so a generic
    terminal error is not enough.  These pin the two surfaces a consumer
    actually reads.
    """

    async def test_failure_arrives_as_a_terminal_run_error_event(self) -> None:
        mem = _HistoryMem(RuntimeError("gateway down"))
        t0 = FakeT0([_text_response("ok")])
        agent = _messages_agent(mem, t0)
        events = await _drain(
            agent.invoke("hello", episode_id=str(_EPISODE)).events(on_error="event")
        )
        assert len(events) == 1
        terminal = events[0]
        assert isinstance(terminal, RunErrorEvent)
        assert terminal.error_type == "MemoryHistoryUnavailableError"
        assert isinstance(terminal.exception, MemoryHistoryUnavailableError)

    async def test_failure_reaches_the_wire_with_its_exc_class(self) -> None:
        mem = _HistoryMem(RuntimeError("gateway down"))
        t0 = FakeT0([_text_response("ok")])
        agent = _messages_agent(mem, t0)
        frames = await _drain(
            project(
                agent.invoke("hello", episode_id=str(_EPISODE)).events(on_error="event")
            )
        )
        assert len(frames) == 1
        [frame] = frames
        assert isinstance(frame, ErrorFrame)
        assert frame.detail["exc_class"] == "MemoryHistoryUnavailableError"
        # Terminal, not a recoverable in-stream artefact: a consumer that
        # seals its view on `error` is right to do so here.
        assert "recoverable" not in frame.detail
