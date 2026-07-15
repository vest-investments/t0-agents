"""Unit tests for :class:`t0_agents.glow.conversation.Conversation`.

Offline: the underlying :class:`Agent` is patched to ``_ConvAgent`` (records
each ``invoke`` on ``_INVOKES``) for the send/guard/tag cases, and a
``FakeT0`` drives the ``list_inferences``-backed hydrate / listing cases.  A
``ScriptedAgent`` override covers the through-the-framework multi-turn path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import UUID

import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.generated_types import (
    ContentBlockChatOutputText,
    StoredInferenceChat,
    StoredInput,
    StoredInputMessage,
    StoredInputMessageContentText,
)

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow import App, ConversationSummary
from t0_agents.stream.events import AgentEvent, DoneEvent
from t0_agents.testing import ScriptedAgent
from tests._fake_t0 import FakeListInferencesResponse, FakeT0

# Every _ConvAgent.invoke appends its kwargs here; the fixture clears it.
_INVOKES: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ConvRun:
    """Duck-typed Run: result()/await, a DoneEvent stream, and text stream."""

    def __init__(self, value: str) -> None:
        self._agent_result: AgentResult[Any] = AgentResult(
            value=value, episode_id="ep_fake", inference_ids=["if"], turns=1
        )

    async def result(self) -> AgentResult[Any]:
        return self._agent_result

    async def events(self, *, on_error: str = "raise") -> AsyncIterator[AgentEvent]:
        yield DoneEvent(result=self._agent_result)

    async def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]:
        yield cast(str, self._agent_result.value)


class _ConvAgent:
    """Fake Agent patched into ``glow.app`` — records each invoke's kwargs."""

    def __init__(self, config: Any, *, t0: Any, tool_provider: Any = None) -> None:
        self._config = config

    def invoke(self, user_message: str, **kwargs: Any) -> Run[Any]:
        _INVOKES.append({"user_message": user_message, **kwargs})
        return cast(Run[Any], _ConvRun(value=f"reply:{user_message}"))


class _FakeMemory:
    """MemoryProvider + load_history — satisfies memory_strategy="messages"."""

    async def load(self, episode_id: str) -> list[Any]:
        return []

    async def save(self, record: Any) -> None:
        return None

    async def search(self, query: str, *, limit: int = 5) -> list[Any]:
        return []

    async def load_history(self, episode_id: str) -> list[Any] | None:
        return None


class _FakeIndex:
    """A ConversationIndex the App should delegate to when passed as index=."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str] | None, int]] = []

    async def list_conversations(
        self, *, tags: dict[str, str] | None = None, limit: int = 20
    ) -> list[ConversationSummary]:
        self.calls.append((tags, limit))
        return [
            ConversationSummary(
                conversation_id="from-index",
                title="t",
                preview="p",
                turn_count=1,
                updated_at=1,
            )
        ]


def _user_text_msg(text: str) -> StoredInputMessage:
    return StoredInputMessage(
        role="user", content=[StoredInputMessageContentText(text=text)]
    )


def _stored_chat(
    episode_id: str, user_messages: list[StoredInputMessage], output_text: str
) -> StoredInferenceChat:
    return StoredInferenceChat(
        episode_id=episode_id,
        function_name="agent",
        inference_id="if1",
        timestamp="2026-05-01T10:00:00Z",
        variant_name="v1",
        input=StoredInput(messages=user_messages),
        output=[ContentBlockChatOutputText(text=output_text)],
    )


def _sentinel_gateway() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


@pytest.fixture
def conv_app(monkeypatch: pytest.MonkeyPatch) -> App:
    _INVOKES.clear()
    monkeypatch.setattr("t0_agents.glow.app.Agent", _ConvAgent)
    return App(t0=_sentinel_gateway())


# ---------------------------------------------------------------------------
# Identity — conv.id == episode_id (UUIDv7)
# ---------------------------------------------------------------------------


def test_conversation_id_is_a_minted_uuidv7(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    assert UUID(conv.id).version == 7


def test_resume_uses_the_explicit_id(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation(id="resumed-episode")
    assert conv.id == "resumed-episode"


async def test_sends_share_one_episode_id(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a")
    await conv.send("b")
    assert [c["episode_id"] for c in _INVOKES] == [conv.id, conv.id]


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


async def test_transcript_accumulates_boundary_pairs(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a")
    await conv.send("b")
    assert conv.transcript == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "reply:a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "reply:b"},
    ]


async def test_streamed_send_records(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    chunks = [c async for c in conv.send("a").stream_text()]
    assert chunks == ["reply:a"]
    assert conv.transcript[-1] == {"role": "assistant", "content": "reply:a"}


# ---------------------------------------------------------------------------
# E1 guards
# ---------------------------------------------------------------------------


def test_sequential_guard_blocks_an_unconsumed_send(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    conv.send("a")  # deliberately not consumed
    with pytest.raises(ConfigurationError, match="not been consumed"):
        conv.send("b")
    assert _INVOKES == []  # neither send actually invoked the agent


async def test_user_sugar_stamps_write_tags(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation(user="c1")
    await conv.send("a")
    assert _INVOKES[0]["tags"] == {"user": "c1"}


async def test_non_messages_agent_seeds_prior_messages(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a")
    await conv.send("b")
    # First turn: empty transcript → no prior_messages seeded.
    assert "prior_messages" not in _INVOKES[0]
    # Second turn: the first turn's boundary pair rides as prior_messages.
    assert _INVOKES[1]["prior_messages"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "reply:a"},
    ]


async def test_messages_mode_omits_prior_messages(conv_app: App) -> None:
    @conv_app.agent("t", memory_strategy="messages", memory=_FakeMemory())
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a")
    await conv.send("b")
    # The provider weaves history via load_history; passing prior_messages
    # would suppress it — the double-injection guard keeps it out.
    assert all("prior_messages" not in c for c in _INVOKES)


# ---------------------------------------------------------------------------
# hydrate
# ---------------------------------------------------------------------------


def test_hydrate_requires_an_id(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    with pytest.raises(ConfigurationError, match="needs id="):
        t.conversation(hydrate=True)


async def test_hydrate_reconstructs_transcript() -> None:
    stored = _stored_chat("ep", [_user_text_msg("hi")], "hello")
    t0 = FakeT0(
        [], list_inferences_responses=[FakeListInferencesResponse(inferences=[stored])]
    )
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation(id="ep", hydrate=True)
    assert conv.transcript == []  # deferred until hydrate / first send
    await conv.hydrate()
    assert conv.transcript == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# app.conversations
# ---------------------------------------------------------------------------


async def test_app_conversations_delegates_to_index() -> None:
    app = App(t0=_sentinel_gateway())

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    index = _FakeIndex()
    out = await app.conversations(t, tags={"customer_id": "c1"}, limit=5, index=index)
    assert index.calls == [({"customer_id": "c1"}, 5)]
    assert [s.conversation_id for s in out] == ["from-index"]


async def test_app_conversations_dev_grade_via_gateway() -> None:
    stored = _stored_chat("ep1", [_user_text_msg("q")], "a")
    t0 = FakeT0(
        [], list_inferences_responses=[FakeListInferencesResponse(inferences=[stored])]
    )
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    out = await app.conversations(t, limit=10)
    assert [s.conversation_id for s in out] == ["ep1"]
    assert out[0].title == "q"


# ---------------------------------------------------------------------------
# through-the-framework multi-turn (ScriptedAgent)
# ---------------------------------------------------------------------------


async def test_scripted_agent_multi_turn() -> None:
    app = App(t0=_sentinel_gateway())

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    app.agent_overrides["t"] = ScriptedAgent(result="fixed reply")

    conv = t.conversation()
    first = await conv.send("q1")
    second = await conv.send("q2")

    assert first == second == "fixed reply"
    assert len(conv.transcript) == 4
    assert conv.transcript[0] == {"role": "user", "content": "q1"}
    assert conv.transcript[2] == {"role": "user", "content": "q2"}
