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

from t0_agents.core.agent import AGENT_TAG
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow import App, ConversationSummary
from t0_agents.stream.events import AgentEvent, DoneEvent
from t0_agents.testing import ScriptedAgent
from tests._fake_t0 import (
    FakeChatResponse,
    FakeListInferencesResponse,
    FakeT0,
)

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


def _tag_filters(request: Any) -> dict[str, str]:
    """The tag equalities a ``ListInferencesRequest`` carries, key -> value.

    One tag arrives as a bare ``InferenceFilterTag``, several wrapped in an
    ``InferenceFilterAnd``; both collapse to the same mapping here.
    """
    node = request.filters
    if node is None:
        return {}
    children = getattr(node, "children", [node])
    return {child.key: child.value for child in children}


def _listing_t0(*, turns: int = 0, listings: int = 1) -> FakeT0:
    return FakeT0(
        [FakeChatResponse(content=[]) for _ in range(turns)],
        list_inferences_responses=[
            FakeListInferencesResponse(inferences=[]) for _ in range(listings)
        ],
    )


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


async def test_consumed_derived_send_unblocks_the_next_send(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a").with_(tags={"topic": "x"})  # the DERIVED run is consumed
    await conv.send("b")
    assert [i["user_message"] for i in _INVOKES] == ["a", "b"]


async def test_consumed_output_override_unblocks_the_next_send(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    await conv.send("a").text()  # ._spawn()-derived, same family
    await conv.send("b")
    assert [i["user_message"] for i in _INVOKES] == ["a", "b"]


def test_sequential_guard_still_blocks_an_unconsumed_derived_send(conv_app: App) -> None:
    @conv_app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation()
    conv.send("a").with_(tags={"topic": "x"})  # derived but never consumed
    with pytest.raises(ConfigurationError, match="not been consumed"):
        conv.send("b")
    assert _INVOKES == []


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


async def test_hydrate_retries_after_transient_failure() -> None:
    stored = _stored_chat("ep", [_user_text_msg("hi")], "hello")
    # First list_inferences raises (transient); the retry returns data.
    t0 = FakeT0(
        [],
        list_inferences_responses=[
            RuntimeError("transient gateway blip"),
            FakeListInferencesResponse(inferences=[stored]),
        ],
    )
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    conv = t.conversation(id="ep", hydrate=True)
    with pytest.raises(RuntimeError, match="transient"):
        await conv.hydrate()  # fails, but un-latches so a retry can re-fetch
    await conv.hydrate()  # retry succeeds
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
    # The index is handed the agent tag alongside the caller's own, so a
    # production inbox scopes exactly like the dev-grade default.
    assert index.calls == [({AGENT_TAG: "t", "customer_id": "c1"}, 5)]
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


# The cases below assert the REQUEST, not the response: the fake answers
# every filter with the same canned rows, so a response assertion cannot
# observe the scope at all.


async def test_app_conversations_scopes_the_request_to_the_agent() -> None:
    """The idiomatic declaration resolves its function and carries its tag."""
    t0 = _listing_t0()
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    await app.conversations(t)

    request = t0.list_inferences_calls[0]["request"]
    assert request.function_name == "agent"
    assert _tag_filters(request) == {AGENT_TAG: "t"}


async def test_app_conversations_separates_agents_sharing_one_function() -> None:
    t0 = _listing_t0(listings=2)
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("first", function_name="shared")
    async def first(q: str) -> str:  # type: ignore[empty-body]
        """First."""

    @app.agent("second", function_name="shared")
    async def second(q: str) -> str:  # type: ignore[empty-body]
        """Second."""

    await app.conversations(first)
    await app.conversations(second)

    requests = [call["request"] for call in t0.list_inferences_calls]
    assert [r.function_name for r in requests] == ["shared", "shared"]
    assert [_tag_filters(r)[AGENT_TAG] for r in requests] == ["first", "second"]


async def test_app_conversations_of_a_model_routed_agent_scopes_by_tag() -> None:
    """``model=`` has no function to filter on — the tag does the scoping."""
    t0 = _listing_t0()
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("m", model="chat_model")
    async def m(q: str) -> str:  # type: ignore[empty-body]
        """M."""

    await app.conversations(m)

    request = t0.list_inferences_calls[0]["request"]
    assert request.function_name is None
    assert _tag_filters(request) == {AGENT_TAG: "m"}


async def test_a_caller_tag_wins_over_the_agent_tag() -> None:
    """Read precedence mirrors ``Agent._merge_tags``: the caller's key wins."""
    t0 = _listing_t0()
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    await app.conversations(t, tags={AGENT_TAG: "neighbour", "customer_id": "c1"})

    assert _tag_filters(t0.list_inferences_calls[0]["request"]) == {
        AGENT_TAG: "neighbour",
        "customer_id": "c1",
    }


async def test_the_listing_filter_matches_the_tags_the_agent_writes() -> None:
    """The read filter is checked against a real write, not against a copy.

    Both sides read ``AGENT_TAG``; a literal on either would drift with no
    test noticing, which is how the listing came to filter on nothing.
    """
    t0 = _listing_t0(turns=1)
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    await t("hi")
    await app.conversations(t)

    written = t0.calls[0]["tags"]
    read = _tag_filters(t0.list_inferences_calls[0]["request"])
    assert read[AGENT_TAG] == written[AGENT_TAG]


async def test_hydrate_targets_the_resolved_function() -> None:
    stored = _stored_chat("ep", [_user_text_msg("hi")], "hello")
    t0 = FakeT0(
        [], list_inferences_responses=[FakeListInferencesResponse(inferences=[stored])]
    )
    app = App(t0=cast(AsyncTensorZeroGateway, t0))

    @app.agent("t")
    async def t(q: str) -> str:  # type: ignore[empty-body]
        """T."""

    await t.conversation(id="ep", hydrate=True).hydrate()

    assert t0.list_inferences_calls[0]["request"].function_name == "agent"


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
