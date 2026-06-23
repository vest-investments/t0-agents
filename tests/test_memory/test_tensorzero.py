"""Unit tests for :class:`TensorZeroMemory`.

Offline by design: every TZ gateway interaction is routed through
``FakeT0.list_inferences`` so tests run without Docker.  The Docker-
required end-to-end coverage lives in ``test_tensorzero_integration.py``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceFilterAnd,
    InferenceFilterTag,
    Text,
    Thought,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputText,
    InferenceFilterTime,
    StoredInferenceChat,
    StoredInferenceJson,
    StoredInput,
    StoredInputMessage,
    StoredInputMessageContentRawText,
    StoredInputMessageContentTemplate,
    StoredInputMessageContentText,
    StoredInputMessageContentThought,
    StoredInputMessageContentToolCall,
    StoredInputMessageContentToolResult,
)

from t0_agents.core.types import MemoryProvider
from t0_agents.memory import TensorZeroMemory
from t0_agents.memory.base import MemoryRecord
from tests._fake_t0 import FakeListInferencesResponse, FakeT0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_text_msg(text: str) -> StoredInputMessage:
    return StoredInputMessage(
        role="user",
        content=[StoredInputMessageContentText(text=text)],
    )


def _user_raw_text_msg(value: str) -> StoredInputMessage:
    return StoredInputMessage(
        role="user",
        content=[StoredInputMessageContentRawText(value=value)],
    )


def _assistant_text_msg(text: str) -> StoredInputMessage:
    return StoredInputMessage(
        role="assistant",
        content=[StoredInputMessageContentText(text=text)],
    )


@dataclass
class _StoredOutputThought:
    """Minimal stand-in for a stored ``thought``-typed chat output block.

    The real ``ContentBlockChatOutputThought`` is matched structurally by
    ``_output_to_content_blocks`` via ``getattr(item, "type", ...)`` so this
    duck-typed shape exercises the same code path without the SDK type.
    """

    text: str = "thinking..."
    type: str = "thought"
    signature: str | None = None
    summary: list[Any] | None = None
    provider_type: str | None = None


def _stored_chat(
    *,
    user_messages: list[StoredInputMessage] | None = None,
    output_text: str | None = "ok",
    inference_id: str | None = None,
    episode_id: str | None = None,
    timestamp: str = "2026-05-01T10:00:00Z",
    function_name: str = "agent",
    output: list[Any] | None = None,
) -> StoredInferenceChat:
    """Build a minimal :class:`StoredInferenceChat` for tests."""
    if output is None:
        output = (
            [ContentBlockChatOutputText(text=output_text)]
            if output_text is not None
            else []
        )
    return StoredInferenceChat(
        episode_id=episode_id or str(uuid4()),
        function_name=function_name,
        inference_id=inference_id or str(uuid4()),
        timestamp=timestamp,
        variant_name="v1",
        input=StoredInput(messages=user_messages or [_user_text_msg("hi")]),
        output=output,
    )


def _make_memory(
    t0: FakeT0,
    **kwargs: Any,
) -> TensorZeroMemory:
    return TensorZeroMemory(cast(AsyncTensorZeroGateway, t0), **kwargs)


# ---------------------------------------------------------------------------
# Scenario 1 — empty episode
# ---------------------------------------------------------------------------


class TestEmptyEpisode:
    async def test_load_returns_empty_list(self) -> None:
        t0 = FakeT0([], list_inferences_responses=[FakeListInferencesResponse()])
        mem = _make_memory(t0)
        assert await mem.load("ep-unknown") == []

    async def test_load_history_returns_none(self) -> None:
        t0 = FakeT0([], list_inferences_responses=[FakeListInferencesResponse()])
        mem = _make_memory(t0)
        assert await mem.load_history("ep-unknown") is None


# ---------------------------------------------------------------------------
# Scenario 2 — one inference, one record
# ---------------------------------------------------------------------------


class TestSingleInferenceRecord:
    async def test_load_extracts_user_message_and_response(self) -> None:
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_text_msg("Mi nombre es Ana")],
            output_text="Hola Ana",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        records = await mem.load(ep)
        assert len(records) == 1
        rec = records[0]
        assert rec.episode_id == ep
        assert rec.user_message == "Mi nombre es Ana"
        assert rec.agent_response == "Hola Ana"
        assert rec.turns == 1
        assert rec.inference_ids == [str(inf.inference_id)]

    async def test_timestamp_z_suffix_parsed_to_utc(self) -> None:
        inf = _stored_chat(timestamp="2026-05-01T02:20:42Z")
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load("ep"))[0]
        assert rec.created_at == dt.datetime(
            2026, 5, 1, 2, 20, 42, tzinfo=dt.UTC,
        )


# ---------------------------------------------------------------------------
# Scenario 3 — three inferences, reversal to oldest-first
# ---------------------------------------------------------------------------


class TestOrdering:
    async def test_load_reverses_desc_to_oldest_first(self) -> None:
        ep = str(uuid4())
        # Gateway returns DESC by timestamp.
        infs = [
            _stored_chat(
                user_messages=[_user_text_msg("third")],
                output_text="r3",
                episode_id=ep,
                timestamp="2026-05-01T12:00:00Z",
            ),
            _stored_chat(
                user_messages=[_user_text_msg("second")],
                output_text="r2",
                episode_id=ep,
                timestamp="2026-05-01T11:00:00Z",
            ),
            _stored_chat(
                user_messages=[_user_text_msg("first")],
                output_text="r1",
                episode_id=ep,
                timestamp="2026-05-01T10:00:00Z",
            ),
        ]
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=infs)],
        )
        mem = _make_memory(t0)
        records = await mem.load(ep)
        assert [r.user_message for r in records] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Scenario 4 — pure-tool-call inference is skipped
# ---------------------------------------------------------------------------


class TestSkipping:
    async def test_pure_tool_call_inference_skipped(self) -> None:
        ep = str(uuid4())
        no_text = _stored_chat(output=[], episode_id=ep)
        with_text = _stored_chat(output_text="real reply", episode_id=ep)
        t0 = FakeT0(
            [],
            list_inferences_responses=[
                FakeListInferencesResponse(inferences=[no_text, with_text])
            ],
        )
        mem = _make_memory(t0)
        records = await mem.load(ep)
        assert len(records) == 1
        assert records[0].agent_response == "real reply"

    async def test_json_inference_skipped(self) -> None:
        ep = str(uuid4())
        chat_inf = _stored_chat(episode_id=ep, output_text="text answer")
        json_inf = StoredInferenceJson(
            episode_id=ep,
            function_name="agent",
            inference_id=str(uuid4()),
            timestamp="2026-05-01T10:00:00Z",
            variant_name="v1",
            input=StoredInput(messages=[_user_text_msg("hi")]),
            output=None,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[
                FakeListInferencesResponse(inferences=[chat_inf, json_inf])
            ],
        )
        mem = _make_memory(t0)
        records = await mem.load(ep)
        assert len(records) == 1
        assert records[0].agent_response == "text answer"


# ---------------------------------------------------------------------------
# Scenario 5 — multi-message input + raw_text + tool_result skipping
# ---------------------------------------------------------------------------


class TestExtractUserMessage:
    async def test_skips_tool_result_returns_real_user_message(self) -> None:
        """Loop intermediate tool_result messages must not shadow the user
        question when extracting the last user-text from input."""
        ep = str(uuid4())
        msgs = [
            _user_text_msg("How's the weather?"),
            StoredInputMessage(
                role="assistant",
                content=[
                    StoredInputMessageContentToolCall(
                        id="c1",
                        name="get_weather",
                        arguments='{"city": "Tokyo"}',
                    )
                ],
            ),
            StoredInputMessage(
                role="user",
                content=[
                    StoredInputMessageContentToolResult(
                        id="c1",
                        name="get_weather",
                        result="Sunny",
                    )
                ],
            ),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="It's sunny in Tokyo.",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load(ep))[0]
        assert rec.user_message == "How's the weather?"

    async def test_raw_text_block_extraction(self) -> None:
        """``RawText`` user messages count as text for the extractor."""
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_raw_text_msg("inline raw user text")],
            output_text="ack",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load(ep))[0]
        assert rec.user_message == "inline raw user text"

    async def test_string_content_path(self) -> None:
        """Some loop paths add ``Message(role='user', content='string')``;
        TZ may persist it as a bare string on input.messages.content.

        The extractor must handle that branch too — exercises the
        ``isinstance(content, str)`` fallback in
        ``_extract_last_user_message``.
        """
        ep = str(uuid4())
        msg = StoredInputMessage(role="user", content=[])
        # Force the string fallback by patching the content attr.
        object.__setattr__(msg, "content", "plain string user message")
        inf = _stored_chat(user_messages=[msg], output_text="ok", episode_id=ep)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load(ep))[0]
        assert rec.user_message == "plain string user message"

    async def test_no_user_text_returns_empty(self) -> None:
        """When no message has any text/raw_text block, user_message is ''."""
        ep = str(uuid4())
        msg = StoredInputMessage(
            role="user",
            content=[
                StoredInputMessageContentToolResult(id="c", name="t", result="r")
            ],
        )
        inf = _stored_chat(user_messages=[msg], output_text="ok", episode_id=ep)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load(ep))[0]
        assert rec.user_message == ""


# ---------------------------------------------------------------------------
# Scenario 6 — filter construction
# ---------------------------------------------------------------------------


class TestFilterConstruction:
    async def test_no_filter_when_neither_set(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0)
        await mem.load("ep")
        request = t0.list_inferences_calls[0]["request"]
        assert request.filters is None

    async def test_single_tag_filter(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0, tag_filter={"customer_id": "alice"})
        await mem.load("ep")
        filt = t0.list_inferences_calls[0]["request"].filters
        assert isinstance(filt, InferenceFilterTag)
        assert filt.key == "customer_id"
        assert filt.value == "alice"
        assert filt.comparison_operator == "="

    async def test_multi_tag_combines_with_and(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(
            t0, tag_filter={"customer_id": "alice", "env": "prod"}
        )
        await mem.load("ep")
        filt = t0.list_inferences_calls[0]["request"].filters
        assert isinstance(filt, InferenceFilterAnd)
        keys = sorted(c.key for c in filt.children if isinstance(c, InferenceFilterTag))
        assert keys == ["customer_id", "env"]

    async def test_extra_filters_passthrough_combines_with_tags(self) -> None:
        time_filter = InferenceFilterTime(
            comparison_operator=">=",
            time="2026-01-01T00:00:00Z",
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(
            t0,
            tag_filter={"customer_id": "alice"},
            extra_filters=time_filter,
        )
        await mem.load("ep")
        filt = t0.list_inferences_calls[0]["request"].filters
        assert isinstance(filt, InferenceFilterAnd)
        # Children: the tag filter + the time filter.
        assert len(filt.children) == 2
        types = {type(c) for c in filt.children}
        assert InferenceFilterTag in types
        assert InferenceFilterTime in types

    async def test_extra_filters_only_no_wrap(self) -> None:
        time_filter = InferenceFilterTime(
            comparison_operator="<",
            time="2026-12-31T00:00:00Z",
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0, extra_filters=time_filter)
        await mem.load("ep")
        filt = t0.list_inferences_calls[0]["request"].filters
        # Single non-tag filter passes through verbatim.
        assert filt is time_filter


# ---------------------------------------------------------------------------
# Scenario 7 — request shape: function_name, load_limit
# ---------------------------------------------------------------------------


class TestRequestShape:
    async def test_function_name_forwarded_when_set(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0, function_name="chat")
        await mem.load("ep")
        assert t0.list_inferences_calls[0]["request"].function_name == "chat"

    async def test_function_name_none_omitted(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0)
        await mem.load("ep")
        assert t0.list_inferences_calls[0]["request"].function_name is None

    async def test_load_limit_forwarded(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0, load_limit=42)
        await mem.load("ep")
        assert t0.list_inferences_calls[0]["request"].limit == 42

    async def test_load_limit_default_is_ten(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0)
        await mem.load("ep")
        assert t0.list_inferences_calls[0]["request"].limit == 10


# ---------------------------------------------------------------------------
# Scenario 8 — search() default + experimental opt-in
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_returns_empty_by_default(self) -> None:
        t0 = FakeT0([])  # no list_inferences queue — must NOT be called.
        mem = _make_memory(t0)
        assert await mem.search("anything") == []
        assert t0.list_inferences_calls == []

    async def test_search_experimental_routes_query(self) -> None:
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_text_msg("Mi color es teal")],
            output_text="Anotado",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0, experimental_search=True)
        results = await mem.search("teal", limit=3)
        assert len(results) == 1
        request = t0.list_inferences_calls[0]["request"]
        assert request.search_query_experimental == "teal"
        assert request.limit == 3

    async def test_search_experimental_skips_pure_tool_call(self) -> None:
        inf_skip = _stored_chat(output=[])
        inf_keep = _stored_chat(output_text="real")
        t0 = FakeT0(
            [],
            list_inferences_responses=[
                FakeListInferencesResponse(inferences=[inf_skip, inf_keep])
            ],
        )
        mem = _make_memory(t0, experimental_search=True)
        results = await mem.search("anything")
        assert len(results) == 1

    async def test_search_experimental_skips_json_inferences(self) -> None:
        chat_inf = _stored_chat(output_text="real")
        json_inf = StoredInferenceJson(
            episode_id=str(uuid4()),
            function_name="agent",
            inference_id=str(uuid4()),
            timestamp="2026-05-01T10:00:00Z",
            variant_name="v1",
            input=StoredInput(messages=[_user_text_msg("hi")]),
            output=None,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[
                FakeListInferencesResponse(inferences=[chat_inf, json_inf])
            ],
        )
        mem = _make_memory(t0, experimental_search=True)
        results = await mem.search("anything")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Scenario 9 — save() is a no-op
# ---------------------------------------------------------------------------


class TestSaveNoop:
    async def test_save_does_not_call_gateway(self) -> None:
        t0 = FakeT0([])  # no queues — must not be touched
        mem = _make_memory(t0)
        await mem.save(
            MemoryRecord(
                episode_id="ep",
                user_message="u",
                agent_response="r",
                turns=1,
                inference_ids=[str(uuid4())],
            )
        )
        assert t0.calls == []
        assert t0.feedback_calls == []
        assert t0.list_inferences_calls == []


# ---------------------------------------------------------------------------
# Scenario 10 — load_history happy path + None
# ---------------------------------------------------------------------------


class TestLoadHistory:
    async def test_returns_native_messages_with_assistant_appended(self) -> None:
        ep = str(uuid4())
        msgs = [
            _user_text_msg("Hello"),
            _assistant_text_msg("Hi there"),
            _user_text_msg("How are you?"),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="I'm fine.",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"
        # Last is the assistant reply derived from output.
        last_content = history[3]["content"]
        assert isinstance(last_content, list)
        assert any(isinstance(b, Text) and b.text == "I'm fine." for b in last_content)

    async def test_returns_none_when_no_inferences(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0)
        assert await mem.load_history("ep-empty") is None

    async def test_uses_limit_one_for_history_query(self) -> None:
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse()],
        )
        mem = _make_memory(t0)
        await mem.load_history("ep")
        assert t0.list_inferences_calls[0]["request"].limit == 1

    async def test_skips_unsupported_blocks_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ep = str(uuid4())
        msgs = [
            StoredInputMessage(
                role="user",
                content=[
                    StoredInputMessageContentText(text="hello"),
                    StoredInputMessageContentTemplate(
                        arguments={"x": 1}, name="my_template",
                    ),
                ],
            ),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="hi",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        with caplog.at_level("WARNING", logger="t0_agents.memory.tensorzero"):
            history = await mem.load_history(ep)
        assert history is not None
        # Text block kept; template skipped.
        first_content = history[0]["content"]
        assert isinstance(first_content, list)
        assert len(first_content) == 1
        assert isinstance(first_content[0], Text)
        # Single warning emitted enumerating skipped kinds.
        warnings = [r for r in caplog.records if "skipped" in r.message]
        assert len(warnings) == 1
        assert "template" in warnings[0].message

    async def test_message_with_only_unsupported_blocks_dropped(self) -> None:
        ep = str(uuid4())
        msgs = [
            StoredInputMessage(
                role="user",
                content=[
                    StoredInputMessageContentTemplate(
                        arguments={"x": 1}, name="my_template",
                    )
                ],
            ),
            _user_text_msg("real question"),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="ok",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        # The template-only message is dropped, the text-only message kept,
        # plus the assistant from output.
        assert len(history) == 2
        assert history[0]["role"] == "user"
        first_content = history[0]["content"]
        assert isinstance(first_content, list)
        assert isinstance(first_content[0], Text)
        assert first_content[0].text == "real question"

    async def test_tool_call_and_tool_result_round_trip(self) -> None:
        """ToolCall + ToolResult must survive the round-trip so the model
        can see its own prior reasoning when memory is replayed."""
        ep = str(uuid4())
        msgs = [
            _user_text_msg("What's the weather?"),
            StoredInputMessage(
                role="assistant",
                content=[
                    StoredInputMessageContentToolCall(
                        id="c1",
                        name="get_weather",
                        arguments='{"city": "Tokyo"}',
                    )
                ],
            ),
            StoredInputMessage(
                role="user",
                content=[
                    StoredInputMessageContentToolResult(
                        id="c1",
                        name="get_weather",
                        result="Sunny",
                    )
                ],
            ),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="It's sunny.",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assert len(history) == 4
        # Assistant turn carries the ToolCall as a runtime block.
        assistant_content = history[1]["content"]
        assert isinstance(assistant_content, list)
        from tensorzero import ToolCall, ToolResult

        assert isinstance(assistant_content[0], ToolCall)
        assert assistant_content[0].id == "c1"
        # The parsed-arguments dict should have round-tripped from the JSON
        # string TZ stores.
        assert assistant_content[0].arguments == {"city": "Tokyo"}
        # User-after-tool turn carries the ToolResult.
        user_after = history[2]["content"]
        assert isinstance(user_after, list)
        assert isinstance(user_after[0], ToolResult)
        assert user_after[0].result == "Sunny"

    async def test_tool_call_with_invalid_json_arguments(self) -> None:
        """Malformed JSON arguments leave ``arguments=None`` but
        preserve ``raw_arguments`` so the model can still see what
        the prior turn emitted."""
        ep = str(uuid4())
        msgs = [
            _user_text_msg("go"),
            StoredInputMessage(
                role="assistant",
                content=[
                    StoredInputMessageContentToolCall(
                        id="c2",
                        name="t",
                        arguments="{not valid json",
                    )
                ],
            ),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output_text="ok",
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        from tensorzero import ToolCall

        tc = history[1]["content"][0]
        assert isinstance(tc, ToolCall)
        assert tc.arguments is None
        assert tc.raw_arguments == "{not valid json"

    async def test_replay_strips_all_thoughts_depth_invariant(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AC1: a deep multi-turn history with thought blocks at every turn —
        including a thought-ONLY assistant turn in the MIDDLE — replays with
        zero thought blocks regardless of depth, never emits an empty-content
        message, leaves the stored record untouched, and warns about nothing
        (the strip is by design, not a decode failure)."""
        ep = str(uuid4())
        # Three exchanges. The middle assistant turn is thought-ONLY (the
        # budget-exhausted "blank" shape) and must be dropped, not emitted as
        # an empty-content assistant message that would break role validity.
        msgs = [
            _user_text_msg("turn 1 question"),
            StoredInputMessage(
                role="assistant",
                content=[
                    StoredInputMessageContentThought(text="t1 reasoning"),
                    StoredInputMessageContentText(text="turn 1 answer"),
                ],
            ),
            _user_text_msg("turn 2 question"),
            StoredInputMessage(
                role="assistant",
                content=[StoredInputMessageContentThought(text="t2 reasoning only")],
            ),
            _user_text_msg("turn 3 question"),
        ]
        inf = _stored_chat(
            user_messages=msgs,
            output=[
                _StoredOutputThought(text="final reasoning"),
                ContentBlockChatOutputText(text="turn 3 answer"),
            ],
            episode_id=ep,
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        with caplog.at_level("WARNING", logger="t0_agents.memory.tensorzero"):
            history = await mem.load_history(ep)
        assert history is not None
        all_blocks = [b for m in history for b in m["content"]]
        # Zero thought blocks anywhere in the replayed history.
        assert not any(isinstance(b, Thought) for b in all_blocks)
        # No empty-content message survives — the thought-only middle turn was
        # dropped, leaving roles user, assistant, user, user, assistant.
        assert all(m["content"] for m in history)
        assert [m["role"] for m in history] == [
            "user",
            "assistant",
            "user",
            "user",
            "assistant",
        ]
        # Text content fully preserved across depth (token count does not grow
        # with conversation depth from re-emitted thoughts).
        assert [b.text for b in all_blocks if isinstance(b, Text)] == [
            "turn 1 question",
            "turn 1 answer",
            "turn 2 question",
            "turn 3 question",
            "turn 3 answer",
        ]
        # Store path intact: every stored thought is still on the
        # StoredInference (two on input, one on output), unmutated.
        stored_input_thoughts = [
            b
            for m in inf.input.messages
            for b in m.content
            if isinstance(b, StoredInputMessageContentThought)
        ]
        assert len(stored_input_thoughts) == 2
        assert any(getattr(b, "type", None) == "thought" for b in inf.output)
        # The strip is silent — no "skipped unsupported kind" warning fires.
        assert not any("skipped" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Scenario 11 — Protocol conformance + Agent acceptance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    async def test_isinstance_memory_provider(self) -> None:
        t0 = FakeT0([])
        mem = _make_memory(t0)
        assert isinstance(mem, MemoryProvider)

    async def test_load_history_method_exists(self) -> None:
        t0 = FakeT0([])
        mem = _make_memory(t0)
        assert hasattr(mem, "load_history")

    async def test_agent_accepts_with_messages_strategy(self) -> None:
        from t0_agents.core.config import AgentConfig

        t0 = FakeT0([])
        mem = _make_memory(t0)
        # Construction must not raise.
        cfg = AgentConfig(
            name="t",
            function_name="agent",
            memory=mem,
            memory_strategy="messages",
        )
        assert cfg.memory is mem


# ---------------------------------------------------------------------------
# Scenario 12 — timestamp parsing branches
# ---------------------------------------------------------------------------


class TestTimestamp:
    async def test_datetime_with_tz_passes_through(self) -> None:
        ts = dt.datetime(2026, 5, 1, 10, 0, 0, tzinfo=dt.UTC)
        inf = _stored_chat()
        # Force a datetime instead of a string (the spec allows both).
        object.__setattr__(inf, "timestamp", ts)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load("ep"))[0]
        assert rec.created_at == ts

    async def test_datetime_naive_is_assumed_utc(self) -> None:
        ts = dt.datetime(2026, 5, 1, 10, 0, 0)
        inf = _stored_chat()
        object.__setattr__(inf, "timestamp", ts)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load("ep"))[0]
        assert rec.created_at.tzinfo == dt.UTC

    async def test_unknown_timestamp_type_passthrough(self) -> None:
        """Non-datetime, non-string timestamps fall through unchanged.

        Defensive: should never happen against a real gateway, but the
        helper handles it via cast() so the run does not crash.
        """
        inf = _stored_chat()
        # Bypass dataclass validation with object.__setattr__.
        sentinel = object()
        object.__setattr__(inf, "timestamp", sentinel)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load("ep"))[0]
        assert rec.created_at is sentinel  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# Scenario 13 — exhaustive helper branch coverage
# ---------------------------------------------------------------------------


class TestHelperBranches:
    """Exercise the private helpers' rarely-hit fallback branches."""

    async def test_extract_agent_text_non_list_returns_empty(self) -> None:
        inf = _stored_chat()
        object.__setattr__(inf, "output", "not-a-list")
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        # No agent_response → record skipped → empty list returned.
        assert await mem.load("ep") == []

    async def test_extract_agent_text_duck_typed_text_block(self) -> None:
        """Duck-typed block carrying ``type='text'`` and ``text=str`` is
        extracted alongside the strict ContentBlockChatOutputText path."""
        from dataclasses import dataclass

        @dataclass
        class _DuckText:
            text: str = "duck-typed reply"
            type: str = "text"

        inf = _stored_chat()
        object.__setattr__(inf, "output", [_DuckText()])
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        rec = (await mem.load("ep"))[0]
        assert rec.agent_response == "duck-typed reply"

    async def test_to_native_message_raw_text_round_trip(self) -> None:
        """Raw-text content blocks survive ``load_history`` as RawText."""
        from tensorzero import RawText

        ep = str(uuid4())
        msgs = [_user_raw_text_msg("raw inline")]
        inf = _stored_chat(user_messages=msgs, output_text="ok", episode_id=ep)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        first_content = history[0]["content"]
        assert isinstance(first_content, list)
        assert isinstance(first_content[0], RawText)
        assert first_content[0].value == "raw inline"

    async def test_input_thought_block_stripped_on_replay(self) -> None:
        """Input-side ``Thought`` blocks are dropped on replay; the
        surrounding ``Text`` survives and the stored record is unmutated."""
        ep = str(uuid4())
        stored_msg = StoredInputMessage(
            role="assistant",
            content=[
                StoredInputMessageContentThought(text="reasoning step"),
                StoredInputMessageContentText(text="visible answer"),
            ],
        )
        inf = _stored_chat(
            user_messages=[stored_msg], output_text="ok", episode_id=ep
        )
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        first_content = history[0]["content"]
        assert isinstance(first_content, list)
        # Thought stripped, Text kept.
        assert not any(isinstance(b, Thought) for b in first_content)
        assert [b.text for b in first_content if isinstance(b, Text)] == [
            "visible answer"
        ]
        # Store path intact: the StoredInference still carries the thought.
        assert any(
            isinstance(b, StoredInputMessageContentThought)
            for b in inf.input.messages[0].content
        )

    async def test_output_to_content_blocks_non_list_returns_empty(self) -> None:
        """When output is not a list, load_history still appends nothing.

        Edge case: if a malformed inference comes back with output=None
        but the chat-type discriminator, we drop the assistant turn and
        return only the user history.
        """
        ep = str(uuid4())
        msgs = [_user_text_msg("hi")]
        inf = _stored_chat(user_messages=msgs, output=[], episode_id=ep)
        # Force output to a non-list to exercise the early-return branch.
        object.__setattr__(inf, "output", None)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        # Only the user turn, no assistant from output.
        assert len(history) == 1
        assert history[0]["role"] == "user"

    async def test_output_to_content_blocks_tool_call(self) -> None:
        """Output-side ToolCall blocks are converted to runtime ToolCall."""
        from dataclasses import dataclass

        from tensorzero import ToolCall

        @dataclass
        class _OutputToolCall:
            arguments: str = '{"x": 1}'
            id: str = "out-tc-1"
            name: str = "do_thing"
            type: str = "tool_call"

        ep = str(uuid4())
        inf = _stored_chat(episode_id=ep)
        object.__setattr__(inf, "output", [_OutputToolCall()])
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assistant = history[-1]
        assert assistant["role"] == "assistant"
        block = assistant["content"][0]
        assert isinstance(block, ToolCall)
        assert block.arguments == {"x": 1}
        assert block.name == "do_thing"

    async def test_output_to_content_blocks_tool_call_invalid_json(self) -> None:
        """Output-side ToolCall with non-JSON arguments → arguments=None,
        raw_arguments preserved."""
        from dataclasses import dataclass

        from tensorzero import ToolCall

        @dataclass
        class _OutputToolCall:
            arguments: str = "not-json"
            id: str = "out-tc-2"
            name: str = "do_thing"
            type: str = "tool_call"

        ep = str(uuid4())
        inf = _stored_chat(episode_id=ep)
        object.__setattr__(inf, "output", [_OutputToolCall()])
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        block = history[-1]["content"][0]
        assert isinstance(block, ToolCall)
        assert block.arguments is None
        assert block.raw_arguments == "not-json"

    async def test_output_thought_block_stripped_on_replay(self) -> None:
        """Output-side ``Thought`` blocks are dropped on replay; ``Text``
        survives and the stored output is left untouched."""
        ep = str(uuid4())
        inf = _stored_chat(episode_id=ep)
        output_blocks = [
            _StoredOutputThought(),
            ContentBlockChatOutputText(text="final"),
        ]
        object.__setattr__(inf, "output", output_blocks)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assistant_blocks = history[-1]["content"]
        kinds = [type(b).__name__ for b in assistant_blocks]
        assert "Thought" not in kinds
        assert "Text" in kinds
        # Store path intact: the raw output still carries the thought block.
        assert any(getattr(b, "type", None) == "thought" for b in inf.output)

    async def test_to_native_message_unknown_block_skipped(self) -> None:
        """Unknown content type → skipped + recorded for the warning."""
        from dataclasses import dataclass

        @dataclass
        class _UnknownBlock:
            type: str = "exotic"

        ep = str(uuid4())
        msgs = [
            StoredInputMessage(
                role="user",
                content=cast(list[Any], [_UnknownBlock()]),
            ),
            _user_text_msg("real"),
        ]
        inf = _stored_chat(user_messages=msgs, output_text="ok", episode_id=ep)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        # The exotic-only message is dropped; the text-only message survives.
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        first_content = history[0]["content"]
        assert isinstance(first_content, list)
        assert isinstance(first_content[0], Text)
