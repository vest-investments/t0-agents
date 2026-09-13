"""Unit tests for :class:`TensorZeroMemory`.

Offline by design: every TZ gateway interaction is routed through
``FakeT0.list_inferences`` so tests run without Docker.  The Docker-
required end-to-end coverage lives in ``test_tensorzero_integration.py``.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool
from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceFilterAnd,
    InferenceFilterTag,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputText,
    ContentBlockChatOutputToolCall,
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
from t0_agents.mcp.loop import run_agent
from t0_agents.memory import TensorZeroMemory
from t0_agents.memory.base import MemoryRecord
from t0_agents.memory.tensorzero import _output_to_content_blocks
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeListInferencesResponse, FakeT0

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


def _lookup_tool() -> MCPTool:
    return MCPTool(
        name="lookup",
        description="lookup tool",
        inputSchema={"type": "object", "properties": {}},
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
# Scenario 10b — load_transcript: the reader's view of an episode
# ---------------------------------------------------------------------------


def _t0_with(inferences: list[Any]) -> FakeT0:
    return FakeT0(
        [], list_inferences_responses=[FakeListInferencesResponse(inferences=inferences)]
    )


def _round(
    episode_id: str,
    seq: int,
    *,
    input_messages: list[StoredInputMessage],
    output: list[Any],
    timestamp: str = "2026-09-05T10:00:00Z",
) -> StoredInferenceChat:
    """One stored round of an episode; ``seq`` orders its UUIDv7-shaped id."""
    return _stored_chat(
        user_messages=input_messages,
        output=output,
        episode_id=episode_id,
        inference_id=f"0199a000-0000-7000-8000-{seq:012d}",
        timestamp=timestamp,
    )


def _stored_call(
    call_id: str, name: str, arguments: dict[str, Any]
) -> ContentBlockChatOutputToolCall:
    return ContentBlockChatOutputToolCall(
        id=call_id,
        raw_arguments=json.dumps(arguments),
        raw_name=name,
        arguments=arguments,
        name=name,
    )


def _assistant_msg(*content: Any) -> StoredInputMessage:
    return StoredInputMessage(role="assistant", content=list(content))


def _tool_result_msg(call_id: str, name: str, result: str) -> StoredInputMessage:
    return StoredInputMessage(
        role="user",
        content=[StoredInputMessageContentToolResult(id=call_id, name=name, result=result)],
    )


def _texts(transcript: list[Any]) -> list[str]:
    return [b.text for m in transcript for b in m["content"] if isinstance(b, Text)]


class TestLoadTranscript:
    async def test_a_terminal_ended_episode_keeps_its_render_call_for_a_reader(
        self,
    ) -> None:
        """The two readers part on exactly one block.

        ``load_history`` must drop the dangling call — the model would be
        replayed a call no result ever answered.  A reader was shown what
        that call rendered, so ``load_transcript`` keeps it, arguments and
        all.
        """
        ep = str(uuid4())
        inf = _round(
            ep,
            1,
            input_messages=[_user_text_msg("show me the mix")],
            output=[
                ContentBlockChatOutputText(text="Here is the mix."),
                _stored_call("r1", "render_chart", {"series": [1, 2]}),
            ],
        )

        history = await _make_memory(_t0_with([inf])).load_history(ep)
        transcript = await _make_memory(_t0_with([inf])).load_transcript(ep)

        assert history is not None and transcript is not None
        assert [type(b).__name__ for b in history[-1]["content"]] == ["Text"]
        assert [m["role"] for m in transcript] == ["user", "assistant"]
        reply = transcript[-1]["content"]
        assert [type(b).__name__ for b in reply] == ["Text", "ToolCall"]
        assert (reply[1].id, reply[1].name, reply[1].arguments) == (
            "r1",
            "render_chart",
            {"series": [1, 2]},
        )

    async def test_the_transcript_recovers_a_call_the_model_history_lost(self) -> None:
        """A later turn carries the earlier reply forward WITHOUT its
        dangling call — that is what the model was replayed — so a
        one-row reader can never see the call again.  Reading each round's
        own output does.
        """
        ep = str(uuid4())
        ask = _user_text_msg("show me the mix")
        first = _round(
            ep,
            1,
            input_messages=[ask],
            output=[
                ContentBlockChatOutputText(text="Here is the mix."),
                _stored_call("r1", "render_chart", {"series": [1]}),
            ],
        )
        second = _round(
            ep,
            2,
            input_messages=[
                ask,
                _assistant_text_msg("Here is the mix."),
                _user_text_msg("and the total?"),
            ],
            output=[ContentBlockChatOutputText(text="Twelve.")],
        )

        history = await _make_memory(_t0_with([second, first])).load_history(ep)
        transcript = await _make_memory(_t0_with([second, first])).load_transcript(ep)

        assert history is not None and transcript is not None
        assert not any(isinstance(b, ToolCall) for m in history for b in m["content"])
        assert [m["role"] for m in transcript] == ["user", "assistant", "user", "assistant"]
        assert [type(b).__name__ for b in transcript[1]["content"]] == ["Text", "ToolCall"]
        assert transcript[1]["content"][1].id == "r1"
        assert _texts(transcript) == [
            "show me the mix",
            "Here is the mix.",
            "and the total?",
            "Twelve.",
        ]

    async def test_the_transcript_carries_each_rounds_results_before_its_reply(self) -> None:
        """A round whose input ends on tool results is a later step of the
        same turn: the results are emitted as the ``user`` message the
        gateway stored them in, then the reply, and no user text is
        re-emitted — a reader that derived something from a result can
        derive it again.
        """
        ep = str(uuid4())
        ask = _user_text_msg("what's the weather?")
        planning = _round(
            ep,
            1,
            input_messages=[ask],
            output=[
                ContentBlockChatOutputText(text="Let me check."),
                _stored_call("c1", "lookup", {"city": "Tokyo"}),
            ],
        )
        carried = [
            ask,
            _assistant_msg(
                StoredInputMessageContentText(text="Let me check."),
                StoredInputMessageContentToolCall(
                    id="c1", name="lookup", arguments='{"city": "Tokyo"}'
                ),
            ),
            _tool_result_msg("c1", "lookup", "Sunny"),
        ]
        answer = _round(
            ep,
            2,
            input_messages=carried,
            output=[ContentBlockChatOutputText(text="Sunny in Tokyo.")],
        )
        thanks = _round(
            ep,
            3,
            input_messages=[
                *carried,
                _assistant_text_msg("Sunny in Tokyo."),
                _user_text_msg("thanks"),
            ],
            output=[ContentBlockChatOutputText(text="Any time.")],
        )

        transcript = await _make_memory(_t0_with([thanks, answer, planning])).load_transcript(ep)

        assert transcript is not None
        assert [m["role"] for m in transcript] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        (result,) = transcript[2]["content"]
        assert isinstance(result, ToolResult)
        assert (result.id, result.name, result.result) == ("c1", "lookup", "Sunny")
        assert _texts(transcript) == [
            "what's the weather?",
            "Let me check.",
            "Sunny in Tokyo.",
            "thanks",
            "Any time.",
        ]

    async def test_a_parallel_batch_of_results_stays_one_message(self) -> None:
        """Two calls answered in one round come back as the one ``user``
        message the gateway stored, blocks in stored order — the unit a
        reader groups a turn's results by.
        """
        ep = str(uuid4())
        ask = _user_text_msg("compare both")
        planning = _round(
            ep,
            1,
            input_messages=[ask],
            output=[
                _stored_call("c1", "lookup", {"city": "Tokyo"}),
                _stored_call("c2", "lookup", {"city": "Osaka"}),
            ],
        )
        results_in = [
            StoredInputMessageContentToolResult(id="c1", name="lookup", result="Sunny"),
            StoredInputMessageContentToolResult(id="c2", name="lookup", result="Rain"),
        ]
        answer = _round(
            ep,
            2,
            input_messages=[
                ask,
                _assistant_msg(
                    StoredInputMessageContentToolCall(
                        id="c1", name="lookup", arguments='{"city": "Tokyo"}'
                    ),
                    StoredInputMessageContentToolCall(
                        id="c2", name="lookup", arguments='{"city": "Osaka"}'
                    ),
                ),
                StoredInputMessage(role="user", content=results_in),
            ],
            output=[ContentBlockChatOutputText(text="Sunny in Tokyo, rain in Osaka.")],
        )

        transcript = await _make_memory(_t0_with([answer, planning])).load_transcript(ep)

        assert transcript is not None
        assert [m["role"] for m in transcript] == ["user", "assistant", "user", "assistant"]
        results = transcript[2]["content"]
        assert len(results) == 2
        assert [(b.id, b.result) for b in results if isinstance(b, ToolResult)] == [
            ("c1", "Sunny"),
            ("c2", "Rain"),
        ]

    async def test_the_transcript_orders_rounds_by_inference_id_within_a_second(
        self,
    ) -> None:
        """The gateway orders by ``timestamp`` at second granularity, so the
        rounds of one turn tie; the UUIDv7 id is the millisecond order, and
        the order the rows arrive in is not trusted.
        """
        ep = str(uuid4())
        ask = _user_text_msg("go")
        planning = _round(
            ep, 1, input_messages=[ask], output=[_stored_call("c1", "lookup", {})]
        )
        carried = [
            ask,
            _assistant_msg(
                StoredInputMessageContentToolCall(id="c1", name="lookup", arguments="{}")
            ),
            _tool_result_msg("c1", "lookup", "ok"),
        ]
        first = _round(
            ep, 2, input_messages=carried, output=[ContentBlockChatOutputText(text="first")]
        )
        second = _round(
            ep,
            3,
            input_messages=[*carried, _assistant_text_msg("first"), _user_text_msg("more")],
            output=[ContentBlockChatOutputText(text="second")],
        )

        transcript = await _make_memory(_t0_with([first, second, planning])).load_transcript(ep)

        assert transcript is not None
        assert _texts(transcript) == ["go", "first", "more", "second"]
        assert [type(b).__name__ for b in transcript[1]["content"]] == ["ToolCall"]

    async def test_the_transcript_never_carries_a_thought(self) -> None:
        """Reasoning is not something the user was shown.  Stored on both the
        input and the output side, none of it reaches the transcript, and
        the stored rows are left as they were.
        """
        ep = str(uuid4())
        first = _round(
            ep,
            1,
            input_messages=[_user_text_msg("q1")],
            output=[_StoredOutputThought(text="t1"), ContentBlockChatOutputText(text="a1")],
        )
        second = _round(
            ep,
            2,
            input_messages=[
                _user_text_msg("q1"),
                _assistant_msg(
                    StoredInputMessageContentThought(text="t1"),
                    StoredInputMessageContentText(text="a1"),
                ),
                _user_text_msg("q2"),
            ],
            output=[_StoredOutputThought(text="t2"), ContentBlockChatOutputText(text="a2")],
        )

        transcript = await _make_memory(_t0_with([second, first])).load_transcript(ep)

        assert transcript is not None
        assert not any(isinstance(b, Thought) for m in transcript for b in m["content"])
        assert _texts(transcript) == ["q1", "a1", "q2", "a2"]
        assert any(getattr(b, "type", None) == "thought" for b in first.output or [])

    async def test_the_transcript_reads_a_bounded_window(self) -> None:
        """One ``list_inferences`` call, newest first, ``window`` rows — the
        bound :meth:`list_conversations` already accepts, so a long episode
        costs what the caller chose and no more.
        """
        t0 = _t0_with([])
        await _make_memory(
            t0, function_name="chat", tag_filter={"tenant": "a"}
        ).load_transcript("ep")
        request = t0.list_inferences_calls[0]["request"]
        assert request.episode_id == "ep"
        assert request.function_name == "chat"
        assert isinstance(request.filters, InferenceFilterTag)
        assert request.limit == TensorZeroMemory.DEFAULT_LIST_WINDOW == 200
        assert [(o.by, o.direction) for o in request.order_by] == [
            ("timestamp", "descending")
        ]

        narrow = _t0_with([])
        await _make_memory(narrow).load_transcript("ep", window=25)
        assert narrow.list_inferences_calls[0]["request"].limit == 25

    async def test_the_transcript_is_none_for_an_empty_episode(self) -> None:
        """No chat inference — a JSON-only episode included — reads as
        ``None``, the same "nothing here" :meth:`load_history` answers, so
        a caller keeps one not-found branch for both readers.
        """
        assert await _make_memory(_t0_with([])).load_transcript("ep") is None
        json_only = StoredInferenceJson(
            episode_id="ep",
            function_name="agent",
            inference_id=str(uuid4()),
            timestamp="2026-09-05T10:00:00Z",
            variant_name="v1",
            input=StoredInput(messages=[_user_text_msg("hi")]),
            output=None,
        )
        assert await _make_memory(_t0_with([json_only])).load_transcript("ep") is None

    async def test_the_transcript_warns_once_about_unreplayable_blocks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ep = str(uuid4())
        opening = StoredInputMessage(
            role="user",
            content=[
                StoredInputMessageContentText(text="hello"),
                StoredInputMessageContentTemplate(arguments={"x": 1}, name="my_template"),
            ],
        )
        inf = _round(
            ep, 1, input_messages=[opening], output=[ContentBlockChatOutputText(text="hi")]
        )

        with caplog.at_level("WARNING", logger="t0_agents.memory.tensorzero"):
            transcript = await _make_memory(_t0_with([inf])).load_transcript(ep)

        assert transcript is not None
        assert _texts(transcript) == ["hello", "hi"]
        warnings = [r for r in caplog.records if "skipped" in r.message]
        assert len(warnings) == 1
        assert "load_transcript" in warnings[0].message
        assert "template" in warnings[0].message


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

    async def test_output_tool_call_replays_the_served_gateway_shape(self) -> None:
        """The converter must carry the model's raw JSON STRING to the wire.

        The gateway serves output-side calls as
        :class:`ContentBlockChatOutputToolCall` — ``arguments`` already
        PARSED (a dict) and the raw JSON string in ``raw_arguments``.
        The replayed input side (``ToolCallWrapper``, an untagged union)
        demands str in both variants, so a dict smuggled into
        ``raw_arguments`` 400s every subsequent message of the episode
        (the 2026-08-17 poisoned-history incident)."""
        stored = ContentBlockChatOutputToolCall(
            id="out-tc-1",
            # Deliberately NOT json.dumps-canonical (no space): the wire
            # must carry the model's literal string, not a re-serialization.
            raw_arguments='{"x":1}',
            raw_name="do_thing",
            arguments={"x": 1},
            name="do_thing",
        )
        blocks = _output_to_content_blocks([stored])
        assert len(blocks) == 1
        block = blocks[0]
        assert isinstance(block, ToolCall)
        assert block.raw_arguments == '{"x":1}'
        assert block.arguments == {"x": 1}
        assert block.name == "do_thing"
        assert block.raw_name == "do_thing"
        wire = block.to_dict()
        assert wire["raw_arguments"] == '{"x":1}'
        assert isinstance(wire["raw_arguments"], str)

    async def test_output_tool_call_invalid_call_replays_raw_fields(self) -> None:
        """An INVALID stored call keeps its raw fields on replay.

        When the model wrote unparseable arguments (or an unknown tool
        name) the gateway persists ``arguments=None`` / ``name=None`` but
        ALWAYS carries ``raw_arguments`` / ``raw_name`` — mirror those
        through instead of dropping them to empty strings."""
        stored = ContentBlockChatOutputToolCall(
            id="out-tc-2",
            raw_arguments="not-json",
            raw_name="do_thing",
            arguments=None,
            name=None,
        )
        blocks = _output_to_content_blocks([stored])
        block = blocks[0]
        assert isinstance(block, ToolCall)
        assert block.raw_arguments == "not-json"
        assert block.arguments is None
        assert block.raw_name == "do_thing"
        assert block.name is None

    async def test_output_tool_call_dict_only_arguments_are_serialized(self) -> None:
        """Defensive fallback: a dict-only block is serialized, not passed raw.

        No known gateway shape lacks ``raw_arguments``, but if one ever
        does, the wire still demands a string — ``json.dumps`` the parsed
        dict rather than emit it verbatim."""
        from dataclasses import dataclass, field

        @dataclass
        class _DictOnlyToolCall:
            arguments: dict[str, Any] = field(default_factory=lambda: {"x": 1})
            id: str = "out-tc-3"
            name: str = "do_thing"
            type: str = "tool_call"

        blocks = _output_to_content_blocks([_DictOnlyToolCall()])
        block = blocks[0]
        assert isinstance(block, ToolCall)
        assert block.raw_arguments == '{"x": 1}'
        assert block.arguments == {"x": 1}

    async def test_dangling_tool_call_dropped_from_final_assistant_message(
        self,
    ) -> None:
        """A mid-tool-loop death leaves an unanswered final tool call.

        The appended assistant message comes from the LAST inference's
        output; a tool_call there is unanswered by construction — its
        result would live in the input of a next inference that never
        ran.  Providers reject an assistant tool_call followed directly
        by a user message, so the dangling call is dropped on replay and
        the model re-decides it.  A tool_call-only output therefore
        appends no assistant message at all (the thought-only precedent).
        """
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_text_msg("rank bond ETFs")], episode_id=ep
        )
        dangling = ContentBlockChatOutputToolCall(
            id="out-tc-4",
            raw_arguments='{"q": "bond"}',
            raw_name="search",
            arguments={"q": "bond"},
            name="search",
        )
        object.__setattr__(inf, "output", [dangling])
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assert len(history) == 1
        assert history[0]["role"] == "user"

    async def test_final_assistant_text_survives_the_dangling_drop(self) -> None:
        """Mixed final output: the text is kept, only the dangling call goes."""
        ep = str(uuid4())
        inf = _stored_chat(episode_id=ep)
        output_blocks = [
            ContentBlockChatOutputText(text="Let me check the market."),
            ContentBlockChatOutputToolCall(
                id="out-tc-5",
                raw_arguments='{"q": "bond"}',
                raw_name="search",
                arguments={"q": "bond"},
                name="search",
            ),
        ]
        object.__setattr__(inf, "output", output_blocks)
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assistant = history[-1]
        assert assistant["role"] == "assistant"
        content = assistant["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        block = content[0]
        assert isinstance(block, Text)
        assert block.text == "Let me check the market."

    async def test_poisoned_episode_replays_history_minus_the_dangling_call(
        self,
    ) -> None:
        """The 2026-08-17 incident shape end to end.

        An episode whose last inference output is a tool_call replays as
        the prior user/tool context with the dangling call dropped:
        answered input-side calls survive untouched (they deserialized
        fine throughout the incident), and nothing malformed is appended.
        """
        ep = str(uuid4())
        msgs = [
            _user_text_msg("rank bond ETFs"),
            StoredInputMessage(
                role="assistant",
                content=[
                    StoredInputMessageContentToolCall(
                        id="c1",
                        name="search_tickers",
                        arguments='{"q": "bond ETF"}',
                    )
                ],
            ),
            StoredInputMessage(
                role="user",
                content=[
                    StoredInputMessageContentToolResult(
                        id="c1",
                        name="search_tickers",
                        result="BND, IEF",
                    )
                ],
            ),
        ]
        inf = _stored_chat(user_messages=msgs, episode_id=ep)
        dangling = ContentBlockChatOutputToolCall(
            id="c2",
            raw_arguments='{"ticker": "BND"}',
            raw_name="get_bars",
            arguments={"ticker": "BND"},
            name="get_bars",
        )
        object.__setattr__(inf, "output", [dangling])
        t0 = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        mem = _make_memory(t0)
        history = await mem.load_history(ep)
        assert history is not None
        assert [m["role"] for m in history] == ["user", "assistant", "user"]
        answered = history[1]["content"]
        assert isinstance(answered, list)
        assert isinstance(answered[0], ToolCall)
        assert answered[0].raw_arguments == '{"q": "bond ETF"}'

    async def test_a_terminal_ended_episode_replays_into_a_working_turn(
        self,
    ) -> None:
        """A turn ended by ``terminal_tools`` stores the dangling shape.

        Its render calls never produce stored results — there is no next
        inference to carry them — so the episode looks exactly like the
        2026-08-17 poisoned shape and leans on the same drop.  What this
        pins is the composition: the reply survives, the calls go, and the
        following turn runs over the reconstructed history.
        """
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_text_msg("show me my portfolio")],
            episode_id=ep,
            output=[
                ContentBlockChatOutputText(text="Here is your portfolio."),
                ContentBlockChatOutputToolCall(
                    id="r1",
                    raw_arguments="{}",
                    raw_name="present_chart",
                    arguments={},
                    name="present_chart",
                ),
            ],
        )
        t0_memory = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        history = await _make_memory(t0_memory).load_history(ep)
        assert history is not None
        assert [m["role"] for m in history] == ["user", "assistant"]
        replayed_reply = history[-1]["content"]
        assert isinstance(replayed_reply, list)
        assert [type(b).__name__ for b in replayed_reply] == ["Text"]
        assert replayed_reply[0].text == "Here is your portfolio."

        t0_next = FakeT0([FakeChatResponse(content=[Text(text="Anything else?")])])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0_next),
            cast(Any, FakeMCPClient(tools=[])),
            "and my goals?",
            "agent",
            episode_id=ep,
            prior_messages=history,
        )
        assert result.response == "Anything else?"
        sent = t0_next.calls[0]["input"]["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]

    async def test_replay_thoughts_is_scoped_to_the_current_turn(self) -> None:
        """``replay_thoughts=True`` does not reach the cross-turn replay path.

        One request carries both halves, which is what makes this
        discriminating: the hydrated turn arrives stripped (the prompt
        cannot be allowed to grow without bound as an episode deepens)
        while this turn's own round keeps its reasoning. A flag wired
        into ``load_history`` would show a ``Thought`` in the first
        assistant message; a flag ignored by the loop would show none in
        the second.
        """
        ep = str(uuid4())
        inf = _stored_chat(
            user_messages=[_user_text_msg("what should I buy?")],
            episode_id=ep,
        )
        object.__setattr__(
            inf,
            "output",
            [
                _StoredOutputThought(text="reasoning from the earlier turn"),
                ContentBlockChatOutputText(text="Consider a broad index fund."),
            ],
        )
        t0_memory = FakeT0(
            [],
            list_inferences_responses=[FakeListInferencesResponse(inferences=[inf])],
        )
        history = await _make_memory(t0_memory).load_history(ep)
        assert history is not None

        lookup_result = {
            "lookup": CallToolResult(
                content=[], structured_content=None, meta=None, data="ok"
            )
        }
        this_turns_thought = Thought(text="reasoning inside this turn")
        call = ToolCall(
            id="c1",
            raw_arguments="{}",
            raw_name="lookup",
            arguments={},
            name="lookup",
            type="tool_call",
        )
        t0_next = FakeT0([
            FakeChatResponse(content=[this_turns_thought, call]),
            FakeChatResponse(content=[Text(text="Here you go.")]),
        ])
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0_next),
            cast(Any, FakeMCPClient(tools=[_lookup_tool()], call_results=lookup_result)),
            "and now?",
            "agent",
            episode_id=ep,
            prior_messages=history,
            replay_thoughts=True,
        )
        assert result.response == "Here you go."

        sent = t0_next.calls[1]["input"]["messages"]
        assistants = [m for m in sent if m["role"] == "assistant"]
        assert len(assistants) == 2

        hydrated, current = assistants
        assert [type(b).__name__ for b in hydrated["content"]] == ["Text"]
        assert hydrated["content"][0].text == "Consider a broad index fund."
        assert current["content"] == [this_turns_thought, call]

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


# ---------------------------------------------------------------------------
# list_conversations — dev-grade client-side group-by-episode
# ---------------------------------------------------------------------------


class TestListConversations:
    async def test_groups_by_episode_newest_first(self) -> None:
        chats = [
            _stored_chat(
                episode_id="ep_b",
                user_messages=[
                    _user_text_msg("B-first"),
                    _assistant_text_msg("a1"),
                    _user_text_msg("B-last"),
                ],
                timestamp="2026-05-02T10:00:00Z",
            ),
            _stored_chat(
                episode_id="ep_a",
                user_messages=[_user_text_msg("A-only")],
                timestamp="2026-05-01T10:00:00Z",
            ),
        ]
        t0 = FakeT0(
            [], list_inferences_responses=[FakeListInferencesResponse(inferences=chats)]
        )
        summaries = await _make_memory(t0, function_name="chat").list_conversations()

        assert [s.conversation_id for s in summaries] == ["ep_b", "ep_a"]
        newest, oldest = summaries
        # Cumulative input → title is the first user turn, preview the latest.
        assert newest.title == "B-first"
        assert newest.preview == "B-last"
        assert newest.turn_count == 2
        assert oldest.title == oldest.preview == "A-only"
        assert oldest.turn_count == 1
        assert isinstance(newest.updated_at, int)
        assert newest.updated_at > oldest.updated_at

    async def test_dedupes_multiple_inferences_of_one_episode(self) -> None:
        # A tool loop produces >1 inference per episode; the newest one (with the
        # most cumulative input) drives the single summary.
        chats = [
            _stored_chat(
                episode_id="ep",
                user_messages=[
                    _user_text_msg("q1"),
                    _assistant_text_msg("a1"),
                    _user_text_msg("q2"),
                ],
                timestamp="2026-05-01T10:05:00Z",
            ),
            _stored_chat(
                episode_id="ep",
                user_messages=[_user_text_msg("q1")],
                timestamp="2026-05-01T10:00:00Z",
            ),
        ]
        t0 = FakeT0(
            [], list_inferences_responses=[FakeListInferencesResponse(inferences=chats)]
        )
        summaries = await _make_memory(t0).list_conversations()

        assert len(summaries) == 1
        assert summaries[0].conversation_id == "ep"
        assert summaries[0].preview == "q2"
        assert summaries[0].turn_count == 2

    async def test_limit_caps_returned_conversations(self) -> None:
        chats = [
            _stored_chat(
                episode_id=f"ep{i}",
                user_messages=[_user_text_msg(f"q{i}")],
                timestamp=f"2026-05-0{i}T10:00:00Z",
            )
            for i in (1, 2, 3)
        ]
        t0 = FakeT0(
            [], list_inferences_responses=[FakeListInferencesResponse(inferences=chats)]
        )
        summaries = await _make_memory(t0).list_conversations(limit=2)

        assert [s.conversation_id for s in summaries] == ["ep3", "ep2"]

    async def test_tags_scope_the_request_filter_and_window(self) -> None:
        t0 = FakeT0(
            [], list_inferences_responses=[FakeListInferencesResponse(inferences=[])]
        )
        await _make_memory(t0, function_name="chat").list_conversations(
            tags={"customer_id": "c1"}, window=50
        )

        req = t0.list_inferences_calls[0]["request"]
        assert req.function_name == "chat"
        assert req.limit == 50
        # A single tag builds a bare InferenceFilterTag (no And wrapper).
        assert isinstance(req.filters, InferenceFilterTag)
        assert req.filters.key == "customer_id"
        assert req.filters.value == "c1"
