"""Unit tests for :class:`TurnStreamAssembler`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tensorzero.types import ChatChunk

from t0_agents.stream.assembler import TurnStreamAssembler
from t0_agents.stream.events import (
    AgentEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
)
from tests._fake_t0 import (
    chat_chunk,
    final_chunk,
    text_chunk,
    thought_chunk,
    tool_call_chunk,
)


async def _drain(chunks: list[ChatChunk]) -> list[AgentEvent]:
    async def _gen() -> AsyncIterator[ChatChunk]:
        for chunk in chunks:
            yield chunk

    assembler = TurnStreamAssembler()
    return [event async for event in assembler.assemble(_gen())]


async def test_assembler_text_only() -> None:
    events = await _drain(
        [
            chat_chunk([text_chunk("Hel", block_id="t1")]),
            chat_chunk([text_chunk("lo", block_id="t1")]),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in deltas] == ["Hel", "lo"]
    assert all(d.id == "t1" for d in deltas)
    ends = [e for e in events if isinstance(e, TextEndEvent)]
    assert len(ends) == 1 and ends[0].id == "t1"


async def test_assembler_tool_call_streaming() -> None:
    events = await _drain(
        [
            chat_chunk(
                [tool_call_chunk(raw_name="get_", raw_arguments="", block_id="c1")]
            ),
            chat_chunk(
                [tool_call_chunk(raw_name="weather", raw_arguments='{"c', block_id="c1")]
            ),
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="",
                        raw_arguments='ity":"Tokyo"}',
                        block_id="c1",
                    )
                ]
            ),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, ToolCallDeltaEvent)]
    # Three chunks → three deltas, each carrying the correlation id.
    assert len(deltas) == 3
    assert all(d.id == "c1" for d in deltas)

    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    call = calls[0]
    assert call.id == "c1"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Tokyo"}


async def test_assembler_thought_chunks() -> None:
    events = await _drain(
        [
            chat_chunk([thought_chunk("Let me think", block_id="r1")]),
            chat_chunk([thought_chunk(" about this", block_id="r1")]),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, ThoughtDeltaEvent)]
    assert [d.delta for d in deltas] == ["Let me think", " about this"]
    thoughts = [e for e in events if isinstance(e, ThoughtEvent)]
    assert len(thoughts) == 1
    assert thoughts[0].text == "Let me think about this"
    assert thoughts[0].id == "r1"


async def test_assembler_mixed_blocks_in_stream() -> None:
    events = await _drain(
        [
            chat_chunk(
                [
                    thought_chunk("planning", block_id="r1"),
                ]
            ),
            chat_chunk(
                [
                    text_chunk("Here ", block_id="t1"),
                    tool_call_chunk(
                        raw_name="search",
                        raw_arguments='{"q":"x"}',
                        block_id="c1",
                    ),
                ]
            ),
            chat_chunk([text_chunk("you go", block_id="t1")]),
            final_chunk(),
        ]
    )
    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in text_deltas] == ["Here ", "you go"]

    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].arguments == {"q": "x"}

    thought_events = [e for e in events if isinstance(e, ThoughtEvent)]
    assert thought_events[0].text == "planning"


async def test_assembler_invalid_tool_args_json() -> None:
    events = await _drain(
        [
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="add",
                        raw_arguments="{not json",
                        block_id="c1",
                    )
                ]
            ),
            final_chunk(),
        ]
    )
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert errors and "invalid tool-call JSON" in errors[0].message
    # Empty-args ToolCallEvent is still emitted so consumers see the id.
    assert calls and calls[0].arguments == {} and calls[0].id == "c1"


async def test_assembler_tool_args_non_object() -> None:
    events = await _drain(
        [
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="echo",
                        raw_arguments='"hello"',  # valid JSON but not an object
                        block_id="c1",
                    )
                ]
            ),
            final_chunk(),
        ]
    )
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert errors and "must be an object" in errors[0].message
    assert calls and calls[0].arguments == {}


async def test_assembler_json_chunk_emits_error() -> None:
    """JsonChunk is out of scope for Agent.stream()."""
    from tensorzero.types import JsonChunk

    async def _gen() -> AsyncIterator[Any]:
        yield JsonChunk(
            inference_id=uuid_for_test(),
            episode_id=uuid_for_test(),
            variant_name="v",
            raw="{}",
        )

    assembler = TurnStreamAssembler()
    events = [event async for event in assembler.assemble(_gen())]
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors and "JsonChunk" in errors[0].message


async def test_assembler_unknown_chunk() -> None:
    """UnknownChunk → ErrorEvent, no crash."""
    from tensorzero.types import UnknownChunk

    chunk = chat_chunk([UnknownChunk(id="u1", data={"foo": "bar"}, type="unknown")])
    events = await _drain([chunk, final_chunk()])
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert errors and "unknown TensorZero chunk type" in errors[0].message


def uuid_for_test() -> Any:
    """Stable UUID factory kept out of the top-level so tests import cleanly."""
    from uuid import uuid4

    return uuid4()
