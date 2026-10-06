"""Unit tests for :class:`TurnStreamAssembler`."""

from __future__ import annotations

import json
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
    _assembler, events = await _drain_with_assembler(chunks)
    return events


async def _drain_with_assembler(
    chunks: list[ChatChunk],
) -> tuple[TurnStreamAssembler, list[AgentEvent]]:
    async def _gen() -> AsyncIterator[ChatChunk]:
        for chunk in chunks:
            yield chunk

    assembler = TurnStreamAssembler()
    return assembler, [event async for event in assembler.assemble(_gen())]


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


async def test_assembler_thought_summary_text_chunks() -> None:
    """Reasoning emitted via ``summary_text`` flows as ``ThoughtDeltaEvent``.

    Anchors the gpt-oss / o-series case where the provider populates
    ``summary_text`` instead of ``text``; before the fix, every chunk was
    silently dropped and the consumer saw an empty stream.
    """
    events = await _drain(
        [
            chat_chunk([thought_chunk(summary_text="reasoning ", block_id="r1")]),
            chat_chunk([thought_chunk(summary_text="summary", block_id="r1")]),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, ThoughtDeltaEvent)]
    assert [d.delta for d in deltas] == ["reasoning ", "summary"]
    thoughts = [e for e in events if isinstance(e, ThoughtEvent)]
    assert len(thoughts) == 1
    assert thoughts[0].text == "reasoning summary"
    assert thoughts[0].id == "r1"


async def test_assembler_thought_text_and_summary_folded() -> None:
    """When a single chunk carries both ``text`` and ``summary_text``,
    the assembler folds them into one delta with a newline separator.

    The newline mirrors ``mcp/loop.py``'s ``"\\n".join(...)`` over
    ``Thought.text`` + ``Thought.summary[*].text`` so a consumer that
    concatenates streamed deltas reconstructs the same string the
    non-streaming path produces in ``AgentResult.response``.
    """
    events = await _drain(
        [
            chat_chunk(
                [
                    thought_chunk(
                        text="raw",
                        summary_text="summary",
                        block_id="r1",
                    )
                ]
            ),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, ThoughtDeltaEvent)]
    assert len(deltas) == 1
    assert deltas[0].delta == "raw\nsummary"
    thoughts = [e for e in events if isinstance(e, ThoughtEvent)]
    assert thoughts[0].text == "raw\nsummary"


async def test_assembler_thought_empty_text_and_summary_dropped() -> None:
    """Chunks with neither ``text`` nor ``summary_text`` content are skipped."""
    events = await _drain(
        [
            chat_chunk([thought_chunk(text="", summary_text="", block_id="r1")]),
            chat_chunk([thought_chunk(summary_text="real", block_id="r1")]),
            final_chunk(),
        ]
    )
    deltas = [e for e in events if isinstance(e, ThoughtDeltaEvent)]
    # Only the non-empty chunk produces a delta.
    assert [d.delta for d in deltas] == ["real"]
    thoughts = [e for e in events if isinstance(e, ThoughtEvent)]
    assert thoughts[0].text == "real"


async def test_assembler_signature_only_thought_keeps_its_signature_silently() -> None:
    """A signature with no text is recorded for the rebuild but never streamed."""
    assembler, events = await _drain_with_assembler(
        [
            chat_chunk([thought_chunk(block_id="1", signature="sig-only")]),
            final_chunk(),
        ]
    )
    assert not [e for e in events if isinstance(e, ThoughtDeltaEvent | ThoughtEvent)]
    assert assembler.thought_blocks == {}
    assert assembler.thought_metadata == {
        "1": {"signature": "sig-only", "provider_type": None}
    }


async def test_assembler_block_order_follows_first_appearance_across_kinds() -> None:
    assembler, _ = await _drain_with_assembler(
        [
            chat_chunk([text_chunk("Adding ", block_id="0")]),
            chat_chunk([text_chunk("first.", block_id="0")]),
            chat_chunk([thought_chunk(block_id="1", signature="sig-only")]),
            chat_chunk([tool_call_chunk("add", "{}", block_id="0")]),
            final_chunk(),
        ]
    )
    assert assembler.block_order == [
        ("text", "0"),
        ("thought", "1"),
        ("tool_call", "0"),
    ]


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


def _large_spec() -> dict[str, Any]:
    return {
        "type": "line",
        "x": [f"2026-08-{1 + i % 28:02d}" for i in range(40)],
        "series": [
            {"name": name, "y": [round(100 + i * k * 0.37, 2) for i in range(40)]}
            for k, name in enumerate(("portfolio", "benchmark", "cash", "bonds"), start=1)
        ],
    }


async def test_assembler_salvages_object_followed_by_extra_brace() -> None:
    """A complete object plus one trailing ``}`` executes with its FULL arguments.

    No ``ErrorEvent``, and the id is not recorded as invalid — so the
    runner dispatches the recovered dict instead of the no-dispatch error.
    """
    spec = _large_spec()
    raw = json.dumps(spec) + "}"
    assert len(raw) > 1800
    assembler, events = await _drain_with_assembler(
        [
            chat_chunk([tool_call_chunk(raw_name="present_chart", block_id="c1")]),
            chat_chunk([tool_call_chunk(raw_arguments=raw[:900], block_id="c1")]),
            chat_chunk([tool_call_chunk(raw_arguments=raw[900:], block_id="c1")]),
            final_chunk(),
        ]
    )
    assert [e for e in events if isinstance(e, ErrorEvent)] == []
    (call,) = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.id == "c1" and call.name == "present_chart"
    assert call.arguments == spec
    assert assembler.invalid_tool_args == {}


async def test_assembler_salvages_closing_noise_arriving_in_its_own_chunk() -> None:
    events = await _drain(
        [
            chat_chunk(
                [tool_call_chunk(raw_name="add", raw_arguments='{"a": 2, "b": 3}', block_id="c1")]
            ),
            chat_chunk([tool_call_chunk(raw_arguments="\n]}", block_id="c1")]),
            final_chunk(),
        ]
    )
    assert [e for e in events if isinstance(e, ErrorEvent)] == []
    (call,) = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.arguments == {"a": 2, "b": 3}


async def test_assembler_missing_comma_keeps_the_error_path() -> None:
    """Interior corruption is never salvaged — today's error path, byte for byte."""
    raw = '{"a": 1, "b": 2 "c": 3}}'
    assembler, events = await _drain_with_assembler(
        [
            chat_chunk([tool_call_chunk(raw_name="add", raw_arguments=raw, block_id="c1")]),
            final_chunk(),
        ]
    )
    (error,) = [e for e in events if isinstance(e, ErrorEvent)]
    assert error.message.startswith("invalid tool-call JSON arguments: Expecting ',' delimiter")
    assert error.detail == {"tool_call_id": "c1", "name": "add"}
    (call,) = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.arguments == {}
    assert assembler.invalid_tool_args == {"c1": raw}


async def test_assembler_second_object_is_not_salvaged() -> None:
    """Two objects are ambiguous about which arguments the model meant."""
    raw = '{"a": 1}{"b": 2}'
    assembler, events = await _drain_with_assembler(
        [
            chat_chunk([tool_call_chunk(raw_name="add", raw_arguments=raw, block_id="c1")]),
            final_chunk(),
        ]
    )
    (error,) = [e for e in events if isinstance(e, ErrorEvent)]
    assert error.message.startswith("invalid tool-call JSON arguments: Extra data")
    (call,) = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.arguments == {}
    assert assembler.invalid_tool_args == {"c1": raw}


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
    """JsonChunk is out of scope for ``Run.events()``."""
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
