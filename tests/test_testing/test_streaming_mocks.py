"""Unit tests for the streaming test helpers."""

from __future__ import annotations

from typing import Any

import pytest
from tensorzero import Text, ToolCall

from t0_agents.core.types import AgentResult
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    TextDeltaEvent,
    ToolResultEvent,
)
from t0_agents.testing import MockStreamingGateway, fake_event_stream


async def _drain(iterator: Any) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for event in iterator:
        events.append(event)
    return events


class TestFakeEventStream:
    async def test_wraps_iterable_into_async_iterator(self) -> None:
        scripted: list[AgentEvent] = [
            TextDeltaEvent(id="t1", delta="hello"),
            TextDeltaEvent(id="t1", delta=" world"),
            DoneEvent(result=AgentResult(response="hello world", turns=1)),
        ]

        events = await _drain(fake_event_stream(scripted))

        assert len(events) == 3
        assert isinstance(events[-1], DoneEvent)
        assert events[-1].result.response == "hello world"

    async def test_empty_iterable_yields_nothing(self) -> None:
        events = await _drain(fake_event_stream([]))
        assert events == []

    async def test_mixed_event_types(self) -> None:
        scripted: list[AgentEvent] = [
            TextDeltaEvent(id="t1", delta="hi"),
            ToolResultEvent(id="c1", name="lookup", result="ok"),
        ]

        events = await _drain(fake_event_stream(scripted))

        assert len(events) == 2
        assert isinstance(events[0], TextDeltaEvent)
        assert isinstance(events[1], ToolResultEvent)


class TestMockStreamingGateway:
    async def test_stream_true_yields_chunks(self) -> None:
        gateway = MockStreamingGateway([[Text(text="hi")]])

        stream_iter = await gateway.inference(stream=True, function_name="f")
        chunks: list[Any] = []
        async for chunk in stream_iter:
            chunks.append(chunk)

        # Expect two chunks: content chunk + terminal usage chunk.
        assert len(chunks) == 2
        assert len(chunks[0].content) == 1
        assert chunks[1].usage is not None

    async def test_stream_false_raises(self) -> None:
        gateway = MockStreamingGateway([])

        with pytest.raises(ValueError, match="stream=False"):
            stream_iter = await gateway.inference(stream=False, function_name="f")
            # Drain to surface the error even if it was deferred.
            async for _ in stream_iter:
                pass

    async def test_records_calls(self) -> None:
        gateway = MockStreamingGateway([[Text(text="ok")]])

        stream_iter = await gateway.inference(
            stream=True, function_name="probe", episode_id=None
        )
        async for _ in stream_iter:
            pass

        assert len(gateway.calls) == 1
        assert gateway.calls[0]["function_name"] == "probe"

    async def test_tool_call_block_translates_to_chunk(self) -> None:
        from tensorzero.types import ToolCallChunk

        gateway = MockStreamingGateway(
            [[ToolCall(id="c1", raw_name="add", raw_arguments="{}", name="add", arguments={})]]
        )

        stream_iter = await gateway.inference(stream=True, function_name="f")
        chunks = [c async for c in stream_iter]

        # First chunk has the tool-call content-block chunk.
        assert len(chunks[0].content) == 1
        chunk = chunks[0].content[0]
        assert isinstance(chunk, ToolCallChunk)
        assert chunk.raw_name == "add"

    async def test_exhausted_script_yields_empty_text(self) -> None:
        """Parity with MockGateway.script: an empty ``Text`` chunk on exhaustion.

        Guarantees the streaming agent loop sees an assistant turn with one
        empty text block (terminating cleanly) rather than an empty content
        list that diverges from the non-streaming mock's documented contract.
        """
        from tensorzero.types import TextChunk

        gateway = MockStreamingGateway([])

        stream_iter = await gateway.inference(stream=True, function_name="f")
        chunks = [c async for c in stream_iter]

        assert len(chunks[0].content) == 1
        assert isinstance(chunks[0].content[0], TextChunk)
        assert chunks[0].content[0].text == ""
