"""Unit tests for :class:`StreamingAgentLoop`."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, Thought
from tensorzero.types import TensorZeroError, ThoughtSummaryBlock

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.stream.events import (
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from t0_agents.stream.runner import StreamingAgentLoop, _final_text
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import (
    FakeT0,
    chat_chunk,
    final_chunk,
    text_chunk,
    thought_chunk,
    tool_call_chunk,
)


def _tool_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        data=text,
        is_error=False,
        meta=None,
    )


async def test_streaming_loop_terminates_on_text_only() -> None:
    """One turn producing text → DoneEvent with the joined text."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("hello ", block_id="t1")]),
                chat_chunk([text_chunk("world", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = []
    async for event in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
    ):
        events.append(event)

    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.delta for d in text_deltas] == ["hello ", "world"]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "hello world"
    assert done[0].result.turns == 1
    assert done[0].result.response_source == "text"
    assert len(done[0].result.inference_ids) == 1

    assert events[-1] is done[0]


async def test_streaming_loop_dispatches_tool_calls() -> None:
    """A tool-call turn emits ToolCallEvent, dispatches, feeds result back."""
    fake_t0 = FakeT0(
        responses=[
            # Turn 1: one tool call
            [
                chat_chunk(
                    [
                        tool_call_chunk(
                            raw_name="add",
                            raw_arguments='{"a":1,"b":2}',
                            block_id="c1",
                        )
                    ]
                ),
                final_chunk(),
            ],
            # Turn 2: text response
            [
                chat_chunk([text_chunk("sum is 3", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    fake_client = FakeMCPClient(
        tools=[
            MCPTool(
                name="add",
                description="add two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            )
        ],
        call_results={"add": _tool_result("3")},
    )

    runner = StreamingAgentLoop()
    events = []
    async for event in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "What is 1+2?",
        "agent",
    ):
        events.append(event)

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"a": 1, "b": 2}

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].id == "c1"
    assert tool_results[0].name == "add"

    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert [t.turn for t in turn_ends] == [1, 2]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "sum is 3"
    assert done[0].result.turns == 2


async def test_streaming_loop_max_turns_raises() -> None:
    """Exhausting max_turns surfaces MaxTurnsExceededError."""
    # Each turn emits one tool call that the client can dispatch indefinitely.
    def _one_tool_turn() -> list:
        return [
            chat_chunk(
                [
                    tool_call_chunk(
                        raw_name="loop",
                        raw_arguments="{}",
                        block_id="loop",
                    )
                ]
            ),
            final_chunk(),
        ]

    fake_t0 = FakeT0(responses=[_one_tool_turn() for _ in range(5)])
    fake_client = FakeMCPClient(
        tools=[
            MCPTool(
                name="loop",
                description="loop forever",
                inputSchema={"type": "object"},
            )
        ],
        call_results={"loop": _tool_result("still looping")},
    )

    runner = StreamingAgentLoop()
    with pytest.raises(MaxTurnsExceededError):
        events: list = []
        async for event in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "loop forever",
            "agent",
            max_turns=3,
        ):
            events.append(event)


async def test_streaming_loop_forwards_config_knobs() -> None:
    """dryrun/tool_choice/tags/namespace must reach the inference call."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
        dryrun=True,
        tags={"scope": "unit-test"},
        namespace="test-ns",
    ):
        pass

    call = fake_t0.calls[0]
    assert call["stream"] is True
    assert call["dryrun"] is True
    assert call["tags"] == {"scope": "unit-test"}
    assert call["namespace"] == "test-ns"


async def test_streaming_loop_system_string_forwarded() -> None:
    """``system="..."`` reaches the streaming inference call as a raw string."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "hi",
        "agent",
        system="Reply only in haiku.",
    ):
        pass

    assert fake_t0.calls[0]["input"]["system"] == "Reply only in haiku."


async def test_streaming_loop_system_and_system_arguments_rejected() -> None:
    """Passing both kwargs raises ``ValueError`` before any chunk is yielded."""
    fake_t0 = FakeT0(responses=[])
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    with pytest.raises(ValueError, match="Pass either `system`"):
        async for _ in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "hi",
            "agent",
            system="raw",
            system_arguments={"foo": "bar"},
        ):
            pass
    assert fake_t0.calls == []


# ---------------------------------------------------------------------------
# `_final_text` — text/thought fallback (parity with mcp/loop.py)
# ---------------------------------------------------------------------------


def test_final_text_prefers_text_over_thought() -> None:
    """A ``Text`` block wins even when ``Thought`` content is present."""
    blocks = [
        Thought(text="hidden reasoning", summary=[ThoughtSummaryBlock(text="s")]),
        Text(text="the answer"),
    ]
    assert _final_text(blocks) == ("the answer", "text")


def test_final_text_mines_thought_summary() -> None:
    """With no ``Text``, summary-only ``Thought`` content is mined and the
    source is flagged ``"thought"`` (the new ``Thought.summary`` branch)."""
    blocks = [
        Thought(
            text=None,
            summary=[
                ThoughtSummaryBlock(text="reasoned "),
                ThoughtSummaryBlock(text="answer"),
            ],
        )
    ]
    assert _final_text(blocks) == ("reasoned \nanswer", "thought")


def test_final_text_mines_thought_text_and_summary_together() -> None:
    """Raw ``Thought.text`` and ``Thought.summary[*].text`` are both mined."""
    blocks = [
        Thought(
            text="raw thought",
            summary=[ThoughtSummaryBlock(text="summary tail")],
        )
    ]
    assert _final_text(blocks) == ("raw thought\nsummary tail", "thought")


def test_final_text_empty_blocks_returns_empty_text_source() -> None:
    """No content at all → empty string with the default ``"text"`` source."""
    assert _final_text([]) == ("", "text")


def test_final_text_skips_non_thought_blocks() -> None:
    """Non-``Thought`` blocks are skipped while mining the fallback — an empty
    ``Text`` block contributes nothing, the ``Thought`` supplies the answer."""
    blocks = [Text(text=""), Thought(text="answer")]
    assert _final_text(blocks) == ("answer", "thought")


# ---------------------------------------------------------------------------
# Thought-only final turn — framework boundary (assembler → runner)
# ---------------------------------------------------------------------------


async def test_streaming_loop_thought_only_summary_yields_response() -> None:
    """AC2 boundary: a turn whose ONLY content is a reasoning summary
    (gpt-oss harmony analysis channel via ``summary_text``) drives a
    non-empty thought-sourced response through the full assembler→runner
    boundary, so the projector is never handed silence."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk(
                    [thought_chunk(summary_text="the reasoned answer", block_id="r1")]
                ),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = [
        ev
        async for ev in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        )
    ]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "the reasoned answer"
    assert done[0].result.response_source == "thought"


async def test_streaming_loop_thought_only_raw_text_yields_response() -> None:
    """AC2 boundary: a thought-only turn carrying raw ``text`` (analysis
    channel, no ``final`` channel) still yields a non-empty thought-sourced
    response rather than an empty bubble."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk(
                    [thought_chunk(text="raw analysis answer", block_id="th1")]
                ),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = [
        ev
        async for ev in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        )
    ]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "raw analysis answer"
    assert done[0].result.response_source == "thought"


async def test_streaming_loop_still_emits_thought_events_for_devui() -> None:
    """DevUI no-regression: a thought+text turn still surfaces
    ThoughtDeltaEvent/ThoughtEvent so the cli/tui devui surface keeps
    rendering reasoning — the final response is the ``Text`` block and the
    ``_final_text`` change does not suppress thought events."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([thought_chunk(text="my reasoning", block_id="th1")]),
                chat_chunk([text_chunk("final answer", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = [
        ev
        async for ev in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        )
    ]

    thought_deltas = [e for e in events if isinstance(e, ThoughtDeltaEvent)]
    thought_ends = [e for e in events if isinstance(e, ThoughtEvent)]
    assert [d.delta for d in thought_deltas] == ["my reasoning"]
    assert len(thought_ends) == 1
    assert thought_ends[0].text == "my reasoning"

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "final answer"
    assert done[0].result.response_source == "text"


# ---------------------------------------------------------------------------
# Recoverable inference errors — parity with run_agent's corrective retry
# ---------------------------------------------------------------------------


async def test_streaming_loop_recovers_from_recoverable_error() -> None:
    """A recoverable ``TensorZeroError`` (bad tool name, status 400) is caught,
    a corrective turn is injected, and the loop recovers to a final answer —
    parity with the non-streaming ``run_agent`` recoverable contract."""
    fake_t0 = FakeT0(
        responses=[
            TensorZeroError(status_code=400, text="tool 'x' not in request.tools"),
            [
                chat_chunk([text_chunk("recovered answer", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    events = [
        ev
        async for ev in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        )
    ]

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert any("recoverable" in (e.message or "") for e in errors)
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "recovered answer"
    assert any(
        i.startswith("error-recoverable-turn-")
        for i in done[0].result.inference_ids
    )


async def test_streaming_loop_reraises_non_recoverable_error() -> None:
    """A non-recoverable status code (500) is re-raised, never retried."""
    fake_t0 = FakeT0(
        responses=[TensorZeroError(status_code=500, text="internal boom")],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    with pytest.raises(TensorZeroError):
        async for _ in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        ):
            pass


async def test_streaming_loop_reraises_recoverable_status_empty_text() -> None:
    """A recoverable status (400) with no error text is NOT retried — the
    pattern match needs text, so the error propagates."""
    fake_t0 = FakeT0(
        responses=[TensorZeroError(status_code=400, text="")],
    )
    fake_client = FakeMCPClient(tools=[])

    runner = StreamingAgentLoop()
    with pytest.raises(TensorZeroError):
        async for _ in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "Q?",
            "agent",
        ):
            pass


async def test_streaming_loop_forwards_model_routing() -> None:
    """model_name/params/cache_options reach the streamed inference call;
    the empty function_name default coalesces to None."""
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk([text_chunk("ok", block_id="t1")]),
                final_chunk(),
            ]
        ],
    )
    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, FakeMCPClient(tools=[])),
        "hi",
        model_name="groq::llama-3.3-70b-versatile",
        params={"chat_completion": {"temperature": 0.1}},
        cache_options={"enabled": "on"},
    ):
        pass

    call = fake_t0.calls[0]
    assert call["stream"] is True
    assert call["model_name"] == "groq::llama-3.3-70b-versatile"
    assert call["function_name"] is None
    assert call["params"] == {"chat_completion": {"temperature": 0.1}}
    assert call["cache_options"] == {"enabled": "on"}
