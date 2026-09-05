"""Unit tests for :class:`StreamingAgentLoop`."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway, Text, Thought, ToolCall
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


def _tool_result(
    text: str, *, structured_content: dict[str, Any] | None = None
) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=structured_content,
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
    assert tool_results[0].structured is None  # tool returned no structured payload

    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert [t.turn for t in turn_ends] == [1, 2]

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "sum is 3"
    assert done[0].result.turns == 2


async def test_streaming_loop_surfaces_observability_on_tool_result() -> None:
    """The streaming ToolResultEvent carries the executor's observability
    fields — typed is_error, structured payload, duration, and the attributed
    server — without altering the model-visible result string."""

    class _AttributingMCP(FakeMCPClient):
        def server_for(self, name: str) -> str | None:
            return "math" if name == "add" else None

    fake_t0 = FakeT0(
        responses=[
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
            [
                chat_chunk([text_chunk("sum is 3", block_id="t1")]),
                final_chunk(),
            ],
        ],
    )
    fake_client = _AttributingMCP(
        tools=[
            MCPTool(
                name="add",
                description="add two numbers",
                inputSchema={"type": "object", "properties": {}},
            )
        ],
        call_results={"add": _tool_result("3", structured_content={"sum": 3})},
    )

    runner = StreamingAgentLoop()
    tool_results = [
        event
        async for event in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "What is 1+2?",
            "agent",
        )
        if isinstance(event, ToolResultEvent)
    ]

    assert len(tool_results) == 1
    assert tool_results[0].result == "3"  # model-visible string unchanged
    assert tool_results[0].structured == {"sum": 3}
    assert tool_results[0].is_error is False
    assert tool_results[0].duration_ms is not None
    assert tool_results[0].server == "math"


async def test_streaming_loop_forwards_none_tool_timeout() -> None:
    """``tool_timeout=None`` reaches ``call_tool(timeout=None)`` verbatim.

    Regression gate for the M1 timeout-truth fix: the streaming loop must
    never coerce ``None`` back to a numeric default (the old ``Agent``
    terminals coerced it to 30 s, silently killing long elicitation waits).
    """
    fake_t0 = FakeT0(
        responses=[
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
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "What is 1+2?",
        "agent",
        tool_timeout=None,
    ):
        pass

    assert fake_client.call_tool_invocations[0][2] is None


async def test_streaming_replay_strips_thought_keeps_tool_call() -> None:
    """The assistant turn replayed as turn-2 input drops its ``Thought`` and
    keeps its ``ToolCall``.

    Reasoning models (e.g. gpt_oss_120 via groq) emit a Thought beside the
    ToolCall on turn 1; TensorZero rejects input thought blocks for
    thought-less providers, so the streaming loop must not replay them. The
    ToolCall is preserved so the tool-result turn stays coherent, and the
    returned answer is unchanged.
    """
    fake_t0 = FakeT0(
        responses=[
            # Turn 1: a reasoning Thought beside the tool call.
            [
                chat_chunk(
                    [
                        thought_chunk(text="let me add these", block_id="th1"),
                        tool_call_chunk(
                            raw_name="add",
                            raw_arguments='{"a":1,"b":2}',
                            block_id="c1",
                        ),
                    ]
                ),
                final_chunk(),
            ],
            # Turn 2: text response → done.
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

    # calls[1] is turn 2's inference; its input replays the turn-1 assistant
    # message — the Thought must be stripped and the ToolCall kept.
    turn2_messages = fake_t0.calls[1]["input"]["messages"]
    assistant = next(m for m in turn2_messages if m["role"] == "assistant")
    blocks = assistant["content"]
    assert not any(isinstance(b, Thought) for b in blocks)
    tool_calls = [b for b in blocks if isinstance(b, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "add"

    # The reasoning-model answer and turn count are unaffected by the strip.
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done) == 1
    assert done[0].result.response == "sum is 3"


async def test_streaming_replay_keeps_the_thought_when_replay_thoughts_is_on() -> None:
    """With ``replay_thoughts=True`` the streamed turn replays its reasoning.

    Same contract as the non-streaming loop, over blocks the runner
    assembled from chunks rather than received whole. The streamed
    ``ThoughtEvent`` s are unaffected either way — the flag governs the
    next request, not the consumer's view.
    """
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk(
                    [
                        thought_chunk(text="let me add these", block_id="th1"),
                        tool_call_chunk(
                            raw_name="add",
                            raw_arguments='{"a":1,"b":2}',
                            block_id="c1",
                        ),
                    ]
                ),
                final_chunk(),
            ],
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
        replay_thoughts=True,
    ):
        events.append(event)

    turn2_messages = fake_t0.calls[1]["input"]["messages"]
    assistant = next(m for m in turn2_messages if m["role"] == "assistant")
    blocks = assistant["content"]
    thoughts = [b for b in blocks if isinstance(b, Thought)]
    assert len(thoughts) == 1
    assert thoughts[0].text == "let me add these"
    tool_calls = [b for b in blocks if isinstance(b, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "add"

    done = [e for e in events if isinstance(e, DoneEvent)]
    assert done[0].result.response == "sum is 3"
    assert done[0].result.turns == 2


async def test_streaming_rebuild_carries_the_reasoning_signature() -> None:
    """A replayed thought keeps the signature and provider the chunks carried.

    A provider that verifies its own reasoning rejects a replayed block
    whose signature is missing or altered, so the rebuild has to carry
    what arrived rather than reconstruct text alone. The metadata can
    land on any chunk of the block — here, the last.
    """
    fake_t0 = FakeT0(
        responses=[
            [
                chat_chunk(
                    [
                        thought_chunk(text="weighing ", block_id="th1"),
                        thought_chunk(
                            text="the options",
                            block_id="th1",
                            signature="sig-abc",
                            provider_type="acme",
                        ),
                        tool_call_chunk(
                            raw_name="add", raw_arguments="{}", block_id="c1"
                        ),
                    ]
                ),
                final_chunk(),
            ],
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
                inputSchema={"type": "object", "properties": {}},
            )
        ],
        call_results={"add": _tool_result("3")},
    )

    runner = StreamingAgentLoop()
    async for _ in runner.stream(
        cast(AsyncTensorZeroGateway, fake_t0),
        cast(Any, fake_client),
        "What is 1+2?",
        "agent",
        replay_thoughts=True,
    ):
        pass

    assistant = next(
        m for m in fake_t0.calls[1]["input"]["messages"] if m["role"] == "assistant"
    )
    (thought,) = [b for b in assistant["content"] if isinstance(b, Thought)]
    assert thought.text == "weighing the options"
    assert thought.signature == "sig-abc"
    assert thought.provider_type == "acme"


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


async def test_streaming_loop_system_arguments_with_model_name_rejected() -> None:
    """stream() rejects system_arguments on the functionless path before
    opening the stream — mirrors run_agent's eager guard."""
    fake_t0 = FakeT0(responses=[])
    runner = StreamingAgentLoop()
    with pytest.raises(ValueError, match="system_arguments"):
        async for _ in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, FakeMCPClient(tools=[])),
            "hi",
            model_name="groq::llama-3.3-70b-versatile",
            system_arguments={"k": "v"},
        ):
            pass
    assert fake_t0.calls == []


# --- terminal tools ---------------------------------------------------------


_PRESENT = ("present_*",)


def _render_tool(name: str = "present_chart") -> MCPTool:
    return MCPTool(
        name=name,
        description="render something for the user",
        inputSchema={"type": "object", "properties": {}},
    )


def _render_turn(text: str | None = "Here is your portfolio.") -> list[Any]:
    blocks: list[Any] = []
    if text is not None:
        blocks.append(text_chunk(text, block_id="t1"))
    blocks.append(
        tool_call_chunk(raw_name="present_chart", raw_arguments="{}", block_id="c1")
    )
    return [chat_chunk(blocks), final_chunk()]


async def _drive(
    fake_t0: FakeT0, fake_client: FakeMCPClient, **kwargs: Any
) -> list[Any]:
    return [
        event
        async for event in StreamingAgentLoop().stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            cast(Any, fake_client),
            "show me my portfolio",
            "agent",
            **kwargs,
        )
    ]


async def test_terminal_round_ends_the_stream_after_one_inference() -> None:
    """The render is the reply: results, then Done, with no second inference."""
    fake_t0 = FakeT0(responses=[_render_turn()])
    fake_client = FakeMCPClient(
        tools=[_render_tool()],
        call_results={"present_chart": _tool_result("rendered")},
    )

    events = await _drive(fake_t0, fake_client, terminal_tools=_PRESENT)

    assert [type(e).__name__ for e in events] == [
        "TextDeltaEvent",
        "ToolCallDeltaEvent",
        "ToolCallEvent",
        "TextEndEvent",
        "TurnEndEvent",
        "ToolResultEvent",
        "DoneEvent",
    ]
    assert len(fake_t0.calls) == 1
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.response == "Here is your portfolio."
    assert done.result.response_source == "text"
    assert done.result.turns == 1


async def test_terminal_round_without_text_runs_another_inference() -> None:
    fake_t0 = FakeT0(
        responses=[
            _render_turn(text=None),
            [chat_chunk([text_chunk("there it is", block_id="t2")]), final_chunk()],
        ]
    )
    fake_client = FakeMCPClient(
        tools=[_render_tool()],
        call_results={"present_chart": _tool_result("rendered")},
    )

    events = await _drive(fake_t0, fake_client, terminal_tools=_PRESENT)

    assert len(fake_t0.calls) == 2
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.response == "there it is"
    assert done.result.turns == 2


async def test_errored_terminal_result_runs_another_inference() -> None:
    fake_t0 = FakeT0(
        responses=[
            _render_turn(),
            [
                chat_chunk([text_chunk("let me try that again", block_id="t2")]),
                final_chunk(),
            ],
        ]
    )
    fake_client = FakeMCPClient(
        tools=[_render_tool()],
        call_results={
            "present_chart": CallToolResult(
                content=[TextContent(type="text", text="bad spec")],
                structured_content=None,
                data="bad spec",
                is_error=True,
                meta=None,
            )
        },
    )

    events = await _drive(fake_t0, fake_client, terminal_tools=_PRESENT)

    assert len(fake_t0.calls) == 2
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.response == "let me try that again"


async def test_default_leaves_the_render_round_running() -> None:
    """Without terminal_tools the same stream behaves exactly as before."""
    fake_t0 = FakeT0(
        responses=[
            _render_turn(),
            [chat_chunk([text_chunk("anything else?", block_id="t2")]), final_chunk()],
        ]
    )
    fake_client = FakeMCPClient(
        tools=[_render_tool()],
        call_results={"present_chart": _tool_result("rendered")},
    )

    events = await _drive(fake_t0, fake_client)

    assert len(fake_t0.calls) == 2
    assert [t.turn for t in events if isinstance(t, TurnEndEvent)] == [1, 2]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.response == "anything else?"
    assert done.result.turns == 2
