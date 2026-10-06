"""Engine-level equivalence: both loops drive the same conversation identically.

Unlike the frozen goldens (which pin each loop's output byte-for-byte),
this test asserts the two loops against EACH OTHER on the shared
scenario: same final ``AgentResult``, same request evolution — message
stacks, episode threading, advertised tools — modulo the ``stream``
flag.  A future change that alters both loops in the same wrong way
slips past this test but trips the goldens; a change that alters only
one loop trips this test with a readable diff.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from t0_agents.mcp.loop import run_agent
from t0_agents.stream.events import DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_t0 import FakeT0
from tests.test_parity._scenario import (
    FUNCTION_NAME,
    GEMINI_SIGNATURES,
    TERMINAL_TOOLS,
    USER_MESSAGE,
    call_to_dict,
    make_gemini_nonstreaming_responses,
    make_gemini_streaming_responses,
    make_mcp_client,
    make_nonstreaming_responses,
    make_streaming_responses,
    make_terminal_nonstreaming_responses,
    make_terminal_streaming_responses,
    result_to_dict,
)


async def test_loop_and_runner_produce_equivalent_runs() -> None:
    t0_loop = FakeT0(responses=make_nonstreaming_responses())
    loop_result = await run_agent(
        cast(Any, t0_loop), cast(Any, make_mcp_client()), USER_MESSAGE, FUNCTION_NAME
    )

    t0_stream = FakeT0(responses=make_streaming_responses())
    events = [
        e
        async for e in StreamingAgentLoop().stream(
            cast(Any, t0_stream),
            cast(Any, make_mcp_client()),
            USER_MESSAGE,
            FUNCTION_NAME,
        )
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]

    assert result_to_dict(loop_result) == result_to_dict(done.result)

    loop_calls = [call_to_dict(c) for c in t0_loop.calls]
    stream_calls = [call_to_dict(c) for c in t0_stream.calls]
    assert all(c.pop("stream") is None for c in loop_calls)
    assert all(c.pop("stream") is True for c in stream_calls)
    assert loop_calls == stream_calls


async def test_loop_and_runner_replay_reasoning_equivalently() -> None:
    """Replaying reasoning must look the same from both loops.

    The scenario's turn 1 already carries a ``Thought``, so the flag
    changes what turns 2 and 3 are sent without changing the
    conversation.  The second half is the control: the same scenario
    with the flag off must produce a *different* request stack, or this
    test would pass on a knob that does nothing.
    """
    t0_loop = FakeT0(responses=make_nonstreaming_responses())
    loop_result = await run_agent(
        cast(Any, t0_loop),
        cast(Any, make_mcp_client()),
        USER_MESSAGE,
        FUNCTION_NAME,
        replay_thoughts=True,
    )

    t0_stream = FakeT0(responses=make_streaming_responses())
    events = [
        e
        async for e in StreamingAgentLoop().stream(
            cast(Any, t0_stream),
            cast(Any, make_mcp_client()),
            USER_MESSAGE,
            FUNCTION_NAME,
            replay_thoughts=True,
        )
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]

    assert result_to_dict(loop_result) == result_to_dict(done.result)

    replayed = [call_to_dict(c) for c in t0_loop.calls]
    stream_calls = [call_to_dict(c) for c in t0_stream.calls]
    assert all(c.pop("stream") is None for c in replayed)
    assert all(c.pop("stream") is True for c in stream_calls)
    assert replayed == stream_calls

    t0_stripped = FakeT0(responses=make_nonstreaming_responses())
    await run_agent(
        cast(Any, t0_stripped),
        cast(Any, make_mcp_client()),
        USER_MESSAGE,
        FUNCTION_NAME,
    )
    stripped = [call_to_dict(c) for c in t0_stripped.calls]
    for c in stripped:
        c.pop("stream")

    assert replayed != stripped
    rounds_replaying_turn_one = 2
    assert (
        _thought_kinds(replayed)
        == ["I should add first."] * rounds_replaying_turn_one
    )
    assert _thought_kinds(stripped) == []


def _thought_kinds(calls: list[dict[str, Any]]) -> list[str]:
    """Every replayed assistant thought across a run's requests, in order."""
    return [
        block["text"]
        for call in calls
        for message in call["messages"]
        if message["role"] == "assistant"
        for block in message["content"]
        if block["kind"] == "thought"
    ]


async def test_loop_and_runner_end_a_terminal_round_equivalently() -> None:
    """A round that ends on its tools must look the same from both loops."""
    t0_loop = FakeT0(responses=make_terminal_nonstreaming_responses())
    loop_result = await run_agent(
        cast(Any, t0_loop),
        cast(Any, make_mcp_client()),
        USER_MESSAGE,
        FUNCTION_NAME,
        terminal_tools=TERMINAL_TOOLS,
    )

    t0_stream = FakeT0(responses=make_terminal_streaming_responses())
    events = [
        e
        async for e in StreamingAgentLoop().stream(
            cast(Any, t0_stream),
            cast(Any, make_mcp_client()),
            USER_MESSAGE,
            FUNCTION_NAME,
            terminal_tools=TERMINAL_TOOLS,
        )
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]

    assert result_to_dict(loop_result) == result_to_dict(done.result)
    assert loop_result.turns == 1

    loop_calls = [call_to_dict(c) for c in t0_loop.calls]
    stream_calls = [call_to_dict(c) for c in t0_stream.calls]
    assert len(loop_calls) == len(stream_calls) == 1
    assert all(c.pop("stream") is None for c in loop_calls)
    assert all(c.pop("stream") is True for c in stream_calls)
    assert loop_calls == stream_calls


GEMINI_SIGNED_CALLS = [
    [],
    [(GEMINI_SIGNATURES[0], "tc-1")],
    [(GEMINI_SIGNATURES[0], "tc-1"), (GEMINI_SIGNATURES[1], "tc-2")],
]


def _signatures_ahead_of_calls(calls: list[dict[str, Any]]) -> list[list[tuple[str, Any]]]:
    """Per request, each replayed signature-only thought and the id of the block after it.

    The gateway folds such a thought into its successor, so the successor
    is the part the signature lands on.
    """
    return [
        [
            (block["signature"], successor.get("id"))
            for message in call["messages"]
            if message["role"] == "assistant"
            for block, successor in zip(message["content"], message["content"][1:])
            if block["kind"] == "thought" and block["text"] is None
        ]
        for call in calls
    ]


async def test_loop_replays_each_gemini_signature_ahead_of_its_call() -> None:
    t0 = FakeT0(responses=make_gemini_nonstreaming_responses(preamble=True))
    await run_agent(
        cast(Any, t0),
        cast(Any, make_mcp_client()),
        USER_MESSAGE,
        FUNCTION_NAME,
        replay_thoughts=True,
    )

    calls = [call_to_dict(c) for c in t0.calls]
    assert _signatures_ahead_of_calls(calls) == GEMINI_SIGNED_CALLS


@pytest.mark.parametrize("preamble", [False, True])
async def test_loop_and_runner_replay_gemini_signatures_equivalently(
    preamble: bool,
) -> None:
    """With text ahead of the call, the signature must still precede the call."""
    t0_loop = FakeT0(responses=make_gemini_nonstreaming_responses(preamble=preamble))
    loop_result = await run_agent(
        cast(Any, t0_loop),
        cast(Any, make_mcp_client()),
        USER_MESSAGE,
        FUNCTION_NAME,
        replay_thoughts=True,
    )

    t0_stream = FakeT0(responses=make_gemini_streaming_responses(preamble=preamble))
    events = [
        e
        async for e in StreamingAgentLoop().stream(
            cast(Any, t0_stream),
            cast(Any, make_mcp_client()),
            USER_MESSAGE,
            FUNCTION_NAME,
            replay_thoughts=True,
        )
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]

    assert result_to_dict(loop_result) == result_to_dict(done.result)

    loop_calls = [call_to_dict(c) for c in t0_loop.calls]
    stream_calls = [call_to_dict(c) for c in t0_stream.calls]
    assert all(c.pop("stream") is None for c in loop_calls)
    assert all(c.pop("stream") is True for c in stream_calls)
    assert loop_calls == stream_calls
    assert _signatures_ahead_of_calls(stream_calls) == GEMINI_SIGNED_CALLS
