"""Golden gates freezing the turn behavior of both agent loops.

``GOLDEN_A`` freezes :func:`t0_agents.mcp.loop.run_agent`: the returned
``AgentResult`` plus the full request evolution (routing, episode
threading, advertised tool names, replay ``messages``) captured from
every ``t0.inference`` call.  ``GOLDEN_B`` freezes
:class:`t0_agents.stream.runner.StreamingAgentLoop`: the complete
:data:`AgentEvent` sequence via ``event_to_dict`` plus the same request
evolution.

Convention (per ``tests/test_wire/test_public_golden.py``): the
snapshots are **frozen** — when a refactor changes the output, fix the
code, never the snapshot.  A deliberate behavior change must update the
snapshot in its own commit with the WHY in the commit body.
"""

from __future__ import annotations

from typing import Any, cast

from t0_agents.mcp.loop import run_agent
from t0_agents.stream.codec import event_to_dict
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_t0 import FakeT0
from tests.test_parity._scenario import (
    FUNCTION_NAME,
    USER_MESSAGE,
    call_to_dict,
    make_mcp_client,
    make_nonstreaming_responses,
    make_streaming_responses,
    result_to_dict,
)

_USER_TURN = {"role": "user", "content": "Add 1 and 2, then greet x."}

_ASSISTANT_TOOL_TURN = {
    "role": "assistant",
    "content": [
        {"kind": "text", "text": "Working on it."},
        {
            "kind": "tool_call",
            "id": "tc-1",
            "name": "add",
            "raw_name": "add",
            "arguments": {"a": 1, "b": 2},
            "raw_arguments": '{"a": 1, "b": 2}',
        },
        {
            "kind": "tool_call",
            "id": "tc-2",
            "name": "greet",
            "raw_name": "greet",
            "arguments": {"name": "x"},
            "raw_arguments": '{"name": "x"}',
        },
    ],
}

_TOOL_RESULTS_TURN = {
    "role": "user",
    "content": [
        {"kind": "tool_result", "id": "tc-1", "name": "add", "result": "3"},
        {"kind": "tool_result", "id": "tc-2", "name": "greet", "result": "Hello, x!"},
    ],
}

_CORRECTIVE_TURN = {
    "role": "user",
    "content": (
        "The previous response contained an invalid tool call that was "
        "rejected: tool 'bogus' not in request.tools\n"
        "Available tools: ['add', 'greet']. Please retry with a valid "
        "tool name, stripping any special tokens."
    ),
}

_RESULT = {
    "value": "The sum is 3 and the greeting was sent.",
    "episode_id": "00000000-0000-4000-8000-0000000000e1",
    "inference_ids": [
        "00000000-0000-4000-8000-0000000000a1",
        "error-recoverable-turn-2",
        "00000000-0000-4000-8000-0000000000a3",
    ],
    "turns": 3,
    "response_source": "text",
}


def _calls(stream: bool | None) -> list[dict[str, Any]]:
    """The three-call request evolution — identical for both loops bar ``stream``."""
    return [
        {
            "function_name": "parity_agent",
            "model_name": None,
            "episode_id": None,
            "stream": stream,
            "tools": ["add", "greet"],
            "messages": [_USER_TURN],
        },
        {
            "function_name": "parity_agent",
            "model_name": None,
            "episode_id": "00000000-0000-4000-8000-0000000000e1",
            "stream": stream,
            "tools": ["add", "greet"],
            "messages": [_USER_TURN, _ASSISTANT_TOOL_TURN, _TOOL_RESULTS_TURN],
        },
        {
            "function_name": "parity_agent",
            "model_name": None,
            "episode_id": "00000000-0000-4000-8000-0000000000e1",
            "stream": stream,
            "tools": ["add", "greet"],
            "messages": [
                _USER_TURN,
                _ASSISTANT_TOOL_TURN,
                _TOOL_RESULTS_TURN,
                {"role": "assistant", "content": []},
                _CORRECTIVE_TURN,
            ],
        },
    ]


GOLDEN_A = {"result": _RESULT, "calls": _calls(stream=None)}

GOLDEN_B = {
    "events": [
        {"v": 1, "id": "th-1", "delta": "I should add first.", "type": "thought_delta"},
        {"v": 1, "id": "t-1", "delta": "Working on it.", "type": "text_delta"},
        {
            "v": 1,
            "id": "tc-1",
            "raw_name_delta": "add",
            "raw_arguments_delta": '{"a": 1, "b": 2}',
            "type": "tool_call_delta",
        },
        {
            "v": 1,
            "id": "tc-2",
            "raw_name_delta": "greet",
            "raw_arguments_delta": '{"name": "x"}',
            "type": "tool_call_delta",
        },
        {"v": 1, "id": "tc-1", "name": "add", "arguments": {"a": 1, "b": 2}, "type": "tool_call"},
        {"v": 1, "id": "tc-2", "name": "greet", "arguments": {"name": "x"}, "type": "tool_call"},
        {"v": 1, "id": "t-1", "type": "text_end"},
        {"v": 1, "id": "th-1", "text": "I should add first.", "type": "thought"},
        {"v": 1, "turn": 1, "type": "turn_end"},
        {
            "v": 1,
            "id": "tc-1",
            "name": "add",
            "result": "3",
            "is_error": False,
            "structured": None,
            "duration_ms": "<ms>",
            "server": None,
            "type": "tool_result",
        },
        {
            "v": 1,
            "id": "tc-2",
            "name": "greet",
            "result": "Hello, x!",
            "is_error": False,
            "structured": None,
            "duration_ms": "<ms>",
            "server": None,
            "type": "tool_result",
        },
        {
            "v": 1,
            "message": "recoverable TensorZeroError; retrying",
            "detail": {"text": "tool 'bogus' not in request.tools"},
            "type": "error",
        },
        {"v": 1, "turn": 2, "type": "turn_end"},
        {
            "v": 1,
            "id": "t-2",
            "delta": "The sum is 3 and the greeting was sent.",
            "type": "text_delta",
        },
        {"v": 1, "id": "t-2", "type": "text_end"},
        {"v": 1, "turn": 3, "type": "turn_end"},
        {
            "v": 1,
            "type": "done",
            "result": {
                **_RESULT,
                "guardrail_log": [],
                "diagnostics": None,
            },
        },
    ],
    "calls": _calls(stream=True),
}


async def test_golden_run_agent() -> None:
    """run_agent reproduces the frozen result + request evolution exactly."""
    t0 = FakeT0(responses=make_nonstreaming_responses())
    result = await run_agent(
        cast(Any, t0), cast(Any, make_mcp_client()), USER_MESSAGE, FUNCTION_NAME
    )
    actual = {
        "result": result_to_dict(result),
        "calls": [call_to_dict(c) for c in t0.calls],
    }
    assert actual == GOLDEN_A


def _freeze_durations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize the wall-clock ``duration_ms`` so the golden stays deterministic.

    A tool_result's ``duration_ms`` is a real ``call_tool`` timing — its
    presence and type are frozen, its value is runtime.  Each is asserted to
    be a non-negative float, then replaced with the ``"<ms>"`` sentinel the
    golden carries.
    """
    for event in events:
        if event.get("type") == "tool_result":
            duration = event["duration_ms"]
            assert isinstance(duration, float) and duration >= 0.0
            event["duration_ms"] = "<ms>"
    return events


async def test_golden_streaming_loop() -> None:
    """StreamingAgentLoop reproduces the frozen event sequence + request evolution."""
    t0 = FakeT0(responses=make_streaming_responses())
    runner = StreamingAgentLoop()
    events = [
        e
        async for e in runner.stream(
            cast(Any, t0), cast(Any, make_mcp_client()), USER_MESSAGE, FUNCTION_NAME
        )
    ]
    actual = {
        "events": _freeze_durations([event_to_dict(e) for e in events]),
        "calls": [call_to_dict(c) for c in t0.calls],
    }
    assert actual == GOLDEN_B
