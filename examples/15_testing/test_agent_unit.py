"""Unit tests for the demo agent — deterministic, no Docker, no API keys.

Every turn is pre-scripted; every tool call returns a canned result.
The assertions inspect the mock's recorded state rather than the
model's free-form output, so the tests don't flake on model
non-determinism.  Runs in milliseconds.

Covers six test shapes:

* the happy path (one tool call, one text response),
* error recovery (handler raises, model answers anyway),
* negation (agent *doesn't* call a tool it wasn't asked to),
* the max-turns safety net,
* pass-through of a prebuilt ``fastmcp.CallToolResult(is_error=True)``,
* the ``MockGateway.recording()`` → ``.replay()`` workflow.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from agent import build_agent
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import ToolProvider
from t0_agents.testing import (
    MockGateway,
    MockToolProvider,
    assert_response_contains,
    assert_tool_args_equal,
    assert_tool_called,
    assert_tool_not_called,
    scenarios,
)


async def test_agent_uses_search_and_summarises() -> None:
    gateway = MockGateway.script(
        [
            [
                ToolCall(
                    id="c1",
                    raw_name="search_notes",
                    raw_arguments='{"query": "groceries"}',
                    name="search_notes",
                    arguments={"query": "groceries"},
                )
            ],
            [Text(text="Found 2 notes: milk, eggs.")],
        ]
    )
    tools = MockToolProvider(
        {"search_notes": lambda args: ["milk", "eggs"]},
    )

    agent = build_agent(
        cast(AsyncTensorZeroGateway, gateway),
        cast(ToolProvider, tools),
    )
    result = await agent.run("Find notes about groceries")

    assert_tool_called("search_notes", tool_provider=tools)
    assert_tool_args_equal(
        "search_notes", {"query": "groceries"}, tool_provider=tools
    )
    assert_response_contains(result, "milk")


async def test_agent_recovers_from_failing_tool() -> None:
    gateway = scenarios.tool_error_recovery(
        "save_note", final_text="I couldn't save it; please try again."
    )

    def broken(_args: dict[str, Any]) -> str:
        raise RuntimeError("disk is full")

    tools = MockToolProvider({"save_note": broken})

    agent = build_agent(
        cast(AsyncTensorZeroGateway, gateway),
        cast(ToolProvider, tools),
    )
    result = await agent.run("Save 'shopping' = 'milk eggs'")

    assert_tool_called("save_note", tool_provider=tools)
    assert_response_contains(result, "couldn't")


async def test_agent_never_calls_read_note_when_not_asked() -> None:
    gateway = MockGateway.script([[Text(text="No lookups needed.")]])
    tools = MockToolProvider(
        {
            "read_note": lambda args: "<not found>",
            "save_note": lambda args: "ok",
        }
    )

    agent = build_agent(
        cast(AsyncTensorZeroGateway, gateway),
        cast(ToolProvider, tools),
    )
    await agent.run("What's the weather?")

    assert_tool_not_called("read_note", tool_provider=tools)
    assert_tool_not_called("save_note", tool_provider=tools)


async def test_agent_aborts_when_max_turns_reached() -> None:
    gateway = scenarios.max_turns_exhausted("search_notes")
    tools = MockToolProvider({"search_notes": lambda args: []})

    agent = build_agent(
        cast(AsyncTensorZeroGateway, gateway),
        cast(ToolProvider, tools),
        max_turns=3,
    )

    with pytest.raises(MaxTurnsExceededError):
        await agent.run("Keep looking forever")


async def test_agent_surfaces_structured_tool_error() -> None:
    """Return a prebuilt ``CallToolResult(is_error=True)`` from the handler.

    ``MockToolProvider`` detects the prebuilt result and passes it through
    verbatim so the production serializer produces the exact
    ``"ERROR: <message>"`` string the real FastMCP client would hand the
    agent loop.  The serializer reads the error message from
    ``content[*].text`` (not ``data``) — so a realistic handler returns
    a :class:`mcp.types.TextContent` block with the failure detail.  The
    canned gateway branches on the ``"ERROR"`` marker to produce the
    final apology.
    """

    def save_note(_args: dict[str, Any]) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text="disk is full")],
            structured_content=None,
            meta=None,
            data=None,
            is_error=True,
        )

    tools = MockToolProvider({"save_note": save_note})

    def branching_model(**kwargs: Any) -> Any:
        # Inspect the last user/tool message to decide the next turn.
        rendered = str(kwargs.get("input", {}))
        if "ERROR" in rendered:
            return [Text(text="I couldn't save the note: disk is full.")]
        return [
            ToolCall(
                id="c1",
                raw_name="save_note",
                raw_arguments='{"title": "x", "body": "y"}',
                name="save_note",
                arguments={"title": "x", "body": "y"},
            )
        ]

    gateway = MockGateway.canned(branching_model)

    agent = build_agent(
        cast(AsyncTensorZeroGateway, gateway),
        cast(ToolProvider, tools),
    )
    result = await agent.run("Please save 'x' = 'y'")

    assert_tool_called("save_note", tool_provider=tools)
    assert_response_contains(result, "disk is full")

    # The second inference's input carries the serializer's "ERROR: ..." string —
    # pinning down the CallToolResult pass-through contract end-to-end.
    second_turn_input = repr(gateway.calls[1])
    assert "ERROR: disk is full" in second_turn_input


async def test_recording_then_replay_is_deterministic() -> None:
    """Demonstrate the ``recording()`` → ``.replay()`` workflow offline.

    In production the recorder would wrap an
    ``AsyncTensorZeroGateway.build_http(...)`` or
    ``.build_embedded(...)`` instance; here another ``MockGateway``
    stands in for that "real" gateway so the example stays dependency-free.
    The mechanics are identical: ``RecordingGateway`` forwards every
    call to the wrapped gateway, stores ``(kwargs, response)`` pairs,
    and ``.replay()`` returns a deterministic ``MockGateway`` you can
    reuse across tests without any network traffic.
    """
    stand_in = MockGateway.script(
        [
            [
                ToolCall(
                    id="c1",
                    raw_name="search_notes",
                    raw_arguments='{"query": "groceries"}',
                    name="search_notes",
                    arguments={"query": "groceries"},
                )
            ],
            [Text(text="Found: milk, eggs.")],
        ]
    )
    recorder = MockGateway.recording(stand_in)
    tools = MockToolProvider({"search_notes": lambda args: ["milk", "eggs"]})

    agent = build_agent(
        cast(AsyncTensorZeroGateway, recorder),
        cast(ToolProvider, tools),
    )
    original = await agent.run("Find notes about groceries")

    # Two turns captured — same as the script.
    assert len(recorder.captured) == 2

    # Snap a deterministic MockGateway from the capture and re-run.
    replay = recorder.replay()
    tools_replay = MockToolProvider({"search_notes": lambda args: ["milk", "eggs"]})

    agent_replay = build_agent(
        cast(AsyncTensorZeroGateway, replay),
        cast(ToolProvider, tools_replay),
    )
    replayed = await agent_replay.run("Find notes about groceries")

    # The replayed run produces the same observable response and tool usage
    # without touching the stand-in gateway again.
    assert original.response == replayed.response
    assert_tool_called("search_notes", tool_provider=tools_replay)
