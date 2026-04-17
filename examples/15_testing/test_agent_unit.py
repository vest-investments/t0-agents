"""Unit tests for the demo agent — deterministic, no Docker, no API keys.

Every turn is pre-scripted; every tool call returns a canned result.
The assertions inspect the mock's recorded state rather than the
model's free-form output, so the tests don't flake on model
non-determinism.  Runs in milliseconds.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from agent import build_agent
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
