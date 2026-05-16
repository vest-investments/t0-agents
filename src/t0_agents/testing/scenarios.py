"""Canned :class:`MockGateway` factories for common agent-loop shapes.

Each factory returns a fresh :class:`MockGateway` wired to a specific
branching behaviour so tests can wire up a realistic scenario in one
line instead of hand-rolling the content blocks.  Pair each scenario
with a matching :class:`MockToolProvider` (or leave it empty when the
scenario covers a text-only exchange).

Example::

    from t0_agents.testing import scenarios, MockToolProvider

    gateway = scenarios.single_tool_happy("lookup", final_text="done")
    tools = MockToolProvider({"lookup": lambda args: "42"})
    agent = Agent(..., t0=gateway, tool_provider=tools)
    result = await agent.invoke("What's the answer?")
"""

from __future__ import annotations

from typing import Any

from tensorzero import Text, ToolCall

from t0_agents.testing.mocks import MockGateway


def single_tool_happy(
    tool_name: str = "do_thing",
    *,
    final_text: str = "done",
) -> MockGateway:
    """Two-turn happy path: model calls *tool_name* once, then answers.

    Turn 1: ``[ToolCall(tool_name, arguments={}, ...)]``.
    Turn 2: ``[Text(text=final_text)]``.
    """
    return MockGateway.script(
        [
            [
                ToolCall(
                    id="mock-call-1",
                    raw_name=tool_name,
                    raw_arguments="{}",
                    name=tool_name,
                    arguments={},
                )
            ],
            [Text(text=final_text)],
        ]
    )


def tool_error_recovery(
    tool_name: str = "flaky",
    *,
    final_text: str = "I couldn't run the tool; answering from memory.",
) -> MockGateway:
    """Two-turn recovery path: tool is called, the model answers anyway.

    Pair with a :class:`MockToolProvider` whose handler for *tool_name*
    raises — the agent loop stringifies the exception into an error
    tool result, and this scripted second turn lets the model produce a
    text response instead of retrying.
    """
    return MockGateway.script(
        [
            [
                ToolCall(
                    id="mock-call-1",
                    raw_name=tool_name,
                    raw_arguments="{}",
                    name=tool_name,
                    arguments={},
                )
            ],
            [Text(text=final_text)],
        ]
    )


def max_turns_exhausted(tool_name: str = "loop") -> MockGateway:
    """Loop-forever path: every turn emits the same tool call.

    Used to test :class:`~t0_agents.core.exceptions.MaxTurnsExceededError`.
    Pair with a :class:`MockToolProvider` whose handler returns something
    innocuous; the loop will dispatch the tool every turn and never see
    a text-only response, so the agent raises after ``max_turns``.
    """

    def source(**_: Any) -> list[Any]:
        return [
            ToolCall(
                id="mock-call-loop",
                raw_name=tool_name,
                raw_arguments="{}",
                name=tool_name,
                arguments={},
            )
        ]

    return MockGateway.canned(source)


__all__ = [
    "max_turns_exhausted",
    "single_tool_happy",
    "tool_error_recovery",
]
