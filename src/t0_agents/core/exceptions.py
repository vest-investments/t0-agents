from __future__ import annotations


class T0AgentsError(Exception):
    """Base class for every exception raised by t0-agents."""


class T0AgentsFatalError(T0AgentsError):
    """Marker class for exceptions that must abort an ``Agent.run()``.

    ``mcp/loop.py::_execute_single_tool_call`` re-raises this class before
    its broad ``except Exception`` belt-and-braces clause, so guardrail
    hard-blocks (and any future fatal hooks) propagate out of the loop
    instead of being stringified into a tool error.
    """


class MaxTurnsExceededError(T0AgentsError):
    """Raised when the agent loop exceeds the maximum number of turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")


class ConfigurationError(T0AgentsError):
    """Raised when AgentConfig or Agent construction parameters are invalid."""


class GuardrailBlockedError(T0AgentsFatalError):
    """Raised by a guardrail to hard-abort an ``Agent.run()``.

    Short-circuits any tool calls that were about to run in the current
    turn and propagates out of the agent loop so the caller sees a clear
    policy violation instead of a stringified tool error.
    """
