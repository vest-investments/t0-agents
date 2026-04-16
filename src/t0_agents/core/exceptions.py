from __future__ import annotations


class T0AgentsError(Exception):
    """Base class for every exception raised by t0-agents."""


class MaxTurnsExceededError(T0AgentsError):
    """Raised when the agent loop exceeds the maximum number of turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")


class ConfigurationError(T0AgentsError):
    """Raised when AgentConfig or Agent construction parameters are invalid."""
