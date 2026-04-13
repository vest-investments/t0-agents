from __future__ import annotations


class MaxTurnsExceededError(Exception):
    """Raised when the agent loop exceeds the maximum number of turns."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")
