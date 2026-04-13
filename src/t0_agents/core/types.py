from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """Result of an agent run."""

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
