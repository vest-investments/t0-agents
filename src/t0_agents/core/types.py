from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentResult:
    """Result of an agent run."""

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
