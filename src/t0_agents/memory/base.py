"""Memory record dataclass and default summary renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class MemoryRecord:
    """One stored interaction.

    The shape is intentionally narrow: enough to rehydrate a conversation
    or render a memory summary, not the full ``ChatInference`` raw log
    (which lives in ClickHouse).
    """

    episode_id: str
    user_message: str
    agent_response: str
    turns: int
    inference_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    extra: dict[str, Any] = field(default_factory=dict)


def render_memory_summary(
    records: list[MemoryRecord],
    *,
    max_records: int = 10,
) -> str:
    """Render memory records into a text summary for prompt injection.

    Returns an empty string when *records* is empty so callers can skip
    injection without an extra branch.
    """
    if not records:
        return ""
    recent = records[-max_records:]
    total = len(records)
    lines: list[str] = [f"Previous conversation ({total} turns):"]
    for r in recent:
        lines.append(f"- User: {r.user_message}")
        lines.append(f"  Assistant: {r.agent_response}")
    return "\n".join(lines)
