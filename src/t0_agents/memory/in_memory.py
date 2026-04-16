"""Ephemeral, dict-backed MemoryProvider implementation."""

from __future__ import annotations

from collections import defaultdict

from t0_agents.memory.base import MemoryRecord


class InMemoryMemory:
    """Ephemeral, dict-backed memory backend.

    Useful for tests and single-process applications that want session
    continuity without cross-process persistence.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryRecord]] = defaultdict(list)

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        """Return all records for *episode_id*, or an empty list."""
        return list(self._store[episode_id])

    async def save(self, record: MemoryRecord) -> None:
        """Append *record* to the episode's memory list."""
        self._store[record.episode_id].append(record)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Case-insensitive substring search across all records."""
        q = query.lower()
        results: list[MemoryRecord] = []
        for records in self._store.values():
            for r in records:
                if q in r.user_message.lower() or q in r.agent_response.lower():
                    results.append(r)
                    if len(results) >= limit:
                        return results
        return results
