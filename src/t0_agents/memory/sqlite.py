"""SQLite-backed MemoryProvider implementation.

Requires the ``memory-sqlite`` extra::

    pip install 't0-agents[memory-sqlite]'
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from t0_agents.memory.base import MemoryRecord

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memory_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    turns INTEGER NOT NULL,
    inference_ids TEXT NOT NULL,
    extra TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
)"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_memory_episode
ON memory_records(episode_id)"""


class SqliteMemory:
    """Persistent, SQLite-backed memory backend.

    Each method opens its own connection so concurrent ``Agent.run()``
    calls do not contend on a shared connection object.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._table_ready = False

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        if self._table_ready:
            return
        await db.execute(_CREATE_TABLE)
        await db.execute(_CREATE_INDEX)
        await db.commit()
        self._table_ready = True

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        """Return all records for *episode_id* ordered by creation time."""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT episode_id, user_message, agent_response, turns, "
                "inference_ids, extra, created_at "
                "FROM memory_records WHERE episode_id = ? "
                "ORDER BY id ASC",
                (episode_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    async def save(self, record: MemoryRecord) -> None:
        """Persist *record* as a new row."""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            await db.execute(
                "INSERT INTO memory_records "
                "(episode_id, user_message, agent_response, turns, "
                "inference_ids, extra, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.episode_id,
                    record.user_message,
                    record.agent_response,
                    record.turns,
                    json.dumps(record.inference_ids),
                    json.dumps(record.extra),
                    record.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Case-insensitive substring search, most recent matches first."""
        pattern = f"%{query}%"
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT episode_id, user_message, agent_response, turns, "
                "inference_ids, extra, created_at "
                "FROM memory_records "
                "WHERE user_message LIKE ? COLLATE NOCASE "
                "OR agent_response LIKE ? COLLATE NOCASE "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: Any) -> MemoryRecord:
        inference_ids_raw: list[str] = json.loads(row[4])
        extra_raw: dict[str, Any] = json.loads(row[5])
        return MemoryRecord(
            episode_id=row[0],
            user_message=row[1],
            agent_response=row[2],
            turns=row[3],
            inference_ids=inference_ids_raw,
            extra=extra_raw,
            created_at=datetime.fromisoformat(row[6]).replace(
                tzinfo=UTC,
            ),
        )
