"""SQLite-backed UsageLedger implementation.

Requires the ``memory-sqlite`` extra::

    pip install 't0-agents[memory-sqlite]'
"""

from __future__ import annotations

import json

import aiosqlite

from t0_agents.usage.records import RunUsageRecord

_CREATE_RUNS = """\
CREATE TABLE IF NOT EXISTS usage_runs (
    run_id TEXT PRIMARY KEY,
    episode_id TEXT,
    function_name TEXT,
    tags TEXT NOT NULL,
    turns INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    error_class TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    cost_is_priced INTEGER NOT NULL,
    provider_cache_read_input_tokens INTEGER NOT NULL,
    provider_cache_write_input_tokens INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL
)"""

_CREATE_INFERENCES = """\
CREATE TABLE IF NOT EXISTS usage_inferences (
    inference_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    function_name TEXT,
    variant_name TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    provider_cache_read_input_tokens INTEGER,
    provider_cache_write_input_tokens INTEGER,
    cost REAL,
    tool_calls TEXT NOT NULL,
    finish_reason TEXT,
    at TEXT NOT NULL
)"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_usage_inferences_run
ON usage_inferences(run_id)"""

_INSERT_RUN = """\
INSERT OR IGNORE INTO usage_runs (
    run_id, episode_id, function_name, tags, turns, outcome, error_class,
    input_tokens, output_tokens, cost, cost_is_priced,
    provider_cache_read_input_tokens, provider_cache_write_input_tokens,
    started_at, finished_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_INSERT_INFERENCE = """\
INSERT OR IGNORE INTO usage_inferences (
    inference_id, run_id, episode_id, function_name, variant_name,
    input_tokens, output_tokens, provider_cache_read_input_tokens,
    provider_cache_write_input_tokens, cost, tool_calls, finish_reason, at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


class SqliteUsageLedger:
    """Persistent, SQLite-backed ledger: one row per run, one per inference.

    Each :meth:`record` opens its own connection and commits both tables in
    one transaction.  Inserts are ``INSERT OR IGNORE`` on the primary keys,
    which is how the protocol's idempotency on ``run_id`` is met.  There is
    no read side — query the file with any SQLite client.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._tables_ready = False

    async def _ensure_tables(self, db: aiosqlite.Connection) -> None:
        if self._tables_ready:
            return
        await db.execute(_CREATE_RUNS)
        await db.execute(_CREATE_INFERENCES)
        await db.execute(_CREATE_INDEX)
        await db.commit()
        self._tables_ready = True

    async def record(self, run: RunUsageRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                _INSERT_RUN,
                (
                    run.run_id,
                    run.episode_id,
                    run.function_name,
                    json.dumps(dict(run.tags)),
                    run.turns,
                    run.outcome,
                    run.error_class,
                    run.usage.input_tokens,
                    run.usage.output_tokens,
                    run.usage.cost,
                    int(run.usage.cost_is_priced),
                    run.usage.provider_cache_read_input_tokens,
                    run.usage.provider_cache_write_input_tokens,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat(),
                ),
            )
            await db.executemany(
                _INSERT_INFERENCE,
                [
                    (
                        inference.inference_id,
                        run.run_id,
                        inference.episode_id,
                        inference.function_name,
                        inference.variant_name,
                        inference.usage.input_tokens,
                        inference.usage.output_tokens,
                        inference.usage.provider_cache_read_input_tokens,
                        inference.usage.provider_cache_write_input_tokens,
                        inference.usage.cost,
                        json.dumps(list(inference.tool_calls)),
                        inference.finish_reason,
                        inference.at.isoformat(),
                    )
                    for inference in run.inferences
                ],
            )
            await db.commit()
