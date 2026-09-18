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

_CREATE_TOOL_CALLS = """\
CREATE TABLE IF NOT EXISTS usage_tool_calls (
    run_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    inference_id TEXT,
    tool_call_id TEXT NOT NULL,
    name TEXT NOT NULL,
    server TEXT,
    duration_ms REAL,
    is_error INTEGER NOT NULL,
    PRIMARY KEY (run_id, ordinal)
)"""

_CREATE_TOOL_CALLS_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_usage_tool_calls_inference
ON usage_tool_calls(inference_id)"""

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

_INSERT_TOOL_CALL = """\
INSERT OR IGNORE INTO usage_tool_calls (
    run_id, ordinal, inference_id, tool_call_id, name, server, duration_ms, is_error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


class SqliteUsageLedger:
    """Persistent, SQLite-backed ledger: one row per run, per inference, per tool call.

    Each :meth:`record` opens its own connection and commits the three tables
    in one transaction.  ``usage_tool_calls`` is keyed by ``(run_id, ordinal)``
    rather than by ``inference_id``, which SQLite would not deduplicate when
    ``NULL``; a file created before the table existed gains it on the next
    :meth:`record`.  Inserts are ``INSERT OR IGNORE`` on the primary keys,
    which is how the protocol's idempotency on ``run_id`` is met.  There is
    no read side — query the file with any SQLite client.  ``db_path`` must
    name a file: ``":memory:"`` and ``""`` are connection-scoped databases
    that would vanish with every call, so they are refused at construction.
    """

    def __init__(self, db_path: str) -> None:
        if db_path in ("", ":memory:"):
            raise ValueError(
                f"SqliteUsageLedger(db_path={db_path!r}) opens a connection per "
                f"record() call, so a connection-scoped database would lose "
                f"every row; pass a file path, or use InMemoryUsageLedger."
            )
        self._db_path = db_path
        self._tables_ready = False

    async def _ensure_tables(self, db: aiosqlite.Connection) -> None:
        if self._tables_ready:
            return
        await db.execute(_CREATE_RUNS)
        await db.execute(_CREATE_INFERENCES)
        await db.execute(_CREATE_INDEX)
        await db.execute(_CREATE_TOOL_CALLS)
        await db.execute(_CREATE_TOOL_CALLS_INDEX)
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
            await db.executemany(
                _INSERT_TOOL_CALL,
                [
                    (
                        run.run_id,
                        ordinal,
                        call.inference_id,
                        call.tool_call_id,
                        call.name,
                        call.server,
                        call.duration_ms,
                        int(call.is_error),
                    )
                    for ordinal, call in enumerate(run.tool_calls)
                ],
            )
            await db.commit()
