"""``SqliteUsageLedger`` — rows read back by value with a plain SQLite client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

from t0_agents.usage.sqlite import (  # noqa: E402
    _CREATE_INDEX,
    _CREATE_INFERENCES,
    _CREATE_RUNS,
    SqliteUsageLedger,
)
from tests.test_usage._records import record  # noqa: E402

_RUN_COLUMNS = (
    "run_id, episode_id, function_name, tags, turns, outcome, error_class, "
    "input_tokens, output_tokens, cost, cost_is_priced, "
    "provider_cache_read_input_tokens, provider_cache_write_input_tokens, "
    "started_at, finished_at"
)
_INFERENCE_COLUMNS = (
    "inference_id, run_id, episode_id, function_name, variant_name, "
    "input_tokens, output_tokens, provider_cache_read_input_tokens, "
    "provider_cache_write_input_tokens, cost, tool_calls, finish_reason, at"
)
_TOOL_CALL_COLUMNS = (
    "run_id, ordinal, inference_id, tool_call_id, name, server, duration_ms, is_error, "
    "status, error_code"
)
_TOOL_CALL_ROWS = [
    ("run-1", 0, "run-1-inf-1", "tc-1", "add", "calc", 12.5, 0, "error", "rate_limited"),
    ("run-1", 1, "run-1-inf-1", "tc-2", "ghost", None, None, 1, None, None),
]
_CREATE_TOOL_CALLS_BEFORE_STATUS = """\
CREATE TABLE usage_tool_calls (
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


async def _file_before_status(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        for ddl in (_CREATE_RUNS, _CREATE_INFERENCES, _CREATE_INDEX):
            await db.execute(ddl)
        await db.execute(_CREATE_TOOL_CALLS_BEFORE_STATUS)
        await db.execute(
            "INSERT INTO usage_tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("run-0", 0, "run-0-inf-1", "tc-0", "add", "calc", 1.0, 0),
        )
        await db.commit()


async def _rows(db_path: str, table: str, columns: str, order: str) -> list[tuple[Any, ...]]:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(f"SELECT {columns} FROM {table} ORDER BY {order}")
        return [tuple(row) for row in await cursor.fetchall()]


class TestSqliteUsageLedger:
    @pytest.mark.parametrize("db_path", [":memory:", ""])
    def test_a_connection_scoped_database_is_refused_at_construction(
        self, db_path: str
    ) -> None:
        with pytest.raises(ValueError, match="InMemoryUsageLedger"):
            SqliteUsageLedger(db_path)

    async def test_a_run_lands_as_one_run_row_and_one_row_per_inference_and_tool_call(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "usage.db")
        await SqliteUsageLedger(db_path).record(record())

        assert await _rows(db_path, "usage_runs", _RUN_COLUMNS, "run_id") == [
            (
                "run-1", "ep-1", "agent", '{"k": "v"}', 2, "completed", None,
                30, 3, 0.5, 1, 4, 0,
                "2026-09-10T12:00:00+00:00", "2026-09-10T12:00:01+00:00",
            ),
        ]
        assert await _rows(
            db_path, "usage_inferences", _INFERENCE_COLUMNS, "inference_id"
        ) == [
            (
                "run-1-inf-1", "run-1", "ep-1", "agent", "v1",
                10, 1, None, None, None, '["add", "ghost"]', "tool_call",
                "2026-09-10T12:00:00+00:00",
            ),
            (
                "run-1-inf-2", "run-1", "ep-1", "agent", None,
                20, 2, 4, None, 0.5, "[]", "stop",
                "2026-09-10T12:00:00.500000+00:00",
            ),
        ]
        assert await _rows(
            db_path, "usage_tool_calls", _TOOL_CALL_COLUMNS, "ordinal"
        ) == _TOOL_CALL_ROWS

    async def test_a_repeated_run_id_is_kept_once(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "usage.db")
        ledger = SqliteUsageLedger(db_path)
        await ledger.record(record())
        await ledger.record(record())

        assert len(await _rows(db_path, "usage_runs", "run_id", "run_id")) == 1
        assert len(await _rows(db_path, "usage_inferences", "inference_id", "inference_id")) == 2
        assert len(await _rows(db_path, "usage_tool_calls", "ordinal", "ordinal")) == 2

    async def test_a_file_from_before_the_tool_calls_table_gains_it(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "usage.db")
        async with aiosqlite.connect(db_path) as db:
            for ddl in (_CREATE_RUNS, _CREATE_INFERENCES, _CREATE_INDEX):
                await db.execute(ddl)
            await db.commit()

        await SqliteUsageLedger(db_path).record(record())

        assert await _rows(
            db_path, "usage_tool_calls", _TOOL_CALL_COLUMNS, "ordinal"
        ) == _TOOL_CALL_ROWS

    async def test_a_file_from_before_the_status_columns_gains_them(
        self, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "usage.db")
        await _file_before_status(db_path)

        await SqliteUsageLedger(db_path).record(record())

        assert await _rows(
            db_path, "usage_tool_calls", _TOOL_CALL_COLUMNS, "run_id, ordinal"
        ) == [
            ("run-0", 0, "run-0-inf-1", "tc-0", "add", "calc", 1.0, 0, None, None),
            *_TOOL_CALL_ROWS,
        ]

    async def test_concurrent_first_records_upgrade_a_file_once(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "usage.db")
        await _file_before_status(db_path)
        ledger = SqliteUsageLedger(db_path)

        run_ids = [f"run-{n:02}" for n in range(1, 51)]
        await asyncio.gather(*(ledger.record(record(run_id)) for run_id in run_ids))

        assert [r[0] for r in await _rows(db_path, "usage_runs", "run_id", "run_id")] == run_ids

    async def test_concurrent_records_all_land(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "usage.db")
        ledger = SqliteUsageLedger(db_path)
        await asyncio.gather(*(ledger.record(record(f"run-{n}")) for n in range(5)))

        assert [r[0] for r in await _rows(db_path, "usage_runs", "run_id", "run_id")] == [
            f"run-{n}" for n in range(5)
        ]

    async def test_rows_survive_a_new_instance(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "usage.db")
        await SqliteUsageLedger(db_path).record(record("run-1"))
        await SqliteUsageLedger(db_path).record(record("run-2"))

        assert [r[0] for r in await _rows(db_path, "usage_runs", "run_id", "run_id")] == [
            "run-1", "run-2",
        ]
