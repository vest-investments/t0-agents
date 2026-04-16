"""Tests for ``SqliteMemory``."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

aiosqlite = pytest.importorskip("aiosqlite")

from t0_agents.memory.base import MemoryRecord  # noqa: E402
from t0_agents.memory.sqlite import SqliteMemory  # noqa: E402


def _record(
    episode_id: str = "ep-1",
    user: str = "hi",
    assistant: str = "hello",
) -> MemoryRecord:
    return MemoryRecord(
        episode_id=episode_id,
        user_message=user,
        agent_response=assistant,
        turns=1,
        inference_ids=["inf-1"],
        extra={"key": "val"},
    )


class TestSqliteMemory:
    async def test_round_trip(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        rec = _record()
        await mem.save(rec)
        loaded = await mem.load("ep-1")
        assert len(loaded) == 1
        assert loaded[0].user_message == "hi"
        assert loaded[0].agent_response == "hello"
        assert loaded[0].inference_ids == ["inf-1"]
        assert loaded[0].extra == {"key": "val"}

    async def test_load_empty_episode(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        assert await mem.load("nonexistent") == []

    async def test_multiple_episodes_isolated(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        await mem.save(_record(episode_id="ep-A", user="alpha"))
        await mem.save(_record(episode_id="ep-B", user="beta"))

        a = await mem.load("ep-A")
        b = await mem.load("ep-B")
        assert len(a) == 1
        assert a[0].user_message == "alpha"
        assert len(b) == 1
        assert b[0].user_message == "beta"

    async def test_multiple_records_same_episode(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        await mem.save(_record(user="first"))
        await mem.save(_record(user="second"))
        loaded = await mem.load("ep-1")
        assert len(loaded) == 2
        assert loaded[0].user_message == "first"
        assert loaded[1].user_message == "second"

    async def test_search_finds_in_user_message(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        await mem.save(_record(user="Weather in Tokyo"))
        results = await mem.search("tokyo")
        assert len(results) == 1

    async def test_search_finds_in_agent_response(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        await mem.save(_record(assistant="The temperature is 22C"))
        results = await mem.search("temperature")
        assert len(results) == 1

    async def test_search_case_insensitive(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        await mem.save(_record(user="HELLO world"))
        assert len(await mem.search("hello")) == 1
        assert len(await mem.search("WORLD")) == 1

    async def test_search_limit(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        for i in range(10):
            await mem.save(_record(user=f"match {i}"))
        results = await mem.search("match", limit=3)
        assert len(results) == 3

    async def test_search_returns_most_recent_first(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        for i in range(5):
            await mem.save(_record(episode_id=f"ep-{i}", user=f"match {i}"))
        results = await mem.search("match", limit=3)
        # Most recently inserted rows come first.
        assert [r.user_message for r in results] == ["match 4", "match 3", "match 2"]

    async def test_concurrent_saves(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        records = [_record(user=f"msg-{i}") for i in range(5)]
        await asyncio.gather(*(mem.save(r) for r in records))
        loaded = await mem.load("ep-1")
        assert len(loaded) == 5

    async def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist.db")
        mem1 = SqliteMemory(db_path)
        await mem1.save(_record(user="remember me"))

        # New instance, same path
        mem2 = SqliteMemory(db_path)
        loaded = await mem2.load("ep-1")
        assert len(loaded) == 1
        assert loaded[0].user_message == "remember me"

    async def test_created_at_roundtrips(self, tmp_path: Path) -> None:
        mem = SqliteMemory(str(tmp_path / "test.db"))
        rec = _record()
        await mem.save(rec)
        loaded = await mem.load("ep-1")
        assert loaded[0].created_at.tzname() == "UTC"
