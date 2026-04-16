"""Tests for ``InMemoryMemory``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from t0_agents.memory.base import MemoryRecord
from t0_agents.memory.in_memory import InMemoryMemory


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
    )


class TestInMemoryMemory:
    async def test_round_trip(self) -> None:
        mem = InMemoryMemory()
        rec = _record()
        await mem.save(rec)
        loaded = await mem.load("ep-1")
        assert len(loaded) == 1
        assert loaded[0].user_message == "hi"
        assert loaded[0].agent_response == "hello"

    async def test_load_empty_episode(self) -> None:
        mem = InMemoryMemory()
        assert await mem.load("nonexistent") == []

    async def test_multiple_episodes_isolated(self) -> None:
        mem = InMemoryMemory()
        await mem.save(_record(episode_id="ep-A", user="alpha"))
        await mem.save(_record(episode_id="ep-B", user="beta"))

        a = await mem.load("ep-A")
        b = await mem.load("ep-B")
        assert len(a) == 1
        assert a[0].user_message == "alpha"
        assert len(b) == 1
        assert b[0].user_message == "beta"

    async def test_multiple_records_same_episode(self) -> None:
        mem = InMemoryMemory()
        await mem.save(_record(user="first"))
        await mem.save(_record(user="second"))
        loaded = await mem.load("ep-1")
        assert len(loaded) == 2
        assert loaded[0].user_message == "first"
        assert loaded[1].user_message == "second"

    async def test_search_finds_in_user_message(self) -> None:
        mem = InMemoryMemory()
        await mem.save(_record(user="Weather in Tokyo"))
        results = await mem.search("tokyo")
        assert len(results) == 1

    async def test_search_finds_in_agent_response(self) -> None:
        mem = InMemoryMemory()
        await mem.save(_record(assistant="The temperature is 22C"))
        results = await mem.search("temperature")
        assert len(results) == 1

    async def test_search_case_insensitive(self) -> None:
        mem = InMemoryMemory()
        await mem.save(_record(user="HELLO world"))
        assert len(await mem.search("hello")) == 1
        assert len(await mem.search("WORLD")) == 1

    async def test_search_limit(self) -> None:
        mem = InMemoryMemory()
        for i in range(10):
            await mem.save(_record(user=f"match {i}"))
        results = await mem.search("match", limit=3)
        assert len(results) == 3

    async def test_search_returns_most_recent_first(self) -> None:
        mem = InMemoryMemory()
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(5):
            await mem.save(MemoryRecord(
                episode_id=f"ep-{i}",
                user_message=f"match {i}",
                agent_response="r",
                turns=1,
                created_at=base + timedelta(minutes=i),
            ))
        results = await mem.search("match", limit=3)
        assert [r.user_message for r in results] == ["match 4", "match 3", "match 2"]

    async def test_concurrent_saves(self) -> None:
        mem = InMemoryMemory()
        records = [_record(user=f"msg-{i}") for i in range(5)]
        await asyncio.gather(*(mem.save(r) for r in records))
        loaded = await mem.load("ep-1")
        assert len(loaded) == 5
