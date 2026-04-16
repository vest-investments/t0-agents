"""Tests for ``render_memory_summary``."""

from __future__ import annotations

from t0_agents.memory.base import MemoryRecord, render_memory_summary


def _record(user: str = "hi", assistant: str = "hello") -> MemoryRecord:
    return MemoryRecord(
        episode_id="ep-1",
        user_message=user,
        agent_response=assistant,
        turns=1,
    )


class TestRenderMemorySummary:
    def test_empty_records_returns_empty(self) -> None:
        assert render_memory_summary([]) == ""

    def test_single_record(self) -> None:
        result = render_memory_summary([_record("What time?", "3 PM")])
        assert "Previous conversation (1 turns):" in result
        assert "- User: What time?" in result
        assert "  Assistant: 3 PM" in result

    def test_multiple_records_all_shown(self) -> None:
        records = [_record(f"q{i}", f"a{i}") for i in range(3)]
        result = render_memory_summary(records)
        assert "Previous conversation (3 turns):" in result
        for i in range(3):
            assert f"- User: q{i}" in result
            assert f"  Assistant: a{i}" in result

    def test_max_records_truncation(self) -> None:
        records = [_record(f"q{i}", f"a{i}") for i in range(15)]
        result = render_memory_summary(records, max_records=3)
        # Header still says total count
        assert "(15 turns):" in result
        # Only last 3 records shown
        assert "q12" in result
        assert "q13" in result
        assert "q14" in result
        assert "q0" not in result
        assert "q11" not in result
