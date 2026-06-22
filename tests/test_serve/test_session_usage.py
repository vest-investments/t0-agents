"""Unit tests for ``SessionUsageTracker`` (serve/session.py).

The tracker keeps one ``ConversationUsage`` per ``episode_id`` for a single
WebSocket connection (Q8: per-episode, not one-per-socket).  A small Run stub
exercises the keying / accumulation / carry-forward logic without spinning up
an Agent — the real-Run path is covered by the multi-turn WS streaming test.
"""

from __future__ import annotations

from t0_agents.core.types import RunUsage, Usage
from t0_agents.serve.session import SessionUsageTracker


class _StubRun:
    """Minimal Run stand-in: the three accessors ``ConversationUsage`` reads."""

    def __init__(
        self,
        *,
        episode_id: str | None,
        usage: RunUsage | None,
        footprint: Usage | None,
    ) -> None:
        self.episode_id = episode_id
        self._usage = usage
        self._footprint = footprint

    def usage(self) -> RunUsage | None:
        return self._usage

    def context_footprint(self) -> Usage | None:
        return self._footprint


def _run(episode_id: str, input_tokens: int, *, cost: float | None = None) -> _StubRun:
    """A completed-Run stub for one turn with the given prompt size."""
    return _StubRun(
        episode_id=episode_id,
        usage=RunUsage(
            input_tokens=input_tokens,
            output_tokens=input_tokens // 10,
            cost=cost or 0.0,
            cost_is_priced=cost is not None,
        ),
        footprint=Usage(
            input_tokens=input_tokens,
            output_tokens=input_tokens // 10,
            cost=cost,
        ),
    )


def test_same_episode_accumulates_billing_footprint_is_level() -> None:
    tracker = SessionUsageTracker()
    acc1 = tracker.record(_run("ep-1", 100))  # type: ignore[arg-type]
    acc2 = tracker.record(_run("ep-1", 220))  # type: ignore[arg-type]

    # Same episode -> same accumulator instance (one running total).
    assert acc1 is acc2
    # Billing sums across turns; footprint is the latest level, not a sum.
    assert acc2.total.input_tokens == 320
    fp = acc2.context_footprint()
    assert fp is not None and fp.input_tokens == 220
    assert acc2.runs_recorded == 2


def test_distinct_episodes_do_not_conflate() -> None:
    tracker = SessionUsageTracker()
    acc_a = tracker.record(_run("ep-a", 100))  # type: ignore[arg-type]
    acc_b = tracker.record(_run("ep-b", 50))  # type: ignore[arg-type]

    assert acc_a is not acc_b
    assert acc_a.total.input_tokens == 100
    assert acc_b.total.input_tokens == 50
    assert acc_a.runs_recorded == 1
    assert acc_b.runs_recorded == 1


def test_footprintless_run_does_not_blank_level() -> None:
    """D3 carry-forward: a Run with no footprint (whole-response cache hit or a
    failure before the first inference) keeps the prior level rather than
    blanking the live compaction signal."""
    tracker = SessionUsageTracker()
    tracker.record(_run("ep-1", 180))  # type: ignore[arg-type]
    acc = tracker.record(  # type: ignore[arg-type]
        _StubRun(
            episode_id="ep-1",
            usage=RunUsage(input_tokens=0, output_tokens=0),
            footprint=None,
        )
    )

    fp = acc.context_footprint()
    assert fp is not None and fp.input_tokens == 180  # carried forward
    assert acc.runs_recorded == 2


def test_none_episode_id_keys_on_empty_string() -> None:
    """A Run whose episode_id is None still records (keyed under '')."""
    tracker = SessionUsageTracker()
    acc = tracker.record(  # type: ignore[arg-type]
        _StubRun(
            episode_id=None,
            usage=RunUsage(input_tokens=42, output_tokens=4),
            footprint=Usage(input_tokens=42, output_tokens=4),
        )
    )

    assert acc.episode_id == ""
    assert acc.total.input_tokens == 42
    assert acc.runs_recorded == 1
