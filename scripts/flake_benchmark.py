#!/usr/bin/env python3
"""Flake-rate benchmark: run an integration test suite N times and report
pass/skip/fail rates.

This is a development tool for measuring the effect of reliability mitigations
(Phase 7 and beyond).  It invokes ``uv run pytest`` as a subprocess and parses
the summary line.

Usage::

    uv run python scripts/flake_benchmark.py --runs 20 --suite phase5
    uv run python scripts/flake_benchmark.py --runs 10 -k "test_f5_1"
    uv run python scripts/flake_benchmark.py --runs 5 --suite all --verbose

Requires ``docker compose up -d`` (TensorZero gateway + ClickHouse).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

SUITE_MAP: dict[str, str] = {
    "phase1": "-m integration -k 'test_f1 or test_tool_choice'",
    "phase2": "-m integration -k test_f2",
    "phase3": "-m integration -k test_f3",
    "phase4": "-m integration -k test_f4",
    "phase5": "-m integration -k test_f5",
    "phase6": "-m integration -k test_f6",
    "all": "-m integration",
}

# Matches pytest summary like "5 passed, 2 skipped, 1 failed in 3.45s"
SUMMARY_RE = re.compile(
    r"(?:(\d+) passed)?"
    r"(?:,\s*(\d+) skipped)?"
    r"(?:,\s*(\d+) failed)?"
    r"(?:,\s*(\d+) error)?"
)


@dataclass
class RunResult:
    """Result of a single pytest invocation."""

    passed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int = 0


@dataclass
class Aggregate:
    """Aggregate across N runs."""

    runs: list[RunResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def total_passed(self) -> int:
        return sum(r.passed for r in self.runs)

    @property
    def total_skipped(self) -> int:
        return sum(r.skipped for r in self.runs)

    @property
    def total_failed(self) -> int:
        return sum(r.failed for r in self.runs)

    @property
    def total_errors(self) -> int:
        return sum(r.errors for r in self.runs)

    @property
    def clean_runs(self) -> int:
        """Runs with 0 failures and 0 errors."""
        return sum(1 for r in self.runs if r.failed == 0 and r.errors == 0)


def _parse_summary(output: str) -> RunResult:
    """Extract pass/skip/fail counts from pytest output."""
    result = RunResult()
    for line in output.splitlines():
        stripped = line.strip()
        if "passed" in stripped or "failed" in stripped or "error" in stripped:
            m = re.search(r"(\d+) passed", stripped)
            if m:
                result.passed = int(m.group(1))
            m = re.search(r"(\d+) skipped", stripped)
            if m:
                result.skipped = int(m.group(1))
            m = re.search(r"(\d+) failed", stripped)
            if m:
                result.failed = int(m.group(1))
            m = re.search(r"(\d+) error", stripped)
            if m:
                result.errors = int(m.group(1))
    return result


def _run_once(
    pytest_args: list[str], verbose: bool = False,
) -> RunResult:
    """Invoke pytest and return parsed results."""
    cmd = ["uv", "run", "pytest", "--tb=short", "-q", *pytest_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    result = _parse_summary(proc.stdout + proc.stderr)
    result.exit_code = proc.returncode
    if verbose:
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flake-rate benchmark for integration tests.",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of times to run the suite (default: 10)",
    )
    parser.add_argument(
        "--suite", choices=SUITE_MAP.keys(),
        help="Named test suite to run",
    )
    parser.add_argument(
        "-k", dest="filter", default=None,
        help="pytest -k filter expression (overrides --suite)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full pytest output for each run",
    )
    args = parser.parse_args()

    if args.filter:
        pytest_extra = ["-m", "integration", "-k", args.filter]
    elif args.suite:
        import shlex
        pytest_extra = shlex.split(SUITE_MAP[args.suite])
    else:
        parser.error("Specify --suite or -k")
        return

    print(f"Running {args.runs} iterations: {' '.join(pytest_extra)}")
    print("=" * 60)

    agg = Aggregate()
    for i in range(args.runs):
        result = _run_once(pytest_extra, verbose=args.verbose)
        agg.runs.append(result)
        status = "CLEAN" if result.failed == 0 and result.errors == 0 else "FLAKE"
        print(
            f"  Run {i + 1:3d}/{args.runs}: "
            f"{result.passed} passed, {result.skipped} skipped, "
            f"{result.failed} failed, {result.errors} errors  [{status}]",
        )

    print("=" * 60)
    print(f"Summary ({agg.total} runs):")
    print(f"  Clean runs:  {agg.clean_runs}/{agg.total} "
          f"({agg.clean_runs / agg.total * 100:.0f}%)")
    print(f"  Total passed:  {agg.total_passed}")
    print(f"  Total skipped: {agg.total_skipped}")
    print(f"  Total failed:  {agg.total_failed}")
    print(f"  Total errors:  {agg.total_errors}")

    if agg.total_failed > 0 or agg.total_errors > 0:
        flake_rate = (
            (agg.total - agg.clean_runs) / agg.total * 100
        )
        print(f"  Flake rate:  {flake_rate:.1f}%")
        sys.exit(1)
    else:
        print("  Flake rate:  0%")


if __name__ == "__main__":
    main()
