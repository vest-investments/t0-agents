"""Integration tests for ``t0-agents`` CLI — require a running gateway.

Skip unless ``docker compose up -d`` is running and ``GROQ_API_KEY`` (or
the equivalent provider key for the configured TensorZero function) is
set.  Marked module-wide with ``pytest.mark.integration`` so
``uv run pytest -m 'not integration'`` excludes them.

The tests spawn the CLI in a subprocess because the ``asyncio.run``
wrapper inside ``run`` / ``serve`` does not compose with
``CliRunner.invoke`` under a running event loop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_SPEC = "examples.10_cli.agent:assistant"


def test_info_subcommand_against_live_example() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        ["uv", "run", "t0-agents", "info", _EXAMPLE_SPEC],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "tui-assistant" in result.stdout
    assert "function_name" in result.stdout


def test_serve_subcommand_health_endpoint() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "t0-agents",
            "serve",
            _EXAMPLE_SPEC,
            "--host",
            "127.0.0.1",
            "--port",
            "18081",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_health("http://127.0.0.1:18081/health", timeout=30)
        response = httpx.get("http://127.0.0.1:18081/health", timeout=5)
        assert response.status_code == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_run_subcommand_single_turn() -> None:
    """Drive one user turn through the real gateway and assert the
    assistant responds with something that contains the arithmetic result.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        ["uv", "run", "t0-agents", "run", _EXAMPLE_SPEC],
        cwd=str(_REPO_ROOT),
        env=env,
        input="What is 2 + 3?\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "tui-assistant" in result.stdout
    assert "5" in result.stdout


def _wait_for_health(url: str, *, timeout: float) -> None:
    deadline = time.time() + timeout
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2)
        except httpx.HTTPError as exc:
            last_exc = exc
        else:
            if response.status_code < 500:
                return
        time.sleep(0.5)
    raise AssertionError(
        f"Health endpoint {url} did not come up in {timeout}s: {last_exc}",
    )
