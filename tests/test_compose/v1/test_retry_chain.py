"""Tests for :func:`t0_agents.retry_chain`."""

from __future__ import annotations

import pytest

from t0_agents import retry_chain
from tests.test_compose.v1._helpers import MockComposable


async def test_retry_chain_first_failure_cascades_to_next() -> None:
    primary = MockComposable(name="primary", raises=ConnectionError("down"))
    backup = MockComposable(name="backup", value="from_backup")

    result = await retry_chain(primary, backup).invoke("seed").result()
    assert result.value == "from_backup"
    assert primary.invocations == ["seed"]
    assert backup.invocations == ["seed"]


async def test_retry_chain_all_fail_reraises_last_unwrapped() -> None:
    """Final exception is the original (not ComposerError) so callers
    keep their ``except SpecificError`` blocks working."""
    a = MockComposable(name="a", raises=ConnectionError("first"))
    b = MockComposable(name="b", raises=TimeoutError("last"))

    composer = retry_chain(a, b)
    with pytest.raises(TimeoutError, match="last"):
        await composer.invoke("seed").result()


async def test_retry_chain_on_error_false_stops_immediately() -> None:
    primary = MockComposable(name="primary", raises=KeyError("auth failed"))
    backup = MockComposable(name="backup", value="should not be reached")

    composer = retry_chain(
        primary, backup,
        on_error=lambda exc: not isinstance(exc, KeyError),
    )
    with pytest.raises(KeyError, match="auth failed"):
        await composer.invoke("seed").result()
    assert backup.invocations == []


async def test_retry_chain_on_error_true_continues() -> None:
    primary = MockComposable(name="primary", raises=ConnectionError("retry me"))
    backup = MockComposable(name="backup", value="recovered")

    composer = retry_chain(
        primary, backup,
        on_error=lambda exc: isinstance(exc, ConnectionError),
    )
    result = await composer.invoke("seed").result()
    assert result.value == "recovered"


async def test_retry_chain_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"retry_chain.*at least one"):
        await retry_chain().invoke("seed").result()


async def test_retry_chain_success_does_not_invoke_backups() -> None:
    primary = MockComposable(name="primary", value="ok")
    backup = MockComposable(name="backup", value="never")

    result = await retry_chain(primary, backup).invoke("seed").result()
    assert result.value == "ok"
    assert backup.invocations == []
