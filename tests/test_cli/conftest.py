"""Shared fixtures for ``tests/test_cli/``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.test_cli import _factories


@pytest.fixture(autouse=True)
def _reset_scripted_state() -> Iterator[None]:
    """Clear the scripted-factory registries before and after each test."""
    _factories.SCRIPTED_TURNS.clear()
    _factories.SCRIPTED_CALLS.clear()
    _factories.SCRIPTED_EXCEPTIONS.clear()
    yield
    _factories.SCRIPTED_TURNS.clear()
    _factories.SCRIPTED_CALLS.clear()
    _factories.SCRIPTED_EXCEPTIONS.clear()
