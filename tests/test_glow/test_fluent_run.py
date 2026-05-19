"""Unit tests for :class:`t0_agents.glow.handle.FluentRun`.

Seven cases (no TZ gateway): plain ``await``, ``.json()`` / ``.json(M)``
override, ``.json(A).json(B)`` last-wins + DEBUG log, ``.text().json(M)``
swap, ``.events()`` builds eagerly + ``await`` reuses, and the
``result()`` escape hatch.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow.handle import FluentRun
from t0_agents.stream.events import AgentEvent


class _ModelA(BaseModel):
    value_a: str


class _ModelB(BaseModel):
    value_b: int


class _FakeRun:
    """Duck-typed stand-in for :class:`~t0_agents.core.run.Run`.

    Exposes the slice of the ``Run`` contract :class:`FluentRun`
    actually consumes: ``result()`` (awaitable AgentResult) and
    ``events()`` (async iterator of events).  Tests cast it to
    ``Run[Any]`` to satisfy mypy strict.
    """

    def __init__(
        self,
        *,
        value: Any = "fake-value",
        episode_id: str | None = "ep_fake",
        inference_ids: list[str] | None = None,
    ) -> None:
        self._value = value
        self._episode_id = episode_id
        self._inference_ids = inference_ids if inference_ids is not None else ["if_fake"]

    async def result(self) -> AgentResult[Any]:
        return AgentResult(
            value=self._value,
            episode_id=self._episode_id,
            inference_ids=list(self._inference_ids),
            turns=1,
        )

    async def events(self) -> AsyncIterator[AgentEvent]:
        # Empty stream — exercises the iterator wiring without coupling
        # the FluentRun tests to the real EventBuffer / drive task.
        if False:  # pragma: no cover - generator-shape marker
            yield


def _make_build(
    captured: list[OutputMode[Any]] | None = None,
    fake_run: _FakeRun | None = None,
) -> Any:
    """Return a build closure that records each ``OutputMode`` invocation
    and returns the supplied fake Run (or a default one)."""
    fake = fake_run if fake_run is not None else _FakeRun()
    sink = captured if captured is not None else []

    def build(mode: OutputMode[Any]) -> Run[Any]:
        sink.append(mode)
        return cast(Run[Any], fake)

    return build


async def test_await_returns_unwrapped_value() -> None:
    captured: list[OutputMode[Any]] = []
    fr: FluentRun[str] = FluentRun(_make_build(captured), ChatOutput())

    val = await fr

    assert val == "fake-value"
    assert len(captured) == 1
    assert isinstance(captured[0], ChatOutput)


async def test_json_no_model_uses_json_output() -> None:
    captured: list[OutputMode[Any]] = []
    fr: FluentRun[str] = FluentRun(_make_build(captured), ChatOutput())

    await fr.json()

    assert len(captured) == 1
    assert isinstance(captured[0], JsonOutput)


async def test_json_with_model_uses_structured_output() -> None:
    captured: list[OutputMode[Any]] = []
    fr: FluentRun[str] = FluentRun(_make_build(captured), ChatOutput())

    await fr.json(_ModelA)

    assert len(captured) == 1
    chosen = captured[0]
    assert isinstance(chosen, StructuredOutput)
    assert chosen.model is _ModelA


async def test_json_chain_is_last_wins_with_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: list[OutputMode[Any]] = []
    fr: FluentRun[str] = FluentRun(_make_build(captured), ChatOutput())

    with caplog.at_level(logging.DEBUG, logger="t0_agents.glow.handle"):
        await fr.json(_ModelA).json(_ModelB)

    assert len(captured) == 1
    chosen = captured[0]
    assert isinstance(chosen, StructuredOutput)
    assert chosen.model is _ModelB
    # Second override on an already-overridden FluentRun logs DEBUG;
    # the first .json(_ModelA) on a default FluentRun is silent.
    debug_entries = [
        r for r in caplog.records
        if r.name == "t0_agents.glow.handle" and r.levelname == "DEBUG"
    ]
    assert len(debug_entries) == 1
    assert "last-wins" in debug_entries[0].message


async def test_text_then_json_swaps_freely() -> None:
    captured: list[OutputMode[Any]] = []
    fr: FluentRun[str] = FluentRun(_make_build(captured), ChatOutput())

    await fr.text().json(_ModelA)

    assert len(captured) == 1
    chosen = captured[0]
    assert isinstance(chosen, StructuredOutput)
    assert chosen.model is _ModelA


async def test_events_builds_eagerly_and_await_reuses_run() -> None:
    captured: list[OutputMode[Any]] = []
    fake = _FakeRun(value="streamed")
    fr: FluentRun[str] = FluentRun(_make_build(captured, fake), ChatOutput())

    # `.events()` is a sync method that fires the build closure NOW so
    # the subsequent `await` joins the same cached buffer.
    iterator = fr.events()
    assert len(captured) == 1

    val = await fr
    assert val == "streamed"
    # The build closure must not have fired a second time.
    assert len(captured) == 1

    # Drain the (empty) iterator so the async generator finalises.
    async for _ in iterator:  # pragma: no cover - empty stream
        pass


async def test_result_returns_full_agent_result() -> None:
    fake = _FakeRun(
        value="boundary-text",
        episode_id="ep_42",
        inference_ids=["if_a", "if_b"],
    )
    fr: FluentRun[str] = FluentRun(_make_build(fake_run=fake), ChatOutput())

    result = await fr.result()

    assert isinstance(result, AgentResult)
    assert result.value == "boundary-text"
    assert result.episode_id == "ep_42"
    assert result.inference_ids == ["if_a", "if_b"]
