"""Unit tests for :class:`t0_agents.glow.handle.FluentRun`.

Cases (no TZ gateway): plain ``await``, ``.json()`` / ``.json(M)``
override, ``.json(A).json(B)`` last-wins + DEBUG log, ``.text().json(M)``
swap, ``.events()`` defers the build to first consumption + ``await``
reuses, the ``prepare`` auto-build hook (M7 seam), and the ``result()``
escape hatch.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow.handle import FluentRun
from t0_agents.stream.events import AgentEvent, DoneEvent


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
        # Escape-hatch call recorders (M2 delegation tests).
        self.feedback_calls: list[tuple[str, Any, str, str | None]] = []
        self.fork_calls: list[tuple[str, Any]] = []
        # M5 streaming-surface recorders.
        self.events_on_error_calls: list[str] = []
        self.stream_text_fallback_calls: list[bool] = []

    async def result(self) -> AgentResult[Any]:
        return AgentResult(
            value=self._value,
            episode_id=self._episode_id,
            inference_ids=list(self._inference_ids),
            turns=1,
        )

    async def events(self, *, on_error: str = "raise") -> AsyncIterator[AgentEvent]:
        # Empty stream — exercises the iterator wiring without coupling
        # the FluentRun tests to the real EventBuffer / drive task.
        self.events_on_error_calls.append(on_error)
        if False:  # pragma: no cover - generator-shape marker
            yield

    async def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]:
        self.stream_text_fallback_calls.append(fallback)
        yield "chunk"


    # Escape-hatch surface consumed by FluentRun.cost/usage/feedback/fork.
    def cost(self) -> float:
        return 0.42

    def usage(self) -> None:
        return None

    async def feedback(
        self,
        name: str,
        value: Any,
        *,
        target: str = "episode",
        inference_id: str | None = None,
    ) -> None:
        self.feedback_calls.append((name, value, target, inference_id))

    async def fork(self, extra_message: str, *, agent: Any = None) -> Run[Any]:
        self.fork_calls.append((extra_message, agent))
        return cast(Run[Any], self)


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


def _make_capturing_build(
    calls: list[tuple[OutputMode[Any], dict[str, Any]]],
    fake_run: _FakeRun | None = None,
) -> Any:
    """Build closure recording ``(mode, call_kwargs)`` per invocation.

    Mirrors ``App._invoke``'s ``build(mode, **call_kwargs)`` shape so the
    ``.with_()`` forwarding + composition can be asserted without a gateway.
    """
    fake = fake_run if fake_run is not None else _FakeRun()

    def build(mode: OutputMode[Any], **call_kwargs: Any) -> Run[Any]:
        calls.append((mode, call_kwargs))
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


async def test_events_defers_build_and_await_reuses_run() -> None:
    """M7 boot collapse: the build fires on first *consumption*, not at
    the ``.events()`` call — and every consumer shares the one Run."""
    captured: list[OutputMode[Any]] = []
    fake = _FakeRun(value="streamed")
    fr: FluentRun[str] = FluentRun(_make_build(captured, fake), ChatOutput())

    # `.events()` only wraps: no build until the stream is iterated.
    iterator = fr.events()
    assert captured == []

    # Draining the (empty) stream is the first consumption — it builds.
    async for _ in iterator:  # pragma: no cover - empty stream
        pass
    assert len(captured) == 1

    val = await fr
    assert val == "streamed"
    # The build closure must not have fired a second time.
    assert len(captured) == 1


async def test_prepare_hook_awaited_once_before_build() -> None:
    """The M7 seam: ``prepare`` runs before the build closure, exactly once."""
    order: list[str] = []
    fake = _FakeRun(value="ready")

    async def prepare() -> None:
        order.append("prepare")

    def build(mode: OutputMode[Any], **call_kwargs: Any) -> Run[Any]:
        order.append("build")
        return cast(Run[Any], fake)

    fr: FluentRun[str] = FluentRun(build, ChatOutput(), prepare=prepare)

    assert await fr == "ready"
    assert order == ["prepare", "build"]

    # Second consumption joins the cached Run — no second prepare/build.
    await fr.result()
    assert order == ["prepare", "build"]


async def test_prepare_hook_propagates_through_with_and_output_overrides() -> None:
    order: list[str] = []
    fake = _FakeRun(value='{"value_a": "x"}')

    async def prepare() -> None:
        order.append("prepare")

    def build(mode: OutputMode[Any], **call_kwargs: Any) -> Run[Any]:
        order.append("build")
        return cast(Run[Any], fake)

    fr: FluentRun[str] = FluentRun(build, ChatOutput(), prepare=prepare)

    # .with_() and .json(M) each spawn a NEW FluentRun — the hook must ride along.
    chained = fr.with_(episode_id="ep_1").json(_ModelA)
    await chained.result()
    assert order == ["prepare", "build"]


async def test_events_selector_validation_stays_eager_and_skips_prepare() -> None:
    """A bad selector raises at the ``.events()`` call — before any boot."""
    prepared: list[bool] = []

    async def prepare() -> None:  # pragma: no cover - must not run
        prepared.append(True)

    fr: FluentRun[str] = FluentRun(_make_build(), ChatOutput(), prepare=prepare)

    with pytest.raises(TypeError, match="expected Selector instances"):
        fr.events(cast(Any, str))
    assert prepared == []


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


# ---------------------------------------------------------------------------
# M2: .with_() per-call overrides
# ---------------------------------------------------------------------------


async def test_with_forwards_and_merges_call_kwargs() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(_make_capturing_build(calls), ChatOutput())

    await fr.with_(episode_id="e1", tags={"customer_id": "c1"}).with_(
        prior_messages=[{"role": "user", "content": "prior"}], episode_id="e2"
    )

    assert len(calls) == 1
    _mode, call_kwargs = calls[0]
    # Later .with_() merges; the repeated key (episode_id) is last-wins.
    assert call_kwargs == {
        "episode_id": "e2",
        "tags": {"customer_id": "c1"},
        "prior_messages": [{"role": "user", "content": "prior"}],
    }


async def test_with_does_not_mutate_original() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(_make_capturing_build(calls), ChatOutput())

    fr.with_(episode_id="e1")  # returns a fresh instance; original untouched
    await fr

    assert calls[0][1] == {}


async def test_with_then_json_keeps_call_kwargs_and_output() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(_make_capturing_build(calls), ChatOutput())

    await fr.with_(episode_id="e1").json(_ModelA)

    mode, call_kwargs = calls[0]
    assert isinstance(mode, StructuredOutput)
    assert mode.model is _ModelA
    assert call_kwargs == {"episode_id": "e1"}


async def test_json_then_with_keeps_call_kwargs_and_output() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(_make_capturing_build(calls), ChatOutput())

    await fr.json(_ModelA).with_(episode_id="e1")

    mode, call_kwargs = calls[0]
    assert isinstance(mode, StructuredOutput)
    assert mode.model is _ModelA
    assert call_kwargs == {"episode_id": "e1"}


async def test_with_preserves_value_transform() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(
        _make_capturing_build(calls, _FakeRun(value="raw")),
        ChatOutput(),
        value_transform=lambda v: f"T({v})",
    )

    result = await fr.with_(episode_id="e1").result()

    assert result.value == "T(raw)"
    assert calls[0][1] == {"episode_id": "e1"}


def test_with_rejects_unknown_kwarg() -> None:
    fr: FluentRun[str] = FluentRun(_make_capturing_build([]), ChatOutput())

    with pytest.raises(ConfigurationError, match="unknown per-call override"):
        fr.with_(max_turns=5)


# ---------------------------------------------------------------------------
# M2: result() carries every AgentResult field via dataclasses.replace
# ---------------------------------------------------------------------------


async def test_result_replace_preserves_all_fields() -> None:
    fake = _FakeRun(value="raw", episode_id="ep_9", inference_ids=["a", "b"])
    fr: FluentRun[str] = FluentRun(
        _make_build(fake_run=fake),
        ChatOutput(),
        value_transform=lambda v: v.upper(),
    )

    result = await fr.result()

    assert result.value == "RAW"  # transformed
    assert result.episode_id == "ep_9"  # every other field carried over
    assert result.inference_ids == ["a", "b"]
    assert result.turns == 1
    assert result.response_source == "text"
    assert result.guardrail_log == []
    assert result.diagnostics is None


# ---------------------------------------------------------------------------
# M2: public escape hatch — cost/usage/feedback/fork delegate to the Run
# ---------------------------------------------------------------------------


def test_escape_hatch_before_run_raises() -> None:
    fr: FluentRun[str] = FluentRun(_make_build(), ChatOutput())

    with pytest.raises(ConfigurationError, match="only after the run has been consumed"):
        fr.cost()


async def test_escape_hatch_delegates_after_run() -> None:
    fake = _FakeRun(value="v")
    fr: FluentRun[str] = FluentRun(_make_build(fake_run=fake), ChatOutput())

    await fr  # consume → underlying Run built

    assert fr.cost() == 0.42
    assert fr.usage() is None
    forked = await fr.fork("continue")
    await fr.feedback("thumbs_up", 1.0, target="inference")

    assert forked is not None
    assert fake.fork_calls == [("continue", None)]
    assert fake.feedback_calls == [("thumbs_up", 1.0, "inference", None)]


# ---------------------------------------------------------------------------
# M5 streaming surface — on_error flip, stream_text, statuses
# ---------------------------------------------------------------------------


async def test_events_defaults_to_on_error_event() -> None:
    """Glow flips the core default: a UI stream never raises mid-iteration."""
    fake = _FakeRun()
    fr: FluentRun[str] = FluentRun(_make_build(fake_run=fake), ChatOutput())
    async for _ in fr.events():  # pragma: no cover - empty stream
        pass
    assert fake.events_on_error_calls == ["event"]


async def test_events_on_error_raise_is_forwarded() -> None:
    fake = _FakeRun()
    fr: FluentRun[str] = FluentRun(_make_build(fake_run=fake), ChatOutput())
    async for _ in fr.events(on_error="raise"):  # pragma: no cover - empty stream
        pass
    assert fake.events_on_error_calls == ["raise"]


async def test_stream_text_builds_lazily_and_forwards_fallback() -> None:
    captured: list[OutputMode[Any]] = []
    fake = _FakeRun()
    fr: FluentRun[str] = FluentRun(_make_build(captured, fake), ChatOutput())

    chunks = [chunk async for chunk in fr.stream_text(fallback=False)]

    assert chunks == ["chunk"]
    assert len(captured) == 1
    assert fake.stream_text_fallback_calls == [False]

    # A second consumption joins the cached Run — no rebuild.
    _ = fr.stream_text()
    assert len(captured) == 1


async def test_statuses_decomposes_with_known_servers() -> None:
    from t0_agents.stream.events import ToolCallEvent
    from t0_agents.stream.status import Status

    fake = _FakeRun()
    call = ToolCallEvent(id="c1", name="fred_get_series", arguments={})

    async def _events(*, on_error: str = "raise") -> Any:
        fake.events_on_error_calls.append(on_error)
        yield call

    fake.events = _events  # type: ignore[method-assign]
    fr: FluentRun[str] = FluentRun(
        _make_build(fake_run=fake), ChatOutput(), known_servers=("fred",)
    )

    items = [item async for item in fr.statuses()]

    assert items == [
        call,
        Status(text="Calling fred_get_series…", server="fred", tool="get_series"),
    ]
    assert fake.events_on_error_calls == ["event"]


async def test_with_and_spawn_propagate_known_servers() -> None:
    fr: FluentRun[str] = FluentRun(
        _make_build(), ChatOutput(), known_servers=("fred", "edgar")
    )
    assert fr.with_(episode_id="e1")._known_servers == ("fred", "edgar")
    assert fr.json()._known_servers == ("fred", "edgar")


# ---------------------------------------------------------------------------
# M8: on_result hook + consumed
# ---------------------------------------------------------------------------


class _DoneRun:
    """Fake Run whose ``events()`` ends with a terminal ``DoneEvent`` (success).

    ``_FakeRun`` streams no ``DoneEvent``, so it exercises the failure-shaped
    path (no fire); this one drives the streamed on_result path.
    """

    def __init__(self, *, value: str = "done-value") -> None:
        self._agent_result: AgentResult[Any] = AgentResult(
            value=value, episode_id="ep_done", inference_ids=["if"], turns=1
        )

    async def result(self) -> AgentResult[Any]:
        return self._agent_result

    async def events(self, *, on_error: str = "raise") -> AsyncIterator[AgentEvent]:
        yield DoneEvent(result=self._agent_result)

    async def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]:
        yield "chunk"


def _done_build(fake: _DoneRun) -> Any:
    def build(mode: OutputMode[Any], **call_kwargs: Any) -> Run[Any]:
        return cast(Run[Any], fake)

    return build


async def test_consumed_is_false_until_consumption() -> None:
    fr: FluentRun[str] = FluentRun(_make_build(), ChatOutput())
    assert fr.consumed is False
    await fr.result()
    assert fr.consumed is True


async def test_consumed_reads_across_with_derivations() -> None:
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(_make_capturing_build(calls), ChatOutput())
    derived = fr.with_(tags={"customer_id": "c1"})
    assert fr.consumed is False
    await derived.result()
    assert derived.consumed is True
    assert fr.consumed is True  # consumption is a family property, not per-handle


async def test_consumed_reads_across_output_derivations() -> None:
    fr: FluentRun[str] = FluentRun(_make_build(), ChatOutput())
    await fr.text().result()
    assert fr.consumed is True


async def test_consumed_does_not_leak_across_independent_runs() -> None:
    first: FluentRun[str] = FluentRun(_make_build(), ChatOutput())
    second: FluentRun[str] = FluentRun(_make_build(), ChatOutput())
    await first.result()
    assert first.consumed is True
    assert second.consumed is False


async def test_on_result_fires_on_result() -> None:
    seen: list[AgentResult[Any]] = []
    fr: FluentRun[str] = FluentRun(
        _make_build(fake_run=_FakeRun(value="v")), ChatOutput(), on_result=seen.append
    )
    await fr.result()
    assert [r.value for r in seen] == ["v"]


async def test_on_result_fires_on_stream_text_tail() -> None:
    seen: list[AgentResult[Any]] = []
    fr: FluentRun[str] = FluentRun(
        _make_build(fake_run=_FakeRun(value="v")), ChatOutput(), on_result=seen.append
    )
    chunks = [c async for c in fr.stream_text()]
    assert chunks == ["chunk"]
    assert len(seen) == 1


async def test_on_result_fires_on_events_done() -> None:
    seen: list[AgentResult[Any]] = []
    fake = _DoneRun(value="dv")
    fr: FluentRun[str] = FluentRun(
        _done_build(fake), ChatOutput(), on_result=seen.append
    )
    events = [e async for e in fr.events()]
    assert any(isinstance(e, DoneEvent) for e in events)
    assert [r.value for r in seen] == ["dv"]


async def test_on_result_fires_exactly_once_across_paths() -> None:
    seen: list[AgentResult[Any]] = []
    fr: FluentRun[str] = FluentRun(
        _done_build(_DoneRun(value="v")), ChatOutput(), on_result=seen.append
    )
    async for _ in fr.events():
        pass
    await fr.result()  # joins the cached Run — latch prevents a second fire
    assert len(seen) == 1


async def test_on_result_rides_through_with_and_output_override() -> None:
    seen: list[AgentResult[Any]] = []
    calls: list[tuple[OutputMode[Any], dict[str, Any]]] = []
    fr: FluentRun[str] = FluentRun(
        _make_capturing_build(calls, fake_run=_FakeRun(value="v")),
        ChatOutput(),
        on_result=seen.append,
    )
    # Both .with_() and the output override spawn fresh FluentRuns — the hook
    # must survive, or a conv.send(x).with_(...).text() chain stops recording.
    await fr.with_(episode_id="e1").text().result()
    assert len(seen) == 1
