"""Unit tests for :func:`t0_agents.glow.guardrails_typing.regex_guardrail`
and the ``Annotated[T, Guardrail(...)]`` extraction integrated into
``@app.agent``.

The integration test (live LLM blocked on a regex match) lives in
``tests/test_glow/test_integration.py`` so it joins the marked-
integration suite.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import AgentResult
from t0_agents.glow import App, regex_guardrail
from t0_agents.glow.guardrails_typing import _extract_guardrails

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


class _StubCtx:
    """Minimal ``GuardrailContext`` stand-in.  The Protocol's hook
    signatures only access ``ctx.user_message`` on ``before_run``; the
    other hooks treat ``ctx`` as opaque.  Mirrors the duck-typing
    pattern used by other glow unit tests."""

    def __init__(self, user_message: str = "") -> None:
        self.user_message = user_message


def _result(value: Any) -> AgentResult[Any]:
    return AgentResult(
        value=value, episode_id="ep_test", inference_ids=["if_test"], turns=1,
    )


# ---------------------------------------------------------------------------
# regex_guardrail — Protocol conformance + direct hook invocation
# ---------------------------------------------------------------------------


def test_regex_guardrail_default_phase_is_after_run() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    assert guard.name == "regex_after_run"


def test_regex_guardrail_explicit_before_run_phase_uses_that_name() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
    assert guard.name == "regex_before_run"


async def test_regex_guardrail_after_run_blocks_string_match() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    with pytest.raises(GuardrailBlockedError, match="no SSN"):
        await guard.after_run(_StubCtx(), _result("Found 123-45-6789 in the file."))


async def test_regex_guardrail_after_run_passes_when_no_match() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    # No raise → pass.
    await guard.after_run(_StubCtx(), _result("Nothing sensitive here."))


async def test_regex_guardrail_after_run_passes_on_none_result() -> None:
    # A prior fatal abort produces ``result=None``; the guardrail must
    # not raise on that path (the middleware fires ``after_run`` even
    # for fatal aborts so guardrails can log).
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    await guard.after_run(_StubCtx(), None)


async def test_regex_guardrail_after_run_scans_dict_results_via_json() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    with pytest.raises(GuardrailBlockedError):
        await guard.after_run(_StubCtx(), _result({"ssn": "123-45-6789"}))


async def test_regex_guardrail_before_run_blocks_user_message_match() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
    with pytest.raises(GuardrailBlockedError):
        await guard.before_run(_StubCtx(user_message="Look up 123-45-6789 please."))


async def test_regex_guardrail_before_run_passes_when_no_match() -> None:
    guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
    await guard.before_run(_StubCtx(user_message="Tell me about photosynthesis."))


async def test_regex_guardrail_phases_are_isolated() -> None:
    """A guardrail with ``phase="after_run"`` must not fire on
    ``before_run`` even if the user_message matches (and vice versa)."""
    guard = regex_guardrail(SSN_PATTERN, "no SSN")
    await guard.before_run(_StubCtx(user_message="Look up 123-45-6789 please."))
    # And vice versa:
    before_guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
    await before_guard.after_run(_StubCtx(), _result("123-45-6789 leaked"))


# ---------------------------------------------------------------------------
# _extract_guardrails — annotation walk
# ---------------------------------------------------------------------------


_NO_SSN = regex_guardrail(SSN_PATTERN, "no SSN")
_TRIVIAL_GUARD = regex_guardrail(r"^banned$", "banned word", phase="before_run")


def test_extract_guardrails_finds_return_annotation_metadata() -> None:
    async def support(query: str) -> Annotated[str, _NO_SSN]:  # type: ignore[empty-body]
        """Support that masks SSNs."""

    guards = _extract_guardrails(support)
    assert _NO_SSN in guards


def test_extract_guardrails_finds_parameter_annotation_metadata() -> None:
    async def support(  # type: ignore[empty-body]
        query: Annotated[str, _TRIVIAL_GUARD],
    ) -> str:
        """Support that blocks the literal banned word."""

    guards = _extract_guardrails(support)
    assert _TRIVIAL_GUARD in guards


def test_extract_guardrails_collects_both_sides() -> None:
    async def support(  # type: ignore[empty-body]
        query: Annotated[str, _TRIVIAL_GUARD],
    ) -> Annotated[str, _NO_SSN]:
        """Support with input+output guardrails."""

    guards = _extract_guardrails(support)
    assert _TRIVIAL_GUARD in guards
    assert _NO_SSN in guards


def test_extract_guardrails_ignores_non_guardrail_metadata() -> None:
    class _Marker:
        """Inert metadata that does NOT satisfy the Guardrail Protocol."""

    async def support(  # type: ignore[empty-body]
        query: Annotated[str, _Marker()],
    ) -> Annotated[str, _Marker()]:
        """Support with non-guardrail metadata only."""

    assert _extract_guardrails(support) == []


def test_extract_guardrails_no_annotations_returns_empty() -> None:
    async def support(query: str) -> str:  # type: ignore[empty-body]
        """Plain support."""

    assert _extract_guardrails(support) == []


# ---------------------------------------------------------------------------
# @app.agent integration: harvested guardrails land on _AgentSpec
# ---------------------------------------------------------------------------


def test_app_agent_decorator_records_guardrails_on_spec() -> None:
    app = App(t0=_fake_t0())

    @app.agent("support")
    async def support(query: str) -> Annotated[str, _NO_SSN]:  # type: ignore[empty-body]
        """Support that masks SSNs."""

    spec = app._agent_registry["support"]
    assert spec.guardrails == [_NO_SSN]


def test_app_agent_decorator_with_no_guardrails_keeps_empty_list() -> None:
    app = App(t0=_fake_t0())

    @app.agent("plain")
    async def plain(query: str) -> str:  # type: ignore[empty-body]
        """Plain agent — no guardrails."""

    spec = app._agent_registry["plain"]
    assert spec.guardrails == []
