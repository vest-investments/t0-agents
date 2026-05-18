"""Unit tests for the glow ``Annotated[T, Guardrail(...)]`` extraction.

The :func:`~t0_agents.guardrails.regex.regex_guardrail` factory itself
is tested at the canonical v1.0 home in
``tests/test_guardrails/test_regex.py``; this module only exercises
the glow-specific Annotated-walk that harvests guardrail metadata
from ``@app.agent`` parameter and return annotations.

The integration test (live LLM blocked on a regex match) lives in
``tests/test_glow/test_integration.py`` so it joins the marked-
integration suite.
"""

from __future__ import annotations

from typing import Annotated, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, regex_guardrail
from t0_agents.glow.guardrails_typing import _extract_guardrails

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


_NO_SSN = regex_guardrail(SSN_PATTERN, "no SSN")
_TRIVIAL_GUARD = regex_guardrail(r"^banned$", "banned word", phase="before_run")


# ---------------------------------------------------------------------------
# _extract_guardrails — annotation walk
# ---------------------------------------------------------------------------


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
