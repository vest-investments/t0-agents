"""Unit tests for :func:`t0_agents.guardrails.regex.regex_guardrail`.

Covers Protocol conformance (the ``name`` attribute), phase isolation
(``before_run`` vs ``after_run``), string and non-string
``AgentResult.value`` matching paths, the ``None``-result short-circuit
the middleware relies on after a fatal abort, and the pre-compiled
:class:`re.Pattern` constructor path.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import AgentResult
from t0_agents.guardrails import regex_guardrail
from t0_agents.guardrails.base import GuardrailContext

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"


def _ctx(user_message: str = "") -> GuardrailContext:
    return GuardrailContext(
        agent_name="t",
        function_name="agent",
        user_message=user_message,
    )


def _result(value: Any) -> AgentResult[Any]:
    return AgentResult(
        value=value,
        episode_id="ep_test",
        inference_ids=["if_test"],
        turns=1,
    )


class TestRegexGuardrail:
    def test_default_phase_is_after_run(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        assert guard.name == "regex_after_run"

    def test_explicit_before_run_phase_uses_that_name(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
        assert guard.name == "regex_before_run"

    async def test_after_run_blocks_string_match(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        with pytest.raises(GuardrailBlockedError, match="no SSN"):
            await guard.after_run(_ctx(), _result("Found 123-45-6789 in the file."))

    async def test_after_run_passes_when_no_match(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        await guard.after_run(_ctx(), _result("Nothing sensitive here."))

    async def test_after_run_passes_on_none_result(self) -> None:
        """A prior fatal abort produces ``result=None``; the guardrail
        must not raise on that path (the middleware fires
        ``after_run`` even for fatal aborts so guardrails can log)."""
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        await guard.after_run(_ctx(), None)

    async def test_after_run_scans_dict_results_via_json(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        with pytest.raises(GuardrailBlockedError):
            await guard.after_run(_ctx(), _result({"ssn": "123-45-6789"}))

    async def test_before_run_blocks_user_message_match(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
        with pytest.raises(GuardrailBlockedError):
            await guard.before_run(_ctx(user_message="Look up 123-45-6789 please."))

    async def test_before_run_passes_when_no_match(self) -> None:
        guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
        await guard.before_run(_ctx(user_message="Tell me about photosynthesis."))

    async def test_phases_are_isolated(self) -> None:
        """A guardrail with ``phase="after_run"`` must not fire on
        ``before_run`` even if the user_message matches (and vice
        versa)."""
        guard = regex_guardrail(SSN_PATTERN, "no SSN")
        await guard.before_run(_ctx(user_message="Look up 123-45-6789 please."))
        before_guard = regex_guardrail(SSN_PATTERN, "no SSN", phase="before_run")
        await before_guard.after_run(_ctx(), _result("123-45-6789 leaked"))

    async def test_accepts_precompiled_pattern(self) -> None:
        """Passing a pre-compiled :class:`re.Pattern` avoids the
        :func:`re.compile` call inside the factory; the resulting
        guardrail behaves identically."""
        guard = regex_guardrail(re.compile(SSN_PATTERN), "no SSN")
        with pytest.raises(GuardrailBlockedError):
            await guard.after_run(_ctx(), _result("123-45-6789"))
