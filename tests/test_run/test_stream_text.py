"""``Run.stream_text(fallback=...)`` — the text one-liner, thought-safe.

Reasoning models route the final answer through thought summaries and
emit no ``TextDeltaEvent`` (``response_source == "thought"``,
``AgentResult.response`` populated).  ``fallback=True`` (default) covers
that path — and any other no-text turn with a non-blank response — by
yielding ``result.response`` once after the stream drains.  The gate is
"no non-blank delta streamed", matching the vest-ai MobileProjector belt
this generalises; the library never invents retry copy for a genuinely
empty turn.
"""

from __future__ import annotations

import pytest

from t0_agents import TextDeltaEvent, ThoughtDeltaEvent, ThoughtEvent
from t0_agents.core.types import AgentResult
from t0_agents.testing import ScriptedAgent


async def _collect(agent: ScriptedAgent, **kwargs: bool) -> list[str]:
    chunks: list[str] = []
    async for chunk in agent.invoke("q").stream_text(**kwargs):
        chunks.append(chunk)
    return chunks


class TestTextPath:
    async def test_deltas_stream_through_without_fallback_duplication(self) -> None:
        agent = ScriptedAgent(
            events=[
                TextDeltaEvent(id="t1", delta="Hello"),
                TextDeltaEvent(id="t1", delta=" world"),
            ],
            result="Hello world",
        )
        assert await _collect(agent) == ["Hello", " world"]

    async def test_whitespace_only_deltas_do_not_latch_saw_text(self) -> None:
        agent = ScriptedAgent(
            events=[TextDeltaEvent(id="t1", delta="  ")],
            result=AgentResult(value="real answer", turns=1),
        )
        assert await _collect(agent) == ["  ", "real answer"]


class TestReasoningFallback:
    def _reasoning_agent(self, response: str = "It's 22°C.") -> ScriptedAgent:
        return ScriptedAgent(
            events=[
                ThoughtDeltaEvent(id="th1", delta="checking the weather"),
                ThoughtEvent(id="th1", text="checking the weather"),
            ],
            result=AgentResult(value=response, turns=1, response_source="thought"),
        )

    async def test_thought_only_run_yields_the_response_once(self) -> None:
        assert await _collect(self._reasoning_agent()) == ["It's 22°C."]

    async def test_fallback_false_restores_the_raw_filter(self) -> None:
        assert await _collect(self._reasoning_agent(), fallback=False) == []

    async def test_blank_response_yields_nothing(self) -> None:
        assert await _collect(self._reasoning_agent(response="   ")) == []

    async def test_non_str_value_without_diagnostics_yields_nothing(self) -> None:
        agent = ScriptedAgent(result=AgentResult(value=123, turns=1))
        assert await _collect(agent) == []


class TestErrorSemantics:
    async def test_fatal_errors_raise_out_of_the_iterator(self) -> None:
        agent = ScriptedAgent(raises=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await _collect(agent)
