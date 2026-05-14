"""Unit tests for the M3 retry orchestration inside ``Agent._run_json``.

Scripts a multi-response ``FakeJsonT0`` to cover every transition in the
three-layer ladder (strict → permissive → retry → none) plus the
per-call override and the ``ConfigurationError`` on a negative retry
budget.  The local fakes duplicate the M1/M2 pattern; M5 will fold them
into the consolidated ``MockGateway``.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    Usage,
)
from tensorzero.types import JsonInferenceOutput

from t0_agents.core.agent import _RETRY_HINT_BASE, Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient

# ---------------------------------------------------------------------------
# JSON fakes — local until M5 generalises MockGateway.
# ---------------------------------------------------------------------------


def _fake_json_response(
    *,
    raw: str | None,
    parsed: dict[str, Any] | None,
    inference_id: UUID | None = None,
    episode_id: UUID | None = None,
) -> JsonInferenceResponse:
    """Build a ``JsonInferenceResponse`` with explicit ``raw`` and ``parsed``.

    Unlike M1/M2's helper, this fake never auto-parses ``raw``: callers
    pass the exact ``(raw, parsed)`` pair they want TZ to have returned,
    which is essential for retry scripts that need invalid-then-valid
    transitions.
    """
    return JsonInferenceResponse(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name="test-variant",
        output=JsonInferenceOutput(raw=raw, parsed=parsed),
        usage=Usage(input_tokens=10, output_tokens=20),
    )


class FakeJsonT0:
    """Scriptable async gateway — pops one response per ``inference()`` call.

    ``calls`` records every captured ``inference()`` kwargs dict so a
    test can assert on (e.g.) ``input.messages`` to verify retry hints
    appear and ``episode_id`` is preserved across attempts.
    """

    def __init__(
        self, responses: list[JsonInferenceResponse | Exception]
    ) -> None:
        self._queue: list[JsonInferenceResponse | Exception] = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def inference(self, **kwargs: Any) -> JsonInferenceResponse:
        captured: dict[str, Any] = dict(kwargs)
        if isinstance(captured.get("input"), dict):
            captured["input"] = dict(captured["input"])
            if isinstance(captured["input"].get("messages"), list):
                captured["input"]["messages"] = [
                    dict(m) if isinstance(m, dict) else m
                    for m in captured["input"]["messages"]
                ]
        self.calls.append(captured)
        if not self._queue:
            raise AssertionError("FakeJsonT0 ran out of canned responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _json_agent(
    *,
    t0: FakeJsonT0,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "extractor",
        "function_name": "extract_customer_info",
        "function_kind": "json",
        "json_mode": "strict",
    }
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryOrchestration:
    async def test_no_retry_when_first_attempt_strict(self) -> None:
        # TZ parsed valid JSON on the first try — no retry, no permissive
        # extraction, single inference id, parse_strategy == "strict".
        t0 = FakeJsonT0(
            [_fake_json_response(raw='{"a": 1}', parsed={"a": 1})]
        )
        agent = _json_agent(t0=t0)
        result = await agent.run("hello")

        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "strict"
        assert result.json_parse_attempts == 0
        assert result.turns == 1
        assert len(result.inference_ids) == 1

    async def test_permissive_recovers_without_retry(self) -> None:
        # TZ returned parsed=None but the raw text wraps JSON in prose.
        # Permissive layer succeeds on attempt 0 — no retry needed.
        t0 = FakeJsonT0(
            [
                _fake_json_response(
                    raw='Sure! Here it is: {"a": 1} — done.',
                    parsed=None,
                )
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.run("hello")

        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "permissive"
        assert result.json_parse_attempts == 0
        assert result.turns == 1
        assert len(result.inference_ids) == 1

    async def test_retry_once_succeeds(self) -> None:
        # Attempt 0: invalid (no recoverable JSON in raw).
        # Attempt 1: TZ parses valid JSON.
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw="totally not json", parsed=None),
                _fake_json_response(raw='{"a": 2}', parsed={"a": 2}),
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.run("hello")

        assert result.parsed == {"a": 2}
        assert result.parse_strategy == "retry"
        assert result.json_parse_attempts == 1
        assert result.turns == 2
        assert len(result.inference_ids) == 2

    async def test_retry_budget_exhausted(self) -> None:
        # Three consecutive failures with max_json_retries=2 → all three
        # attempts run, parse_strategy == "none", parsed is None.
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw="fail one", parsed=None),
                _fake_json_response(raw="fail two", parsed=None),
                _fake_json_response(raw="fail three", parsed=None),
            ]
        )
        agent = _json_agent(t0=t0, max_json_retries=2)
        result = await agent.run("hello")

        assert result.parsed is None
        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 2
        assert result.turns == 3
        assert len(result.inference_ids) == 3
        assert result.response == "fail three"  # raw of final attempt

    async def test_per_call_retry_override_zeroes_budget(self) -> None:
        # Config grants 5 retries; the per-call kwarg overrides to 0.
        # Only one inference is made; permissive layer still tried but
        # has no JSON to extract, so parse_strategy == "none".
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw="fail one", parsed=None),
                _fake_json_response(raw='{"a": 1}', parsed={"a": 1}),
            ]
        )
        agent = _json_agent(t0=t0, max_json_retries=5)
        result = await agent.run("hello", max_json_retries=0)

        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1
        # Second response is left unconsumed — proves retry was skipped.
        assert len(t0._queue) == 1

    async def test_retry_preserves_episode_id(self) -> None:
        # First response carries episode_id=E1; second call must observe
        # episode_id=E1 in its inference kwargs, *not* the caller's None.
        first_ep = uuid4()
        t0 = FakeJsonT0(
            [
                _fake_json_response(
                    raw="bad", parsed=None, episode_id=first_ep
                ),
                _fake_json_response(
                    raw='{"a": 1}', parsed={"a": 1}, episode_id=first_ep
                ),
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.run("hello")

        assert result.episode_id == str(first_ep)
        # First call: caller-supplied episode_id (None) reaches the SDK.
        assert t0.calls[0]["episode_id"] is None
        # Second call: the captured episode_id from response 0 is used.
        # ``StructuredOutput.episode_id`` is ``str`` (stringified UUID),
        # so the kwarg threaded into the SDK is also a string.
        assert t0.calls[1]["episode_id"] == str(first_ep)

    async def test_permissive_disabled_skips_layer(self) -> None:
        # permissive_json_parsing=False + max_json_retries=0 = TZ-pure mode.
        # Even though raw contains recoverable JSON, the permissive
        # layer is bypassed and parse_strategy stays "none".
        t0 = FakeJsonT0(
            [
                _fake_json_response(
                    raw='Here is: {"a": 1}', parsed=None
                ),
            ]
        )
        agent = _json_agent(
            t0=t0, permissive_json_parsing=False, max_json_retries=0
        )
        result = await agent.run("hello")

        assert result.parsed is None
        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1

    def test_negative_max_json_retries_is_config_error(self) -> None:
        with pytest.raises(
            ConfigurationError, match="max_json_retries must be ≥ 0"
        ):
            AgentConfig(
                name="bad",
                function_name="extract",
                function_kind="json",
                json_mode="strict",
                max_json_retries=-1,
            )

    async def test_retry_hint_appears_in_history(self) -> None:
        # The second call must carry the RETRY_HINT_BASE substring as a
        # user-turn message.  Exact match would lock the schema-aware
        # additions, so we only assert the base sentence is present.
        t0 = FakeJsonT0(
            [
                _fake_json_response(raw="fail", parsed=None),
                _fake_json_response(raw='{"a": 1}', parsed={"a": 1}),
            ]
        )
        agent = _json_agent(t0=t0)
        await agent.run("hello")

        second_call_messages = t0.calls[1]["input"]["messages"]
        # Expected shape after one retry:
        #   [original user, failed assistant, retry-hint user]
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"
        assert second_call_messages[1]["role"] == "assistant"
        assert second_call_messages[2]["role"] == "user"
        last_user_content = second_call_messages[-1]["content"]
        assert _RETRY_HINT_BASE in last_user_content
