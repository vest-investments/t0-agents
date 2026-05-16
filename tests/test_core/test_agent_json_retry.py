"""Unit tests for the JSON retry orchestration in :class:`JsonRetryMiddleware`.

Scripts a multi-response :class:`MockGateway` to cover every transition
in the three-layer ladder (strict → permissive → retry → none) plus the
per-call override and the ``TypeError`` raised when v0.12 legacy
``function_kind=``/``max_json_retries=`` config kwargs leak in.  Uses
``MockGateway.script_json`` for canned JSON responses; see
``tests/test_testing/test_json_helpers.py`` for the dict-shape semantic.

Migrated from v0.12 in M3 — the test moved from exercising
``Agent._run_json`` directly to exercising the
:class:`JsonRetryMiddleware`-driven loop around
:class:`Agent._single_shot_terminal` (the dispatcher is the same code
path; only the shape of the configuration changed).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.outputs import JsonOutput
from t0_agents.core.types import ToolProvider
from t0_agents.testing import MockGateway
from tests._fake_mcp import FakeMCPClient


def _json_agent(
    *,
    t0: MockGateway,
    output: JsonOutput | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "extractor",
        "function_name": "extract_customer_info",
        "output": output if output is not None else JsonOutput(json_mode="strict"),
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
        t0 = MockGateway.script_json(
            [{"raw": '{"a": 1}', "parsed": {"a": 1}}]
        )
        agent = _json_agent(t0=t0)
        result = await agent.invoke("hello")

        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "strict"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1

    async def test_permissive_recovers_without_retry(self) -> None:
        # TZ returned parsed=None but the raw text wraps JSON in prose.
        # Permissive layer succeeds on attempt 0 — no retry needed.
        t0 = MockGateway.script_json(
            [
                {
                    "raw": 'Sure! Here it is: {"a": 1} — done.',
                    "parsed": None,
                }
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.invoke("hello")

        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "permissive"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1

    async def test_retry_once_succeeds(self) -> None:
        # Attempt 0: invalid (no recoverable JSON in raw).
        # Attempt 1: TZ parses valid JSON.
        t0 = MockGateway.script_json(
            [
                {"raw": "totally not json", "parsed": None},
                {"raw": '{"a": 2}', "parsed": {"a": 2}},
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.invoke("hello")

        assert result.parsed == {"a": 2}
        assert result.parse_strategy == "retry"
        assert result.json_parse_attempts == 1
        assert len(result.inference_ids) == 2

    async def test_retry_budget_exhausted(self) -> None:
        # Three consecutive failures with max_retries=2 → all three
        # attempts run, parse_strategy == "none", parsed is None.
        t0 = MockGateway.script_json(
            [
                {"raw": "fail one", "parsed": None},
                {"raw": "fail two", "parsed": None},
                {"raw": "fail three", "parsed": None},
            ]
        )
        agent = _json_agent(
            t0=t0,
            output=JsonOutput(json_mode="strict", max_retries=2),
        )
        result = await agent.invoke("hello")

        assert result.parsed is None
        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 2
        assert len(result.inference_ids) == 3
        assert result.response == "fail three"  # raw of final attempt

    async def test_per_call_retry_override_zeroes_budget(self) -> None:
        # Config grants 5 retries; the per-call output= overrides to 0.
        # Only one inference is made; permissive layer still tried but
        # has no JSON to extract, so parse_strategy == "none".
        t0 = MockGateway.script_json(
            [
                {"raw": "fail one", "parsed": None},
                {"raw": '{"a": 1}', "parsed": {"a": 1}},
            ]
        )
        agent = _json_agent(
            t0=t0, output=JsonOutput(json_mode="strict", max_retries=5)
        )
        # Per-call ``output=JsonOutput(max_retries=0)`` zeroes the retry
        # budget for this invocation only without touching the config.
        result = await agent.invoke(
            "hello", output=JsonOutput(json_mode="strict", max_retries=0)
        )

        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1
        # Only the first scripted response was consumed — proves retry
        # was skipped (the second entry is still in the script queue).
        assert len(t0.calls) == 1

    async def test_retry_preserves_episode_id(self) -> None:
        # First response carries episode_id=E1; second call must observe
        # episode_id=E1 in its inference kwargs, *not* the caller's None.
        first_ep = uuid4()
        t0 = MockGateway.script_json(
            [
                {"raw": "bad", "parsed": None, "episode_id": first_ep},
                {
                    "raw": '{"a": 1}',
                    "parsed": {"a": 1},
                    "episode_id": first_ep,
                },
            ]
        )
        agent = _json_agent(t0=t0)
        result = await agent.invoke("hello")

        assert result.episode_id == str(first_ep)
        # First call: caller-supplied episode_id (None) reaches the SDK.
        assert t0.calls[0]["episode_id"] is None
        # Second call: the captured episode_id from response 0 is used.
        # ``StructuredInferenceResult.episode_id`` is ``str`` (stringified UUID),
        # so the kwarg threaded into the SDK is also a string.
        assert t0.calls[1]["episode_id"] == str(first_ep)

    async def test_permissive_disabled_skips_layer(self) -> None:
        # JsonOutput(permissive=False, max_retries=0) = TZ-pure mode.
        # Even though raw contains recoverable JSON, the permissive
        # layer is bypassed and parse_strategy stays "none".
        t0 = MockGateway.script_json(
            [{"raw": 'Here is: {"a": 1}', "parsed": None}]
        )
        agent = _json_agent(
            t0=t0,
            output=JsonOutput(
                json_mode="strict", permissive=False, max_retries=0
            ),
        )
        result = await agent.invoke("hello")

        assert result.parsed is None
        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 0
        assert len(result.inference_ids) == 1

    def test_legacy_max_json_retries_kwarg_raises_typeerror(self) -> None:
        # v0.12 ``AgentConfig(max_json_retries=...)`` was removed in
        # v1.0 (Tier 2 — contract §2.4).  Passing it now raises
        # ``TypeError`` from the dataclass constructor.
        with pytest.raises(
            TypeError, match="max_json_retries"
        ):
            AgentConfig(
                name="bad",
                function_name="extract",
                max_json_retries=-1,  # type: ignore[call-arg]
            )

    async def test_retry_hint_appears_in_history(self) -> None:
        # The second call must carry the RETRY_HINT_BASE substring as a
        # user-turn message.  Exact match would lock the schema-aware
        # additions, so we only assert the base sentence is present.
        t0 = MockGateway.script_json(
            [
                {"raw": "fail", "parsed": None},
                {"raw": '{"a": 1}', "parsed": {"a": 1}},
            ]
        )
        agent = _json_agent(t0=t0)
        await agent.invoke("hello")

        second_call_messages = t0.calls[1]["input"]["messages"]
        # Expected shape after one retry:
        #   [original user, failed assistant, retry-hint user]
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"
        assert second_call_messages[1]["role"] == "assistant"
        assert second_call_messages[2]["role"] == "user"
        last_user_content = second_call_messages[-1]["content"]
        assert JsonOutput.RETRY_HINT_BASE in last_user_content
