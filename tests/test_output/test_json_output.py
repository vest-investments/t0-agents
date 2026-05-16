"""Unit tests for :class:`~t0_agents.core.outputs.JsonOutput`."""

from __future__ import annotations

from uuid import uuid4

import pytest
from tensorzero.types import (
    JsonInferenceOutput,
    JsonInferenceResponse,
    Usage,
)

from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    ParseFailure,
)
from t0_agents.core.outputs import JsonOutput


def _make_response(
    *, raw: str, parsed: dict | None = None
) -> JsonInferenceResponse:
    return JsonInferenceResponse(
        inference_id=uuid4(),
        episode_id=uuid4(),
        variant_name="v",
        output=JsonInferenceOutput(raw=raw, parsed=parsed),
        usage=Usage(input_tokens=5, output_tokens=3),
    )


class TestCapabilityFlags:
    def test_expects_tool_loop_is_false(self) -> None:
        assert JsonOutput().expects_tool_loop is False

    def test_supports_streaming_is_false(self) -> None:
        assert JsonOutput().supports_streaming is False

    def test_name_is_json(self) -> None:
        assert JsonOutput().name == "json"


class TestShapeInference:
    def test_shape_inference_sets_json_mode(self) -> None:
        out = JsonOutput(json_mode="strict")
        req = InferenceRequest(
            function_name="extract", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.params == {"chat_completion": {"json_mode": "strict"}}

    def test_shape_inference_sets_output_schema(self) -> None:
        schema = {"type": "object", "required": ["name"]}
        out = JsonOutput(schema=schema, json_mode="strict")
        req = InferenceRequest(
            function_name="extract", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.output_schema == schema

    def test_shape_inference_omits_schema_when_none(self) -> None:
        out = JsonOutput(json_mode="strict")
        req = InferenceRequest(
            function_name="extract", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.output_schema is None

    def test_shape_inference_preserves_other_params(self) -> None:
        out = JsonOutput(json_mode="on")
        req = InferenceRequest(
            function_name="extract",
            input={"messages": []},
            params={"chat_completion": {"temperature": 0.5}},
        )
        result = out.shape_inference(req)
        assert result.params == {
            "chat_completion": {"temperature": 0.5, "json_mode": "on"}
        }


class TestMaterialize:
    async def test_materialize_strict_path_on_first_attempt(self) -> None:
        out = JsonOutput()
        response = _make_response(
            raw='{"a": 1}', parsed={"a": 1}
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == {"a": 1}
        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "strict"
        assert result.json_parse_attempts == 0

    async def test_materialize_permissive_recovers_from_wrapped_json(self) -> None:
        out = JsonOutput(permissive=True)
        response = _make_response(
            raw='Here you go: {"a": 1} — done.', parsed=None
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == {"a": 1}
        assert result.parse_strategy == "permissive"

    async def test_materialize_returns_none_on_total_failure(self) -> None:
        out = JsonOutput(permissive=True)
        response = _make_response(raw="totally not json", parsed=None)
        result = await out.materialize(response, attempts=[])
        assert result.value is None
        assert result.parse_strategy == "none"
        assert result.diagnostics is not None
        assert result.diagnostics.raw_text == "totally not json"

    async def test_materialize_skips_permissive_when_disabled(self) -> None:
        out = JsonOutput(permissive=False)
        response = _make_response(
            raw='Wrapped: {"a": 1}', parsed=None
        )
        result = await out.materialize(response, attempts=[])
        assert result.value is None
        assert result.parse_strategy == "none"

    async def test_materialize_uses_retry_strategy_after_prior_attempt(self) -> None:
        out = JsonOutput()
        response = _make_response(
            raw='{"a": 2}', parsed={"a": 2}
        )
        prior_attempt = Attempt(
            inference_id=str(uuid4()),
            raw_text="bad",
            parsed=None,
            usage=None,
        )
        result = await out.materialize(response, attempts=[prior_attempt])
        # parse_strategy upgrades to "retry" when attempts is non-empty.
        assert result.parse_strategy == "retry"
        assert result.json_parse_attempts == 1


class TestRecover:
    async def test_recover_returns_retry_under_budget(self) -> None:
        out = JsonOutput(max_retries=2)
        failure = ParseFailure(
            raw_text="bad",
            inference_id="x",
            reason="parse_failed",
            detail="",
        )
        # 1 attempt so far (≤ max_retries=2) → retry.
        recovery = await out.recover(
            failure,
            attempts=[
                Attempt(inference_id="x", raw_text="bad", parsed=None, usage=None),
            ],
        )
        assert recovery.kind == "retry"
        assert recovery.hint is not None
        assert JsonOutput.RETRY_HINT_BASE in recovery.hint

    async def test_recover_returns_fail_past_budget(self) -> None:
        out = JsonOutput(max_retries=2)
        # 3 attempts (> max_retries=2) → fail.
        attempts = [
            Attempt(inference_id=f"x{i}", raw_text="bad", parsed=None, usage=None)
            for i in range(3)
        ]
        failure = ParseFailure(
            raw_text="bad",
            inference_id="x",
            reason="parse_failed",
            detail="",
        )
        recovery = await out.recover(failure, attempts=attempts)
        assert recovery.kind == "fail"

    def test_build_retry_hint_with_schema_title(self) -> None:
        out = JsonOutput(schema={"title": "Customer", "type": "object"})
        hint = out._build_retry_hint()
        assert "Customer" in hint
        assert JsonOutput.RETRY_HINT_BASE in hint

    def test_build_retry_hint_with_required_fields(self) -> None:
        out = JsonOutput(
            schema={"type": "object", "required": ["name", "email"]}
        )
        hint = out._build_retry_hint()
        assert "'name'" in hint
        assert "'email'" in hint

    def test_build_retry_hint_minimal_when_schema_none(self) -> None:
        out = JsonOutput(schema=None)
        hint = out._build_retry_hint()
        assert hint == JsonOutput.RETRY_HINT_BASE


class TestRejectsWrongResponseType:
    async def test_materialize_rejects_chat_response(self) -> None:
        out = JsonOutput()
        with pytest.raises(TypeError, match=r"JsonOutput\.materialize"):
            await out.materialize("not a response", attempts=[])
