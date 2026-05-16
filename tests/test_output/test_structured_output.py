"""Unit tests for :class:`~t0_agents.core.outputs.StructuredOutput`."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from tensorzero.types import (
    JsonInferenceOutput,
    JsonInferenceResponse,
    Usage,
)

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    ParseFailure,
)
from t0_agents.core.outputs import StructuredOutput


class Customer(BaseModel):
    name: str | None = Field(None, description="Customer name")
    email: str | None = Field(None, description="Customer email")


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
        out = StructuredOutput(Customer)
        assert out.expects_tool_loop is False

    def test_supports_streaming_is_false(self) -> None:
        out = StructuredOutput(Customer)
        assert out.supports_streaming is False

    def test_name_includes_model(self) -> None:
        out = StructuredOutput(Customer)
        assert out.name == "structured/Customer"


class TestConstruction:
    def test_rejects_non_basemodel_subclass(self) -> None:
        class NotAModel:
            pass

        with pytest.raises(ConfigurationError, match="BaseModel"):
            StructuredOutput(NotAModel)  # type: ignore[type-var]

    def test_rejects_basemodel_instance(self) -> None:
        with pytest.raises(ConfigurationError, match="BaseModel"):
            # pass an instance instead of a class.
            StructuredOutput(Customer(name="x", email="y"))  # type: ignore[arg-type]


class TestShapeInference:
    def test_shape_inference_derives_schema_from_model(self) -> None:
        out = StructuredOutput(Customer)
        req = InferenceRequest(
            function_name="extract", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.output_schema is not None
        assert isinstance(result.output_schema, dict)
        # Pydantic v2 schema shape.
        assert result.output_schema.get("type") == "object"

    def test_shape_inference_sets_json_mode(self) -> None:
        out = StructuredOutput(Customer, json_mode="strict")
        req = InferenceRequest(
            function_name="extract", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.params == {"chat_completion": {"json_mode": "strict"}}


class TestMaterialize:
    async def test_materialize_validates_pydantic_model(self) -> None:
        out = StructuredOutput(Customer)
        response = _make_response(
            raw='{"name": "Ada", "email": "ada@example.com"}',
            parsed={"name": "Ada", "email": "ada@example.com"},
        )
        result = await out.materialize(response, attempts=[])
        assert isinstance(result.value, Customer)
        assert result.value.name == "Ada"
        assert result.value.email == "ada@example.com"
        assert result.parse_strategy == "strict"

    async def test_materialize_records_validation_error(self) -> None:
        class StrictCustomer(BaseModel):
            name: str
            email: str

        out = StructuredOutput(StrictCustomer)
        # Missing both required fields.
        response = _make_response(raw='{}', parsed={})
        result = await out.materialize(response, attempts=[])
        assert result.value is None
        assert result.parse_strategy == "none"
        assert result.diagnostics is not None
        assert result.diagnostics.failure_detail is not None

    async def test_materialize_permissive_recovery(self) -> None:
        out = StructuredOutput(Customer, permissive=True)
        response = _make_response(
            raw='Here you go: {"name": "Ada", "email": "ada@example.com"}',
            parsed=None,
        )
        result = await out.materialize(response, attempts=[])
        assert isinstance(result.value, Customer)
        assert result.value.name == "Ada"


class TestRecover:
    async def test_recover_uses_validation_detail_when_present(self) -> None:
        out = StructuredOutput(Customer, max_retries=2)
        failure = ParseFailure(
            raw_text='{}',
            inference_id="x",
            reason="validation_failed",
            detail="email: missing; name: missing",
        )
        recovery = await out.recover(
            failure,
            attempts=[Attempt("x", "{}", None, None)],
        )
        assert recovery.kind == "retry"
        assert recovery.hint is not None
        assert "Validation failed" in recovery.hint
        assert "email: missing" in recovery.hint

    async def test_recover_falls_back_to_generic_hint_for_parse_failure(
        self,
    ) -> None:
        out = StructuredOutput(Customer, max_retries=2)
        failure = ParseFailure(
            raw_text="garbage",
            inference_id="x",
            reason="parse_failed",
            detail="",
        )
        recovery = await out.recover(
            failure,
            attempts=[Attempt("x", "garbage", None, None)],
        )
        assert recovery.kind == "retry"
        assert recovery.hint is not None
        # Falls back to the generic JsonOutput hint.
        from t0_agents.core.outputs import JsonOutput

        assert JsonOutput.RETRY_HINT_BASE in recovery.hint

    async def test_recover_fails_past_budget(self) -> None:
        out = StructuredOutput(Customer, max_retries=1)
        failure = ParseFailure(
            raw_text="bad",
            inference_id="x",
            reason="parse_failed",
            detail="",
        )
        attempts = [
            Attempt(inference_id=f"x{i}", raw_text="bad", parsed=None, usage=None)
            for i in range(2)
        ]
        recovery = await out.recover(failure, attempts=attempts)
        assert recovery.kind == "fail"
