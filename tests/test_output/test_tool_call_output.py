"""Unit tests for :class:`~t0_agents.core.outputs.ToolCallOutput`."""

from __future__ import annotations

from uuid import uuid4

import pytest
from tensorzero.types import (
    ChatInferenceResponse,
    Text,
    ToolCall,
    Usage,
)

from t0_agents.core.output import InferenceRequest, ParseFailure
from t0_agents.core.outputs import ToolCallOutput


class TestCapabilityFlags:
    def test_expects_tool_loop_is_false(self) -> None:
        # We WANT the tool calls, not their results.
        assert ToolCallOutput().expects_tool_loop is False

    def test_supports_streaming_is_false(self) -> None:
        assert ToolCallOutput().supports_streaming is False

    def test_name_is_tool_call(self) -> None:
        assert ToolCallOutput().name == "tool_call"


class TestShapeInference:
    def test_shape_inference_sets_tool_choice_required(self) -> None:
        out = ToolCallOutput(tool_choice="required")
        req = InferenceRequest(
            function_name="router", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.tool_choice == "required"

    def test_shape_inference_passes_through_specific_tool_choice(self) -> None:
        out = ToolCallOutput(tool_choice={"specific": "search"})
        req = InferenceRequest(
            function_name="router", input={"messages": []}
        )
        result = out.shape_inference(req)
        assert result.tool_choice == {"specific": "search"}


class TestMaterialize:
    async def test_materialize_extracts_tool_calls(self) -> None:
        out = ToolCallOutput()
        tc = ToolCall(
            id="call_1",
            raw_arguments='{"x": 1}',
            raw_name="add",
            arguments={"x": 1},
            name="add",
            type="tool_call",
        )
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[tc],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert len(result.value) == 1
        assert result.value[0] is tc
        # parse_strategy reflects the routing succeeded.
        assert result.parse_strategy == "strict"

    async def test_materialize_filters_non_toolcall_blocks(self) -> None:
        out = ToolCallOutput()
        tc = ToolCall(
            id="call_2",
            raw_arguments="{}",
            raw_name="lookup",
            arguments={},
            name="lookup",
            type="tool_call",
        )
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[
                Text(text="reasoning", type="text"),
                tc,
            ],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert len(result.value) == 1
        assert result.value[0] is tc
        # The text block alongside the tool call is preserved on
        # diagnostics so debuggers can recover the model's reasoning
        # via ``result.response`` (the v0.12 read-side property).
        assert result.response == "reasoning"

    async def test_materialize_joins_multiple_text_blocks(self) -> None:
        from tensorzero.types import Thought

        out = ToolCallOutput()
        tc = ToolCall(
            id="call_3",
            raw_arguments="{}",
            raw_name="x",
            arguments={},
            name="x",
            type="tool_call",
        )
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[
                Text(text="step 1", type="text"),
                Thought(text="hmm", type="thought"),
                tc,
                Text(text="step 2", type="text"),
            ],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == [tc]
        # Text + Thought blocks joined with newline, in document order.
        assert result.response == "step 1\nhmm\nstep 2"

    async def test_materialize_empty_tool_calls_reports_failure(self) -> None:
        out = ToolCallOutput()
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[Text(text="no tools used", type="text")],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == []
        assert result.parse_strategy == "none"

    async def test_materialize_rejects_wrong_response_type(self) -> None:
        out = ToolCallOutput()
        with pytest.raises(TypeError, match=r"ToolCallOutput\.materialize"):
            await out.materialize("not a response", attempts=[])


class TestRecover:
    async def test_recover_always_fails(self) -> None:
        out = ToolCallOutput()
        failure = ParseFailure(
            raw_text="", inference_id="x", reason="parse_failed", detail="",
        )
        recovery = await out.recover(failure, attempts=[])
        assert recovery.kind == "fail"
