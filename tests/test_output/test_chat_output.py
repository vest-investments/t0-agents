"""Unit tests for :class:`~t0_agents.core.outputs.ChatOutput`."""

from __future__ import annotations

from uuid import uuid4

import pytest
from tensorzero.types import (
    ChatInferenceResponse,
    Text,
    Thought,
    Usage,
)

from t0_agents.core.output import InferenceRequest, ParseFailure
from t0_agents.core.outputs import ChatOutput
from t0_agents.core.types import AgentResult


class TestCapabilityFlags:
    def test_expects_tool_loop_is_true(self) -> None:
        assert ChatOutput().expects_tool_loop is True

    def test_supports_streaming_is_true(self) -> None:
        assert ChatOutput().supports_streaming is True

    def test_name_is_chat(self) -> None:
        assert ChatOutput().name == "chat"


class TestShapeInference:
    def test_shape_inference_is_noop(self) -> None:
        out = ChatOutput()
        req = InferenceRequest(
            function_name="agent",
            input={"messages": []},
        )
        result = out.shape_inference(req)
        assert result is req
        assert result.params is None
        assert result.output_schema is None
        assert result.tool_choice is None


class TestMaterialize:
    async def test_materialize_joins_text_blocks(self) -> None:
        out = ChatOutput()
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[Text(text="hello", type="text")],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert isinstance(result, AgentResult)
        assert result.value == "hello"
        assert result.response == "hello"
        assert result.response_source == "text"
        assert result.function_kind == "chat"
        assert len(result.inference_ids) == 1

    async def test_materialize_falls_back_to_thought(self) -> None:
        out = ChatOutput()
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[Thought(text="thinking aloud", type="thought")],
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == "thinking aloud"
        assert result.response_source == "thought"

    async def test_materialize_handles_empty_content(self) -> None:
        out = ChatOutput()
        response = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[],
            usage=Usage(input_tokens=0, output_tokens=0),
        )
        result = await out.materialize(response, attempts=[])
        assert result.value == ""

    async def test_materialize_rejects_wrong_response_type(self) -> None:
        out = ChatOutput()
        with pytest.raises(TypeError, match=r"ChatOutput\.materialize"):
            await out.materialize("not a response", attempts=[])


class TestRecover:
    async def test_recover_always_returns_fail(self) -> None:
        out = ChatOutput()
        failure = ParseFailure(
            raw_text="", inference_id="x", reason="parse_failed", detail="",
        )
        recovery = await out.recover(failure, attempts=[])
        assert recovery.kind == "fail"
