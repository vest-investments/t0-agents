"""Unit tests for :func:`run_structured_inference` and :class:`StructuredOutput`.

Drives :func:`run_structured_inference` through ``MockGateway.script_json``
— the JSON-flavoured sibling of ``MockGateway.script`` that pops one
``JsonInferenceResponse`` per call.  See
``tests/test_testing/test_json_helpers.py`` for the script_json semantic
(auto-derive ``parsed`` from ``raw`` when the key is absent; honour
explicit ``parsed`` when present).
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from tensorzero import (
    AsyncTensorZeroGateway,
    JsonInferenceResponse,
    Usage,
)
from tensorzero.types import JsonInferenceOutput

from t0_agents.core.structured import StructuredOutput, run_structured_inference
from t0_agents.testing import MockGateway


class TestRunStructuredInference:
    async def test_returns_parsed_when_tz_parsed_succeeded(self) -> None:
        # ``parsed`` key omitted → auto-derived from ``raw``.
        t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
        result = await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
        )
        assert isinstance(result, StructuredOutput)
        assert result.parsed == {"a": 1}
        assert result.raw == '{"a": 1}'
        assert result.parse_strategy == "strict"
        assert result.retry_attempts == 0
        assert result.usage is not None

    async def test_returns_none_when_tz_could_not_parse(self) -> None:
        # The helper trusts TZ's parse only; the permissive recovery layer
        # is layered on top by ``Agent._run_json``.  Explicit ``parsed:
        # None`` because ``raw`` ("not json") doesn't parse anyway, but
        # spelling it out documents the intent.
        t0 = MockGateway.script_json(
            [{"raw": "not json", "parsed": None}]
        )
        result = await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
        )
        assert result.parsed is None
        assert result.raw == "not json"
        assert result.parse_strategy == "none"

    async def test_threads_every_kwarg_into_inference_call(self) -> None:
        t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
            output_schema={"type": "object", "required": ["a"]},
            json_mode="strict",
            system_arguments={"who": "world"},
            episode_id="ep-123",
            tags={"trace": "abc"},
            namespace="ns",
            dryrun=True,
        )
        assert len(t0.calls) == 1
        kwargs = t0.calls[0]
        assert kwargs["function_name"] == "extract"
        assert kwargs["episode_id"] == "ep-123"
        assert kwargs["tags"] == {"trace": "abc"}
        assert kwargs["namespace"] == "ns"
        assert kwargs["dryrun"] is True
        assert kwargs["output_schema"] == {
            "type": "object",
            "required": ["a"],
        }
        assert kwargs["params"] == {"chat_completion": {"json_mode": "strict"}}
        assert kwargs["input"] == {
            "messages": [{"role": "user", "content": "hello"}],
            "system": {"who": "world"},
        }

    async def test_omits_params_when_json_mode_is_none(self) -> None:
        # Caller leaves json_mode at None → no params dict so TZ uses
        # the variant's configured json_mode.
        t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
        )
        assert t0.calls[0]["params"] is None
        assert t0.calls[0]["output_schema"] is None
        assert "system" not in t0.calls[0]["input"]

    async def test_system_string_forwarded_as_input_system(self) -> None:
        """``system="..."`` reaches the gateway as a raw string."""
        t0 = MockGateway.script_json([{"raw": '{"a": 1}'}])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
            system="Extract structured data from prose.",
        )
        assert t0.calls[0]["input"]["system"] == (
            "Extract structured data from prose."
        )

    async def test_system_and_system_arguments_mutually_exclusive(self) -> None:
        """Mixing both kwargs raises ``ValueError`` before the inference fires."""
        t0 = MockGateway.script_json([])  # never reached
        with pytest.raises(ValueError, match="Pass either `system`"):
            await run_structured_inference(
                cast(AsyncTensorZeroGateway, t0),
                function_name="extract",
                user_message="hello",
                system="raw",
                system_arguments={"who": "world"},
            )
        assert t0.calls == []

    async def test_on_inference_callback_fires_once(self) -> None:
        seen: list[JsonInferenceResponse] = []

        async def cb(resp: JsonInferenceResponse) -> None:
            seen.append(resp)

        # Pre-built response so we can pin identity for the assertion.
        canned = JsonInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="test-variant",
            output=JsonInferenceOutput(raw='{"a": 1}', parsed={"a": 1}),
            usage=Usage(input_tokens=10, output_tokens=20),  # type: ignore[call-arg]
        )
        t0 = MockGateway.script_json([canned])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
            on_inference=cb,
        )
        assert len(seen) == 1
        assert seen[0] is canned

    async def test_raises_when_function_returns_chat_response(self) -> None:
        # Sniff: caller pointed us at a chat function.  Drop down to the
        # raw ``MockGateway`` constructor — ``script_json`` is typed for
        # JSON responses only, so a chat response needs the wider source
        # surface.
        from tensorzero import ChatInferenceResponse, Text

        chat_resp = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[Text(text="hi", type="text")],
            usage=Usage(input_tokens=1, output_tokens=1),  # type: ignore[call-arg]
        )
        gateway = MockGateway(lambda **_: cast(Any, chat_resp))
        with pytest.raises(RuntimeError, match="type=\"json\""):
            await run_structured_inference(
                cast(AsyncTensorZeroGateway, gateway),
                function_name="chat-fn",
                user_message="hello",
            )
