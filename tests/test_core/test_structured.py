"""Unit tests for :func:`run_structured_inference` and :class:`StructuredOutput`.

Uses a hand-rolled ``FakeJsonT0`` (mirroring ``tests/_fake_t0.py``'s
``FakeT0``) that returns real ``JsonInferenceResponse`` dataclasses
built from the SDK's own types.  ``MockGateway`` is chat-only in v0.9.0;
M5 will fold the JSON branch into it and the carry-forward note in the
milestone-1 dev-log captures that hand-off.
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

from t0_agents.core.structured import StructuredOutput, run_structured_inference

# ---------------------------------------------------------------------------
# Test fakes — local to this file until M5 generalises ``MockGateway``.
# ---------------------------------------------------------------------------


def _fake_json_response(
    *,
    raw: str | None = '{"a": 1}',
    parsed: dict[str, Any] | None = None,
    inference_id: UUID | None = None,
    episode_id: UUID | None = None,
) -> JsonInferenceResponse:
    """Build a real ``JsonInferenceResponse`` from SDK dataclasses.

    Using the real type rather than a duck-typed fake means
    ``isinstance(response, JsonInferenceResponse)`` in
    :func:`run_structured_inference` works without a separate runtime
    branch for tests.
    """
    if parsed is None and raw is not None and raw.strip().startswith("{"):
        # Default: mirror the gateway's own parse for the happy path.
        import json as _json

        try:
            parsed_value = _json.loads(raw)
        except _json.JSONDecodeError:
            parsed_value = None
    else:
        parsed_value = parsed
    return JsonInferenceResponse(
        inference_id=inference_id or uuid4(),
        episode_id=episode_id or uuid4(),
        variant_name="test-variant",
        output=JsonInferenceOutput(raw=raw, parsed=parsed_value),
        usage=Usage(input_tokens=10, output_tokens=20),  # type: ignore[call-arg]
    )


class FakeJsonT0:
    """Records every ``inference()`` call; returns scripted JSON responses.

    Mirrors ``tests._fake_t0.FakeT0`` shape but yields
    :class:`JsonInferenceResponse` instances.  The ``calls`` list is the
    test-assertion surface: each entry is the captured kwargs of one
    ``inference()`` call.
    """

    def __init__(self, responses: list[JsonInferenceResponse | Exception]) -> None:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunStructuredInference:
    async def test_returns_parsed_when_tz_parsed_succeeded(self) -> None:
        t0 = FakeJsonT0(
            [_fake_json_response(raw='{"a": 1}', parsed={"a": 1})],
        )
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
        assert result.usage.input_tokens == 10

    async def test_returns_none_when_tz_could_not_parse(self) -> None:
        # M1 trusts TZ's parse only — no permissive recovery yet.
        t0 = FakeJsonT0(
            [_fake_json_response(raw="not json", parsed=None)],
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
        t0 = FakeJsonT0([_fake_json_response()])
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
        t0 = FakeJsonT0([_fake_json_response()])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
        )
        assert t0.calls[0]["params"] is None
        assert t0.calls[0]["output_schema"] is None
        assert "system" not in t0.calls[0]["input"]

    async def test_on_inference_callback_fires_once(self) -> None:
        seen: list[JsonInferenceResponse] = []

        async def cb(resp: JsonInferenceResponse) -> None:
            seen.append(resp)

        canned = _fake_json_response(raw='{"a": 1}', parsed={"a": 1})
        t0 = FakeJsonT0([canned])
        await run_structured_inference(
            cast(AsyncTensorZeroGateway, t0),
            function_name="extract",
            user_message="hello",
            on_inference=cb,
        )
        assert len(seen) == 1
        assert seen[0] is canned

    async def test_raises_when_function_returns_chat_response(self) -> None:
        # Sniff: caller pointed us at a chat function.  The fake returns
        # a non-JsonInferenceResponse value to simulate this.
        from tensorzero import ChatInferenceResponse, Text

        chat_resp = ChatInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="v",
            content=[Text(text="hi", type="text")],
            usage=Usage(input_tokens=1, output_tokens=1),  # type: ignore[call-arg]
        )
        t0 = FakeJsonT0([cast(Any, chat_resp)])
        with pytest.raises(RuntimeError, match="type=\"json\""):
            await run_structured_inference(
                cast(AsyncTensorZeroGateway, t0),
                function_name="chat-fn",
                user_message="hello",
            )
