"""Unit tests for ``MockGateway`` in its three construction modes."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from tensorzero import Text, ToolCall
from tensorzero.types import ChatInferenceResponse, Usage

from t0_agents.testing import MockChatResponse, MockGateway


class TestScriptMode:
    async def test_returns_turns_in_order(self) -> None:
        gateway = MockGateway.script(
            [
                [Text(text="first")],
                [Text(text="second")],
            ]
        )

        first = await gateway.inference(function_name="f")
        second = await gateway.inference(function_name="f")

        assert isinstance(first.content[0], Text)
        assert first.content[0].text == "first"
        assert isinstance(second.content[0], Text)
        assert second.content[0].text == "second"

    async def test_exhausted_script_yields_empty_text(self) -> None:
        """When the script is exhausted, an empty Text turn is returned.

        Guarantees a test that over-runs doesn't hang — the agent loop
        terminates cleanly on an empty text-only response.
        """
        gateway = MockGateway.script([])
        response = await gateway.inference(function_name="f")

        assert isinstance(response.content[0], Text)
        assert response.content[0].text == ""

    async def test_stable_episode_id_across_turns(self) -> None:
        """Consecutive turns share the same episode_id — matches real T0."""
        gateway = MockGateway.script(
            [
                [Text(text="first")],
                [Text(text="second")],
            ]
        )

        first = await gateway.inference(function_name="f")
        second = await gateway.inference(function_name="f")

        assert first.episode_id == second.episode_id
        assert first.inference_id != second.inference_id

    async def test_calls_records_kwargs(self) -> None:
        gateway = MockGateway.script([[Text(text="ok")]])

        await gateway.inference(
            function_name="assistant",
            episode_id=None,
            input={"messages": []},
            additional_tools=[],
            dryrun=True,
            tool_choice="auto",
            tags={"k": "v"},
            namespace=None,
        )

        assert len(gateway.calls) == 1
        call = gateway.calls[0]
        assert call["function_name"] == "assistant"
        assert call["dryrun"] is True
        assert call["tool_choice"] == "auto"
        assert call["tags"] == {"k": "v"}

    async def test_responses_recorded_in_parallel(self) -> None:
        """responses[i] matches the response returned for calls[i]."""
        gateway = MockGateway.script([[Text(text="one")], [Text(text="two")]])

        r1 = await gateway.inference(function_name="f")
        r2 = await gateway.inference(function_name="f")

        assert gateway.responses == [r1, r2]


class TestCannedMode:
    async def test_callable_receives_kwargs(self) -> None:
        captured: dict[str, Any] = {}

        def source(**kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return [Text(text="got it")]

        gateway = MockGateway.canned(source)

        await gateway.inference(
            function_name="probe",
            episode_id=uuid4(),
            tags={"k": "v"},
        )

        assert captured["function_name"] == "probe"
        assert captured["tags"] == {"k": "v"}

    async def test_callable_can_return_full_response(self) -> None:
        """Source can bypass the wrapping and return a built response."""

        def source(**_: Any) -> ChatInferenceResponse:
            return MockChatResponse(
                [Text(text="full")],
                usage=Usage(input_tokens=10, output_tokens=20),
            ).to_response()

        gateway = MockGateway.canned(source)
        response = await gateway.inference(function_name="f")

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20


class TestRecordingMode:
    async def test_records_and_replays(self) -> None:
        """Recording forwards to real gateway; replay yields captured responses."""

        real = MockGateway.script(
            [
                [ToolCall(id="c1", raw_name="add", raw_arguments="{}", name="add", arguments={})],
                [Text(text="done")],
            ]
        )
        recorder = MockGateway.recording(real)

        await recorder.inference(function_name="f")
        await recorder.inference(function_name="f")

        assert len(recorder.captured) == 2

        replayed = recorder.replay()
        r1 = await replayed.inference(function_name="f")
        r2 = await replayed.inference(function_name="f")

        assert isinstance(r1.content[0], ToolCall)
        assert isinstance(r2.content[0], Text)
        assert r2.content[0].text == "done"

    async def test_replay_after_exhaustion_yields_empty_text(self) -> None:
        real = MockGateway.script([[Text(text="only")]])
        recorder = MockGateway.recording(real)
        await recorder.inference(function_name="f")

        replayed = recorder.replay()
        await replayed.inference(function_name="f")  # first: exhausts the capture
        overflow = await replayed.inference(function_name="f")

        assert isinstance(overflow.content[0], Text)
        assert overflow.content[0].text == ""


class TestFeedback:
    async def test_feedback_returns_ephemeral_id(self) -> None:
        gateway = MockGateway.script([])

        result = await gateway.feedback(
            metric_name="approved", inference_id=uuid4(), value=True
        )

        assert result.feedback_id is not None
        assert len(gateway.feedback_calls) == 1
        assert gateway.feedback_calls[0]["metric_name"] == "approved"


@pytest.mark.parametrize("mode", ["script", "canned"])
async def test_multiple_sequential_calls_increment_responses(mode: str) -> None:
    if mode == "script":
        gateway = MockGateway.script(
            [[Text(text=f"turn {i}")] for i in range(3)]
        )
    else:
        counter = {"n": 0}

        def source(**_: Any) -> list[Any]:
            counter["n"] += 1
            return [Text(text=f"turn {counter['n']}")]

        gateway = MockGateway.canned(source)

    for _ in range(3):
        await gateway.inference(function_name="f")

    assert len(gateway.calls) == 3
    assert len(gateway.responses) == 3
