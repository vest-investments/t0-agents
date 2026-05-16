"""Tests for :meth:`Run.feedback`.

Verifies that :meth:`Run.feedback` forwards to
:meth:`tensorzero.AsyncTensorZeroGateway.feedback` with the right
correlation kwargs — ``episode_id`` for the default ``target="episode"``
and the latest ``inference_id`` (or an explicit override) for
``target="inference"``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


def _terminal_response(text: str = "done") -> FakeChatResponse:
    return FakeChatResponse(content=[Text(type="text", text=text)])


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    client: FakeMCPClient | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "test", "function_name": "agent"}
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_terminal_response()])),
        tool_provider=cast(
            ToolProvider, client or FakeMCPClient(tools=[]),
        ),
    )


class TestFeedbackToEpisode:
    async def test_default_target_is_episode(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        await run.feedback("thumbs_up", True)
        assert len(fake_t0.feedback_calls) == 1
        call = fake_t0.feedback_calls[0]
        assert call["metric_name"] == "thumbs_up"
        assert call["value"] is True
        assert "episode_id" in call
        assert call["episode_id"] == run.episode_id

    async def test_feedback_before_result_raises(self) -> None:
        agent = _make_agent()
        run = agent.invoke("hi")
        # Cancel so the run never produces an episode_id.
        await run.cancel()
        with pytest.raises(RuntimeError, match="episode_id"):
            await run.feedback("any", 1.0)


class TestFeedbackToInference:
    async def test_target_inference_uses_latest_id(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        result = await run.result()
        await run.feedback("rating", 4.5, target="inference")
        call = fake_t0.feedback_calls[0]
        assert call["metric_name"] == "rating"
        assert call["value"] == 4.5
        assert call["inference_id"] == result.inference_ids[-1]

    async def test_explicit_inference_id_override(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        await run.feedback(
            "rating",
            5.0,
            target="inference",
            inference_id="manual-override-id",
        )
        call = fake_t0.feedback_calls[0]
        assert call["inference_id"] == "manual-override-id"


class TestFeedbackValueTypes:
    """The TensorZero feedback API accepts float / str / bool values."""

    async def test_float_value(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        await run.feedback("score", 0.87)
        assert fake_t0.feedback_calls[0]["value"] == 0.87

    async def test_string_value(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        await run.feedback("comment", "great")
        assert fake_t0.feedback_calls[0]["value"] == "great"

    async def test_bool_value(self) -> None:
        fake_t0 = FakeT0([_terminal_response()])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        await run.feedback("correct", False)
        assert fake_t0.feedback_calls[0]["value"] is False
