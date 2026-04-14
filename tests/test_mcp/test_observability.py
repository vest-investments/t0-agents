"""Observability and feedback (unit tests).

Tests validate that ``inference_id`` and ``episode_id`` are captured on every
turn, that feedback calls (boolean, float, comment, demonstration) pass the
correct parameters through to ``t0.feedback()``, and that the agent loop emits
structured log records for tool calls, results, errors, and timestamps.

All tests reuse ``FakeT0`` / ``FakeChatResponse`` / ``FakeFeedbackResponse``
from ``tests._fake_t0`` and the ``FakeMCPClient`` / ``_tool_call`` /
``_make_result`` / ``_mcp_tool`` helpers from ``tests.test_mcp.test_loop``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast
from uuid import UUID

from fastmcp import Client
from fastmcp.exceptions import ToolError
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import run_agent
from tests._fake_t0 import FakeChatResponse, FakeFeedbackResponse, FakeT0
from tests.test_mcp.test_loop import (
    FakeMCPClient,
    _make_result,
    _mcp_tool,
    _tool_call,
)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I,
)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


# --- TestObservabilityIds ---------------------------------------------------


class TestObservabilityIds:
    """inference_id and episode_id captured from each T0 response."""

    async def test_ids_captured_on_every_turn(self) -> None:
        """3-turn flow captures episode_id and one inference_id per turn."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(
            content=[_tool_call(name="add", call_id="call-2")],
            episode_id=r1.episode_id,
        )
        r3 = FakeChatResponse(
            content=[Text(text="done")],
            episode_id=r1.episode_id,
        )
        t0 = FakeT0([r1, r2, r3])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert result.episode_id is not None
        assert _is_uuid(result.episode_id), f"Not a UUID: {result.episode_id}"
        assert result.turns == 3
        assert len(result.inference_ids) == 3
        for iid in result.inference_ids:
            assert _is_uuid(iid), f"inference_id not a UUID: {iid}"
        # All inference IDs should be distinct (each turn gets its own)
        assert len(set(result.inference_ids)) == 3


# --- TestFeedback -----------------------------------------------------------


class TestFeedback:
    """Feedback calls pass correct parameters to t0.feedback()."""

    async def _run_and_get_result(self) -> tuple[FakeT0, Any]:
        """Run a minimal 1-turn agent loop and return (FakeT0, AgentResult)."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})
        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "compute 2+3",
            "agent",
        )
        return t0, result

    async def test_boolean_feedback_on_episode(self) -> None:
        """Boolean feedback sent to episode_id with task_success metric."""
        t0, result = await self._run_and_get_result()

        resp = await t0.feedback(
            metric_name="task_success",
            episode_id=result.episode_id,
            value=True,
        )

        assert isinstance(resp, FakeFeedbackResponse)
        assert isinstance(resp.feedback_id, UUID)
        assert len(t0.feedback_calls) == 1
        call = t0.feedback_calls[0]
        assert call["metric_name"] == "task_success"
        assert call["episode_id"] == result.episode_id
        assert call["value"] is True

    async def test_boolean_feedback_false_value(self) -> None:
        """Boolean feedback also works with value=False."""
        t0, result = await self._run_and_get_result()

        await t0.feedback(
            metric_name="task_success",
            episode_id=result.episode_id,
            value=False,
        )

        assert t0.feedback_calls[0]["value"] is False

    async def test_float_feedback_on_inference(self) -> None:
        """Float feedback sent to inference_id with quality_score metric."""
        t0, result = await self._run_and_get_result()

        resp = await t0.feedback(
            metric_name="quality_score",
            inference_id=result.inference_ids[0],
            value=8.5,
        )

        assert isinstance(resp.feedback_id, UUID)
        call = t0.feedback_calls[0]
        assert call["metric_name"] == "quality_score"
        assert call["inference_id"] == result.inference_ids[0]
        assert call["value"] == 8.5

    async def test_comment_feedback(self) -> None:
        """Comment feedback (built-in metric, no config needed)."""
        t0, result = await self._run_and_get_result()

        await t0.feedback(
            metric_name="comment",
            inference_id=result.inference_ids[-1],
            value="Good tool selection.",
        )

        call = t0.feedback_calls[0]
        assert call["metric_name"] == "comment"
        assert call["inference_id"] == result.inference_ids[-1]
        assert call["value"] == "Good tool selection."

    async def test_demonstration_feedback(self) -> None:
        """Demonstration feedback (inference_id only, built-in metric)."""
        t0, result = await self._run_and_get_result()

        await t0.feedback(
            metric_name="demonstration",
            inference_id=result.inference_ids[-1],
            value="The sum of 2 and 3 is 5.",
        )

        call = t0.feedback_calls[0]
        assert call["metric_name"] == "demonstration"
        assert call["inference_id"] == result.inference_ids[-1]
        assert call["value"] == "The sum of 2 and 3 is 5."


# --- TestLogging ------------------------------------------------------------


class TestLogging:
    """Agent loop log shows tool calls, results, errors, timestamps."""

    async def test_logging_shows_tool_calls_and_results(
        self, caplog: Any,
    ) -> None:
        """DEBUG records contain tool name and result for a 2-turn flow."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        with caplog.at_level(logging.DEBUG, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "compute 2+3",
                "agent",
            )

        messages = [r.message for r in caplog.records]
        # Tool execution logged
        assert any("Executing tool 'add'" in m for m in messages)
        # Tool result logged
        assert any("Tool 'add' returned:" in m for m in messages)

    async def test_logging_shows_errors(self, caplog: Any) -> None:
        """ERROR records appear when a tool raises ToolError."""
        r1 = FakeChatResponse(content=[_tool_call(name="fail")])
        r2 = FakeChatResponse(content=[Text(text="sorry")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(
            tools=[_mcp_tool("fail")],
            call_results={"fail": ToolError("boom")},
        )

        with caplog.at_level(logging.DEBUG, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "try fail",
                "agent",
            )

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1
        assert any("fail" in r.message and "ToolError" in r.message for r in error_records)

    async def test_logging_shows_turn_info(self, caplog: Any) -> None:
        """INFO records mention turn number and inference_id."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        with caplog.at_level(logging.INFO, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "compute 2+3",
                "agent",
            )

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        # Turn info logged
        assert any("Turn 1:" in m and "inference_id=" in m for m in info_messages)
        # Loop completion logged
        assert any("Agent loop completed" in m for m in info_messages)
        # Loop start logged
        assert any("Agent loop started" in m for m in info_messages)
        # All records have a timestamp (LogRecord.created is always set)
        for record in caplog.records:
            assert record.created > 0
