"""Harness engineering for tool-use reliability (unit tests).

All tests reuse ``FakeT0`` / ``FakeChatResponse`` from ``tests._fake_t0`` and
the ``FakeMCPClient`` / ``_tool_call`` / ``_make_result`` / ``_mcp_tool``
helpers from ``tests.test_mcp.test_loop``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, Thought
from tensorzero.types import TensorZeroError

from t0_agents import run_agent
from t0_agents.mcp.loop import RECOVERABLE_INFERENCE_PATTERNS
from tests._fake_t0 import FakeChatResponse, FakeT0
from tests.test_mcp.test_loop import (
    FakeMCPClient,
    _make_result,
    _mcp_tool,
    _tool_call,
)

# --- TestLoggingMicrogaps -----------------------------------------------------


class TestLoggingMicrogaps:
    """Logging completeness: duration, tool names, content types, thought text."""

    async def test_duration_ms_logged_for_tool_call(
        self, caplog: Any,
    ) -> None:
        """DEBUG record contains 'completed in' with a numeric ms value."""
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
        duration_msgs = [m for m in messages if "completed in" in m]
        assert len(duration_msgs) >= 1, "No 'completed in' log record found"
        # Verify numeric ms value is present (e.g. "completed in 0.1ms")
        assert any("ms" in m for m in duration_msgs)

    async def test_duration_ms_logged_on_error(
        self, caplog: Any,
    ) -> None:
        """duration_ms is also logged when a tool raises an error."""
        from fastmcp.exceptions import ToolError

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

        messages = [r.message for r in caplog.records]
        duration_msgs = [m for m in messages if "completed in" in m]
        assert len(duration_msgs) >= 1

    async def test_tool_names_in_dispatch_log(
        self, caplog: Any,
    ) -> None:
        """DEBUG dispatch record includes tool names."""
        r1 = FakeChatResponse(content=[
            _tool_call(name="add", call_id="c1"),
            _tool_call(name="greet", raw_name="greet", call_id="c2"),
        ])
        r2 = FakeChatResponse(content=[Text(text="done")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(
            tools=[_mcp_tool("add"), _mcp_tool("greet")],
            call_results={
                "add": _make_result(data=5),
                "greet": _make_result(data="hello"),
            },
        )

        with caplog.at_level(logging.DEBUG, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        messages = [r.message for r in caplog.records]
        dispatch_msgs = [m for m in messages if "dispatching" in m]
        assert len(dispatch_msgs) >= 1
        # Both tool names should appear in the dispatch log
        msg = dispatch_msgs[0]
        assert "add" in msg
        assert "greet" in msg

    async def test_content_type_breakdown_logged(
        self, caplog: Any,
    ) -> None:
        """INFO record includes content_types= with block type names."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        with caplog.at_level(logging.INFO, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        messages = [r.message for r in caplog.records]
        ct_msgs = [m for m in messages if "content_types=" in m]
        assert len(ct_msgs) >= 1
        # Turn 1 should show ToolCall
        assert any("ToolCall" in m for m in ct_msgs)
        # Turn 2 should show Text
        assert any("Text" in m for m in ct_msgs)

    async def test_thought_text_logged_at_debug(
        self, caplog: Any,
    ) -> None:
        """Thought.text logged at DEBUG on a Thought-only terminal turn."""
        r1 = FakeChatResponse(content=[
            Thought(text="I'm thinking about the answer..."),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        with caplog.at_level(logging.DEBUG, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        messages = [r.message for r in caplog.records]
        thought_msgs = [m for m in messages if "Thought content:" in m]
        assert len(thought_msgs) >= 1
        assert "thinking about the answer" in thought_msgs[0]


# --- TestThoughtOnlyHandling -------------------------------------------------


class TestThoughtOnlyHandling:
    """Thought-only responses produce response_source='thought'."""

    async def test_thought_only_response_extracted(self) -> None:
        """Thought-only response returns response_source='thought'."""
        r1 = FakeChatResponse(content=[
            Thought(text="The answer is 5"),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert result.response == "The answer is 5"
        assert result.response_source == "thought"
        assert result.turns == 1

    async def test_thought_none_text_returns_empty(self) -> None:
        """Thought(text=None) results in empty response, source='text'."""
        r1 = FakeChatResponse(content=[
            Thought(text=None),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert result.response == ""
        assert result.response_source == "text"

    async def test_text_takes_priority_over_thought(self) -> None:
        """When both Text and Thought are present, Text wins."""
        r1 = FakeChatResponse(content=[
            Thought(text="internal reasoning"),
            Text(text="The sum is 5"),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert result.response == "The sum is 5"
        assert result.response_source == "text"

    async def test_multiple_thoughts_joined(self) -> None:
        """Multiple Thought blocks are joined with newlines."""
        r1 = FakeChatResponse(content=[
            Thought(text="Step 1: add 2+3"),
            Thought(text="Step 2: the answer is 5"),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert "Step 1: add 2+3" in result.response
        assert "Step 2: the answer is 5" in result.response
        assert result.response_source == "thought"

    async def test_thought_only_logs_fallback(self, caplog: Any) -> None:
        """INFO log emitted when thought-only fallback is used."""
        r1 = FakeChatResponse(content=[
            Thought(text="reasoning"),
        ])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        with caplog.at_level(logging.INFO, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        messages = [r.message for r in caplog.records]
        assert any("thought-only response used as fallback" in m for m in messages)


# --- TestToolChoice -----------------------------------------------------------


class TestToolChoice:
    """tool_choice parameter forwarded to t0.inference()."""

    async def test_tool_choice_forwarded_every_turn(self) -> None:
        """tool_choice is passed to every t0.inference() call."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "compute 2+3",
            "agent",
            tool_choice={"specific": "add"},
        )

        assert len(t0.calls) == 2
        assert t0.calls[0]["tool_choice"] == {"specific": "add"}
        assert t0.calls[1]["tool_choice"] == {"specific": "add"}

    async def test_tool_choice_none_by_default(self) -> None:
        """When tool_choice is not specified, None is passed."""
        r1 = FakeChatResponse(content=[Text(text="hi")])
        t0 = FakeT0([r1])
        mcp = FakeMCPClient()

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "hello",
            "agent",
        )

        assert t0.calls[0]["tool_choice"] is None

    async def test_tool_choice_literal_values(self) -> None:
        """Literal tool_choice values ('auto', 'required') are forwarded."""
        r1 = FakeChatResponse(content=[_tool_call(name="add")])
        r2 = FakeChatResponse(content=[Text(text="5")])
        t0 = FakeT0([r1, r2])
        mcp = FakeMCPClient(call_results={"add": _make_result(data=5)})

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
            tool_choice="required",
        )

        assert t0.calls[0]["tool_choice"] == "required"
        assert t0.calls[1]["tool_choice"] == "required"


# --- TestRecoverableInferenceError --------------------------------------------


class TestRecoverableInferenceError:
    """Recoverable TensorZeroError caught in-loop with retry."""

    async def test_malformed_tool_name_caught_and_retried(self) -> None:
        """TensorZeroError with 'not in request.tools' is caught;
        loop injects corrective message and continues to next turn."""
        error = TensorZeroError(
            status_code=400,
            text=(
                'tool call validation failed: "add<|channel|>commentary" '
                "not in request.tools"
            ),
        )
        r2 = FakeChatResponse(content=[Text(text="The answer is 5")])
        t0 = FakeT0([error, r2])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "compute 2+3",
            "agent",
        )

        assert result.response == "The answer is 5"
        assert result.turns == 2
        # The second inference call should contain the corrective message.
        # Message is a TypedDict (dict), so use dict access.
        second_call_messages = t0.calls[1]["input"]["messages"]
        corrective_msgs = [
            m for m in second_call_messages
            if isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and "invalid tool call" in m["content"]
        ]
        assert len(corrective_msgs) >= 1

    async def test_corrective_message_lists_available_tools(self) -> None:
        """The corrective message includes available tool names."""
        error = TensorZeroError(
            status_code=400,
            text='validation failed: "bad_name" not in request.tools',
        )
        r2 = FakeChatResponse(content=[Text(text="ok")])
        t0 = FakeT0([error, r2])
        mcp = FakeMCPClient(tools=[_mcp_tool("add"), _mcp_tool("greet")])

        await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        # Find the corrective user message (Message is a TypedDict/dict).
        second_call_messages = t0.calls[1]["input"]["messages"]
        corrective = [
            m for m in second_call_messages
            if isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and "Available tools" in m["content"]
        ]
        assert len(corrective) >= 1
        assert "add" in corrective[-1]["content"]
        assert "greet" in corrective[-1]["content"]

    async def test_non_matching_error_re_raised(self) -> None:
        """TensorZeroError that doesn't match patterns is re-raised."""
        error = TensorZeroError(status_code=500, text="internal server error")
        t0 = FakeT0([error])
        mcp = FakeMCPClient()

        with pytest.raises(TensorZeroError) as exc_info:
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        assert exc_info.value.status_code == 500

    async def test_503_with_matching_text_re_raised(self) -> None:
        """503 Service Unavailable is NOT recoverable even with matching text."""
        error = TensorZeroError(
            status_code=503, text="not in request.tools",
        )
        t0 = FakeT0([error])
        mcp = FakeMCPClient()

        with pytest.raises(TensorZeroError) as exc_info:
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        assert exc_info.value.status_code == 503

    async def test_502_bad_gateway_recovered(self) -> None:
        """502 Bad Gateway with matching text IS recoverable.

        TensorZero v2026.4.0 wraps provider-level 400 tool-validation
        failures as gateway-level 502s.  The loop must treat them the
        same as direct 400s to preserve the ``<|channel|>``-artifact
        recovery path.
        """
        error = TensorZeroError(
            status_code=502,
            text=(
                'All variants failed with errors: gpt_oss_20: All model providers '
                'failed to infer with errors: groq: Error 400 Bad Request from groq '
                'client: tool call validation failed: attempted to call tool '
                "'add<|channel|>commentary' which was not in request.tools"
            ),
        )
        r2 = FakeChatResponse(content=[Text(text="done")])
        t0 = FakeT0([error, r2])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert result.response == "done"
        assert result.turns == 2

    async def test_error_logged_as_warning(self, caplog: Any) -> None:
        """Caught TensorZeroError is logged at WARNING level."""
        error = TensorZeroError(
            status_code=400,
            text='tool call validation failed: "x" not in request.tools',
        )
        r2 = FakeChatResponse(content=[Text(text="ok")])
        t0 = FakeT0([error, r2])
        mcp = FakeMCPClient()

        with caplog.at_level(logging.WARNING, logger="t0_agents.mcp.loop"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], mcp),
                "test",
                "agent",
            )

        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("recoverable TensorZeroError" in m for m in warning_msgs)

    async def test_sentinel_inference_id_recorded(self) -> None:
        """Error turn records a sentinel inference_id."""
        error = TensorZeroError(
            status_code=400,
            text='"bad" not in request.tools',
        )
        r2 = FakeChatResponse(content=[Text(text="ok")])
        t0 = FakeT0([error, r2])
        mcp = FakeMCPClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            cast(Client[Any], mcp),
            "test",
            "agent",
        )

        assert len(result.inference_ids) == 2
        assert result.inference_ids[0].startswith("error-recoverable-turn-")

    async def test_recoverable_patterns_extensible(self) -> None:
        """RECOVERABLE_INFERENCE_PATTERNS is a module-level list
        that can be extended with new patterns."""
        import re

        assert isinstance(RECOVERABLE_INFERENCE_PATTERNS, list)
        assert len(RECOVERABLE_INFERENCE_PATTERNS) >= 1
        assert all(
            isinstance(p, re.Pattern) for p in RECOVERABLE_INFERENCE_PATTERNS
        )


# --- TestDryrunDocstring ------------------------------------------------------


class TestDryrunDocstring:
    """run_agent docstring documents dryrun + feedback constraint."""

    def test_dryrun_feedback_constraint_documented(self) -> None:
        """run_agent docstring mentions dryrun and feedback constraint."""
        doc = run_agent.__doc__
        assert doc is not None
        assert "dryrun" in doc
        assert "feedback" in doc
        assert "ClickHouse" in doc
