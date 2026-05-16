"""Unit tests for ``t0_agents.compose.delegation.agent_as_tool``."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.tools import Tool
from tensorzero import AsyncTensorZeroGateway

from t0_agents.compose.delegation import agent_as_tool
from t0_agents.compose.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    bind_parent_episode_id,
)
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    GuardrailBlockedError,
    SubAgentError,
)
from t0_agents.core.types import AgentResult, ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0


def _build_sub_agent(name: str = "researcher") -> Agent:
    config = AgentConfig(name=name, function_name=f"{name}_fn")
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


def _stub_result(text: str = "ok") -> AgentResult:
    return AgentResult(
        response=text,
        episode_id="ep-sub",
        inference_ids=["i-1"],
        turns=1,
    )


def _mock_invoke_returning(result: AgentResult) -> MagicMock:
    """Build a sync mock for ``Agent.invoke`` that returns a Run-shaped mock.

    The delegation handler calls ``sub_agent.invoke(message, tags=...).result()``.
    The mock returned here exposes that exact shape: a sync callable
    whose return value has an awaitable ``.result()`` resolving to the
    supplied :class:`AgentResult`.
    """
    mock_run = MagicMock()
    mock_run.result = AsyncMock(return_value=result)
    return MagicMock(return_value=mock_run)


def _mock_invoke_raising(exc: BaseException) -> MagicMock:
    """Build an invoke mock whose Run.result() raises ``exc``."""
    mock_run = MagicMock()
    mock_run.result = AsyncMock(side_effect=exc)
    return MagicMock(return_value=mock_run)


class TestAgentAsTool:
    def test_returns_tool_with_default_name(self) -> None:
        sub = _build_sub_agent("researcher")

        tool = agent_as_tool(sub)

        assert isinstance(tool, Tool)
        assert tool.name == "researcher_ask"
        # Description is the default template; must mention the sub-agent's
        # name so the orchestrator LLM can disambiguate between multiple
        # delegation tools.
        assert tool.description is not None
        assert "researcher" in tool.description

    def test_custom_name_and_description(self) -> None:
        sub = _build_sub_agent()
        tool = agent_as_tool(
            sub,
            tool_name="ask_researcher",
            description="Ask the researcher for a paper summary",
        )

        assert tool.name == "ask_researcher"
        assert tool.description == "Ask the researcher for a paper summary"

    async def test_handler_invokes_sub_agent_and_forwards_parent_episode(
        self,
    ) -> None:
        sub = _build_sub_agent()
        # Replace .invoke so we don't exercise the real loop — we're
        # asserting the handler contract, not the sub-agent's semantics.
        # v1.0: delegation dispatches via Agent.invoke().result();
        # the mock exposes that shape.
        sub.invoke = _mock_invoke_returning(_stub_result("hello"))  # type: ignore[method-assign]

        tool = agent_as_tool(sub)
        handler = tool.fn  # type: ignore[attr-defined]

        with bind_parent_episode_id("ep-parent"):
            out = await handler(message="hi")

        assert out == "hello"
        sub.invoke.assert_called_once()
        call_args = sub.invoke.call_args
        assert call_args.kwargs["tags"] == {PARENT_EPISODE_ID_TAG: "ep-parent"}
        assert call_args.args == ("hi",)

    async def test_handler_falls_back_to_unknown_when_no_parent_bound(
        self,
    ) -> None:
        sub = _build_sub_agent()
        sub.invoke = _mock_invoke_returning(_stub_result())  # type: ignore[method-assign]

        tool = agent_as_tool(sub)
        handler = tool.fn  # type: ignore[attr-defined]

        out = await handler(message="hi")

        assert out == "ok"
        call_kwargs = sub.invoke.call_args.kwargs
        assert call_kwargs["tags"] == {PARENT_EPISODE_ID_TAG: "unknown"}

    async def test_handler_wraps_non_fatal_exception(self) -> None:
        sub = _build_sub_agent("researcher")
        sub.invoke = _mock_invoke_raising(ValueError("boom"))  # type: ignore[method-assign]

        tool = agent_as_tool(sub)
        handler = tool.fn  # type: ignore[attr-defined]

        with pytest.raises(SubAgentError) as exc_info:
            await handler(message="hi")

        # Message should carry the sub-agent name so orchestrator logs
        # aren't anonymous.
        assert "researcher" in str(exc_info.value)
        # Cause chain preserved — debugging relies on the original error.
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "boom"

    async def test_handler_reraises_fatal_unchanged(self) -> None:
        sub = _build_sub_agent("researcher")
        sub.invoke = _mock_invoke_raising(  # type: ignore[method-assign]
            GuardrailBlockedError("hard stop"),
        )

        tool = agent_as_tool(sub)
        handler = tool.fn  # type: ignore[attr-defined]

        # Fatal errors (subclasses of T0AgentsFatalError) must propagate
        # unchanged so orchestrator callers see the real type.
        with pytest.raises(GuardrailBlockedError):
            await handler(message="hi")

    def test_schema_shape_is_single_string_argument(self) -> None:
        sub = _build_sub_agent()
        tool = agent_as_tool(sub)

        # FunctionTool derives the schema from the handler signature.  A
        # single ``message: str`` parameter yields a flat object schema —
        # T0's MCP adapter keeps it as-is for the LLM.
        schema: dict[str, Any] = tool.parameters  # type: ignore[assignment]
        assert schema["type"] == "object"
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["required"] == ["message"]
