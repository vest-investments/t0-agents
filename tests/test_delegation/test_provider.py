"""Unit tests for ``t0_agents.delegation.provider.SubAgentToolProvider``.

The provider replaces the legacy dual-vía (server-side
``install_sub_agents`` + middleware-side ``_OrchestrationToolProvider``)
with a single class that synthesises ``mcp.types.Tool`` objects from
the sub-agent list and dispatches matching ``call_tool`` invocations
inline through :meth:`Agent.invoke`.  These tests cover every branch
of that contract.

The Agent-level integration (``Agent(config_with_subagents,
tool_provider=None)`` constructs zero FastMCP instances) lands together
with the auto-allocation commit; see ``test_orchestrator.py`` /
``test_agent.py`` for that assertion.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
    SubAgentError,
)
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.delegation.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    bind_parent_episode_id,
)
from t0_agents.delegation.provider import SubAgentToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0


def _build_sub_agent(name: str = "researcher") -> Agent:
    config = AgentConfig(name=name, function_name=f"{name}_fn")
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


def _spy_sub_agent(name: str = "researcher") -> Agent:
    """Sub-agent with ``invoke`` replaced by an :class:`AgentResult`-returning spy."""
    sub = _build_sub_agent(name)
    mock_run = MagicMock()
    mock_run.result = AsyncMock(
        return_value=AgentResult(
            response="sub-answer",
            episode_id="ep-sub",
            inference_ids=["i-sub"],
            turns=1,
        ),
    )
    # provider.call_tool now reads child_run.usage() to attach billing to the
    # result's meta; the spy has no real Run, so report "no usage" (meta=None).
    mock_run.usage = MagicMock(return_value=None)
    sub.invoke = MagicMock(return_value=mock_run)  # type: ignore[method-assign]
    return sub


def _ok_call_result(text: str = "from-inner") -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        meta=None,
        data=None,
        is_error=False,
    )


class _StubInnerProvider:
    """Minimal ToolProvider that records call_tool invocations."""

    def __init__(
        self,
        *,
        tools: list[MCPTool] | None = None,
        result: CallToolResult | None = None,
    ) -> None:
        self._tools = tools if tools is not None else []
        self._result = result if result is not None else _ok_call_result()
        self.list_calls = 0
        self.tool_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def list_tools(self) -> list[Any]:
        self.list_calls += 1
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **_: Any,
    ) -> CallToolResult:
        self.tool_calls.append((name, arguments))
        return self._result


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_sub_agents_list_is_valid(self) -> None:
        # Degenerate but constructible — useful when callers build the
        # provider eagerly from a (possibly empty) source list.
        provider = SubAgentToolProvider([])
        assert provider._by_tool_name == {}

    def test_duplicate_sub_agent_names_raise(self) -> None:
        a = _build_sub_agent("researcher")
        b = _build_sub_agent("researcher")
        with pytest.raises(ConfigurationError, match="Duplicate sub-agent names"):
            SubAgentToolProvider([a, b])

    def test_accepts_inner_provider(self) -> None:
        sub = _build_sub_agent("researcher")
        inner = _StubInnerProvider()
        provider = SubAgentToolProvider([sub], inner=cast(ToolProvider, inner))
        assert provider._inner is inner


# ---------------------------------------------------------------------------
# list_tools — synthesis
# ---------------------------------------------------------------------------


class TestListTools:
    async def test_synthesises_one_tool_per_sub_agent(self) -> None:
        researcher = _build_sub_agent("researcher")
        writer = _build_sub_agent("writer")
        provider = SubAgentToolProvider([researcher, writer])

        tools = await provider.list_tools()

        names = {t.name for t in tools}
        assert names == {"researcher_ask", "writer_ask"}

    async def test_synthetic_schema_is_flat_message_string(self) -> None:
        sub = _build_sub_agent("researcher")
        provider = SubAgentToolProvider([sub])

        (tool,) = await provider.list_tools()

        assert isinstance(tool, MCPTool)
        assert tool.name == "researcher_ask"
        assert tool.description is not None
        # Description mentions the sub-agent name so orchestrators with
        # multiple delegates can disambiguate.
        assert "researcher" in tool.description
        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert schema["properties"] == {"message": {"type": "string"}}
        assert schema["required"] == ["message"]

    async def test_inner_tools_come_first_then_synthetic(self) -> None:
        # Documents the ordering contract: real MCP tools first, sub-agent
        # delegations after.  Mirrors the legacy `add_tool`-then-`list_tools`
        # behaviour where pre-existing tools registered before
        # install_sub_agents would surface in registration order.
        inner = _StubInnerProvider(
            tools=[
                MCPTool(
                    name="add",
                    description="add tool",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ],
        )
        sub = _build_sub_agent("researcher")
        provider = SubAgentToolProvider(
            [sub], inner=cast(ToolProvider, inner),
        )

        tools = await provider.list_tools()

        assert [t.name for t in tools] == ["add", "researcher_ask"]
        assert inner.list_calls == 1

    async def test_empty_provider_returns_empty_list(self) -> None:
        provider = SubAgentToolProvider([])
        assert await provider.list_tools() == []


# ---------------------------------------------------------------------------
# call_tool — sub-agent dispatch
# ---------------------------------------------------------------------------


class TestCallToolDispatch:
    async def test_dispatches_match_with_live_parent_id(self) -> None:
        sub = _spy_sub_agent("researcher")
        provider = SubAgentToolProvider([sub])

        with bind_parent_episode_id("ep-orch-123"):
            result = await provider.call_tool(
                "researcher_ask", {"message": "hi"},
            )

        sub.invoke.assert_called_once_with(
            "hi", tags={PARENT_EPISODE_ID_TAG: "ep-orch-123"},
        )
        assert result.is_error is False
        assert result.content[0].text == "sub-answer"  # type: ignore[union-attr]

    async def test_falls_back_to_unknown_when_no_parent_bound(self) -> None:
        sub = _spy_sub_agent()
        provider = SubAgentToolProvider([sub])

        await provider.call_tool("researcher_ask", {"message": "hi"})

        sub.invoke.assert_called_once_with(
            "hi", tags={PARENT_EPISODE_ID_TAG: "unknown"},
        )

    async def test_missing_message_returns_is_error(self) -> None:
        sub = _spy_sub_agent()
        provider = SubAgentToolProvider([sub])

        result = await provider.call_tool("researcher_ask", {})

        assert result.is_error is True
        assert "message" in result.content[0].text  # type: ignore[union-attr]
        sub.invoke.assert_not_called()

    async def test_non_string_message_returns_is_error(self) -> None:
        sub = _spy_sub_agent()
        provider = SubAgentToolProvider([sub])

        result = await provider.call_tool(
            "researcher_ask", {"message": 42},
        )

        assert result.is_error is True
        sub.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# call_tool — fall-through to inner
# ---------------------------------------------------------------------------


class TestCallToolFallThrough:
    async def test_unknown_tool_with_inner_delegates(self) -> None:
        inner_result = _ok_call_result("from-inner")
        inner = _StubInnerProvider(result=inner_result)
        sub = _spy_sub_agent("researcher")
        provider = SubAgentToolProvider(
            [sub], inner=cast(ToolProvider, inner),
        )

        result = await provider.call_tool("add", {"a": 1, "b": 2}, timeout=5.0)

        assert result is inner_result
        assert inner.tool_calls == [("add", {"a": 1, "b": 2})]
        sub.invoke.assert_not_called()

    async def test_unknown_tool_no_inner_returns_is_error(self) -> None:
        sub = _spy_sub_agent("researcher")
        provider = SubAgentToolProvider([sub])

        result = await provider.call_tool("does_not_exist", {"x": 1})

        assert result.is_error is True
        # Message lists known sub-agents so the LLM has a useful hint.
        text = result.content[0].text  # type: ignore[union-attr]
        assert "does_not_exist" in text
        assert "researcher_ask" in text
        sub.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    async def test_wraps_non_fatal_exception_in_sub_agent_error(self) -> None:
        sub = _spy_sub_agent("researcher")
        sub.invoke.return_value.result.side_effect = ValueError("boom")
        provider = SubAgentToolProvider([sub])

        with pytest.raises(SubAgentError) as exc_info:
            await provider.call_tool("researcher_ask", {"message": "hi"})

        # Carry the sub-agent name so orchestrator logs aren't anonymous.
        assert "researcher" in str(exc_info.value)
        # Preserve the cause chain so debuggers can introspect the root.
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "boom"

    async def test_reraises_fatal_error_unchanged(self) -> None:
        sub = _spy_sub_agent("researcher")
        sub.invoke.return_value.result.side_effect = GuardrailBlockedError(
            "hard stop",
        )
        provider = SubAgentToolProvider([sub])

        # T0AgentsFatalError subclasses (GuardrailBlockedError, …) must
        # propagate so the orchestrator surfaces the real type instead of
        # a stringified tool error.
        with pytest.raises(GuardrailBlockedError):
            await provider.call_tool("researcher_ask", {"message": "hi"})
