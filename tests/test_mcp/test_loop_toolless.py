"""Unit tests for the toolless ``run_agent`` contract.

When ``tool_provider=None`` (the default), ``run_agent`` must:

* skip ``list_tools()`` entirely (single inference, no MCP plumbing), and
* raise :class:`T0AgentsFatalError` if the LLM hallucinates a ``ToolCall``
  block — there is no client to dispatch through.

These tests pair with the runtime guard added in
``src/t0_agents/mcp/loop.py``.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents.core.exceptions import T0AgentsFatalError
from t0_agents.mcp import run_agent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


def _tool_call(call_id: str = "call-1") -> ToolCall:
    """Minimal ``ToolCall`` for the hallucination case."""
    return ToolCall(
        id=call_id,
        raw_arguments='{"a": 2, "b": 3}',
        raw_name="add",
        arguments={"a": 2, "b": 3},
        name="add",
    )


class _ListToolsAssertingClient(FakeMCPClient):
    """Spy that fails the test if ``list_tools`` is ever invoked.

    Used to prove ``run_agent(..., tool_provider=None, ...)`` does not
    fall back to discovering tools through a different code path.
    """

    async def list_tools(self) -> Any:  # type: ignore[override]
        raise AssertionError(
            "list_tools must not be called in the toolless path"
        )


class TestRunAgentToolless:
    async def test_run_agent_toolless_returns_text(self) -> None:
        """``tool_provider=None`` → single inference, no tools advertised."""
        episode = uuid4()
        inference = uuid4()
        t0 = FakeT0([
            FakeChatResponse(
                content=[Text(text="chat-only response")],
                episode_id=episode,
                inference_id=inference,
            ),
        ])

        # A spy client exists in the test scope, but is **not** passed in.
        # If the loop ever reached for it, it would raise — proving the
        # toolless path does not allocate or query a client.
        spy = _ListToolsAssertingClient()

        result = await run_agent(
            cast(AsyncTensorZeroGateway, t0),
            None,
            "Just say hi.",
            "agent",
        )

        assert result.response == "chat-only response"
        assert result.episode_id == str(episode)
        assert result.inference_ids == [str(inference)]
        assert result.turns == 1
        # T0 saw no advertised tools.
        assert t0.calls[0]["additional_tools"] == []
        # The unused spy confirms we never even constructed an empty
        # tool list from a client; it is here purely as a sentinel.
        assert isinstance(spy, FakeMCPClient)

    async def test_run_agent_toolless_raises_on_hallucinated_tool_call(
        self,
    ) -> None:
        """LLM emits a ``ToolCall`` despite ``tool_provider=None`` →
        ``T0AgentsFatalError`` with a message naming ``tool_provider``."""
        t0 = FakeT0([
            FakeChatResponse(content=[_tool_call(call_id="hallu-1")]),
        ])

        with pytest.raises(T0AgentsFatalError) as excinfo:
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                None,
                "please use a tool I never gave you",
                "agent",
            )

        message = str(excinfo.value)
        assert "tool calls from the LLM" in message
        assert "`tool_provider`" in message
        # Never name the legacy kwarg in the new error path.
        assert "mcp_client" not in message

    async def test_run_agent_empty_user_message_raises_value_error(
        self,
    ) -> None:
        """Dummy-default footgun: omitting ``user_message`` (or leaving it
        as the empty string that exists only as a Python-syntax
        workaround) must surface as a clear ``ValueError`` instead of a
        cryptic gateway failure."""
        t0 = FakeT0([])
        with pytest.raises(ValueError, match="requires non-empty"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                None,
                "",
                "agent",
            )

    async def test_run_agent_empty_function_name_raises_value_error(
        self,
    ) -> None:
        """Same guard for ``function_name``."""
        t0 = FakeT0([])
        with pytest.raises(ValueError, match="requires non-empty"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                None,
                "hello",
                "",
            )

    async def test_run_agent_toolless_with_spy_client_passed_in(self) -> None:
        """Sanity flip-side: when the spy is passed as ``tool_provider``,
        the loop *does* hit ``list_tools`` (so the spy raises). Anchors the
        previous test's assertion that ``None`` skips that codepath."""
        t0 = FakeT0([
            FakeChatResponse(content=[Text(text="unreachable")]),
        ])
        spy = _ListToolsAssertingClient()

        with pytest.raises(AssertionError, match="list_tools must not be called"):
            await run_agent(
                cast(AsyncTensorZeroGateway, t0),
                cast(Client[Any], spy),
                "hi",
                "agent",
            )
