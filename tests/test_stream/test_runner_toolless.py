"""Unit tests for the toolless ``StreamingAgentLoop.stream`` contract.

When ``tool_provider=None`` (the default), ``StreamingAgentLoop.stream`` must:

* skip ``list_tools()`` entirely (single streamed inference, no MCP plumbing), and
* raise :class:`T0AgentsFatalError` if the LLM hallucinates a ``ToolCall``
  block — there is no client to dispatch through.

Mirrors the contract documented in ``tests/test_mcp/test_loop_toolless.py``
for the non-streaming ``run_agent``; both tests anchor the same
operator-facing fragment ``"tool calls from the LLM but no `tool_provider`
was configured"`` so log greps stay stable across the two runners.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import T0AgentsFatalError
from t0_agents.stream.events import DoneEvent
from t0_agents.stream.runner import StreamingAgentLoop
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0, chat_chunk, final_chunk, text_chunk, tool_call_chunk


class _ListToolsAssertingClient(FakeMCPClient):
    """Spy that fails the test if ``list_tools`` is ever invoked.

    Used to prove ``StreamingAgentLoop.stream(..., tool_provider=None, ...)``
    does not fall back to discovering tools through a different code path.
    """

    async def list_tools(self) -> Any:  # type: ignore[override]
        raise AssertionError(
            "list_tools must not be called in the toolless path"
        )


class TestStreamingLoopToolless:
    async def test_streaming_loop_toolless_yields_done_event(self) -> None:
        """``tool_provider=None`` → single streamed inference, no tools advertised."""
        episode = uuid4()
        inference = uuid4()
        fake_t0 = FakeT0(
            responses=[
                [
                    chat_chunk(
                        [text_chunk("chat-only stream", block_id="t1")],
                        episode_id=episode,
                        inference_id=inference,
                    ),
                    final_chunk(
                        episode_id=episode, inference_id=inference
                    ),
                ],
            ],
        )

        # A spy client exists in the test scope, but is **not** passed in.
        # If the loop ever reached for it, it would raise — proving the
        # toolless path does not allocate or query a client.
        spy = _ListToolsAssertingClient()

        runner = StreamingAgentLoop()
        events: list[Any] = []
        async for event in runner.stream(
            cast(AsyncTensorZeroGateway, fake_t0),
            None,
            "say hi without tools",
            "agent",
        ):
            events.append(event)

        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].result.response == "chat-only stream"
        assert done[0].result.turns == 1
        assert done[0].result.episode_id == str(episode)
        assert done[0].result.inference_ids == [str(inference)]
        # T0 saw no advertised tools and the call was streamed.
        assert fake_t0.calls[0]["additional_tools"] == []
        assert fake_t0.calls[0]["stream"] is True
        # The unused spy confirms we never even constructed an empty
        # tool list from a client; it is here purely as a sentinel.
        assert isinstance(spy, FakeMCPClient)

    async def test_streaming_loop_toolless_raises_on_hallucinated_tool_call(
        self,
    ) -> None:
        """LLM emits a ``ToolCall`` chunk despite ``tool_provider=None`` →
        ``T0AgentsFatalError`` with a message naming ``tool_provider``."""
        fake_t0 = FakeT0(
            responses=[
                [
                    chat_chunk(
                        [
                            tool_call_chunk(
                                raw_name="add",
                                raw_arguments='{"a": 1, "b": 2}',
                                block_id="hallu-1",
                            )
                        ]
                    ),
                    final_chunk(),
                ],
            ],
        )

        runner = StreamingAgentLoop()
        with pytest.raises(T0AgentsFatalError) as excinfo:
            async for _ in runner.stream(
                cast(AsyncTensorZeroGateway, fake_t0),
                None,
                "please use a tool I never gave you",
                "agent",
            ):
                pass

        message = str(excinfo.value)
        assert "tool calls from the LLM" in message
        assert "`tool_provider`" in message
        # Never name the legacy kwarg in the new error path.
        assert "mcp_client" not in message

    async def test_streaming_loop_empty_user_message_raises_value_error(
        self,
    ) -> None:
        """Dummy-default footgun: omitting ``user_message`` (or leaving it
        as the empty string that exists only as a Python-syntax
        workaround) must surface as a clear ``ValueError`` on the first
        ``__anext__`` instead of a cryptic gateway failure."""
        fake_t0 = FakeT0(responses=[])
        runner = StreamingAgentLoop()
        with pytest.raises(ValueError, match="requires non-empty"):
            async for _ in runner.stream(
                cast(AsyncTensorZeroGateway, fake_t0),
                None,
                "",
                "agent",
            ):
                pass

    async def test_streaming_loop_empty_function_name_raises_value_error(
        self,
    ) -> None:
        """Same guard for ``function_name``."""
        fake_t0 = FakeT0(responses=[])
        runner = StreamingAgentLoop()
        with pytest.raises(ValueError, match="requires non-empty"):
            async for _ in runner.stream(
                cast(AsyncTensorZeroGateway, fake_t0),
                None,
                "hello",
                "",
            ):
                pass

    async def test_streaming_loop_with_spy_client_passed_in(self) -> None:
        """Sanity flip-side: when the spy is passed as ``tool_provider``,
        the loop *does* hit ``list_tools`` (so the spy raises). Anchors the
        previous test's assertion that ``None`` skips that codepath."""
        fake_t0 = FakeT0(
            responses=[
                [
                    chat_chunk([text_chunk("unreachable", block_id="t1")]),
                    final_chunk(),
                ],
            ],
        )
        spy = _ListToolsAssertingClient()

        runner = StreamingAgentLoop()
        with pytest.raises(AssertionError, match="list_tools must not be called"):
            async for _ in runner.stream(
                cast(AsyncTensorZeroGateway, fake_t0),
                cast(Client[Any], spy),
                "hi",
                "agent",
            ):
                pass
