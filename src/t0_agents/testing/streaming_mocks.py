"""Streaming-specific testing helpers.

Two tools:

* :func:`fake_event_stream` — wrap a list of :class:`AgentEvent` s into
  an ``AsyncIterator[AgentEvent]``.  Most useful when testing *consumers*
  of :meth:`Agent.stream` (TUI renderers, SSE routes, WS bridges, etc.)
  without spinning up a real gateway.
* :class:`MockStreamingGateway` — a deliberately simple gateway whose
  ``inference(stream=True)`` yields a single :class:`ChatChunk` per
  scripted turn (no intra-turn delta streaming).  Enough to exercise
  :meth:`Agent.stream`'s turn-loop and tool-dispatch paths deterministically;
  not a substitute for real delta-level streaming tests.

**Test-only.**  Like the rest of :mod:`t0_agents.testing`, never import
these in a production code path.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from typing import Any
from uuid import UUID, uuid4

# ``ContentBlock`` comes via ``tensorzero.types`` so the chunk-building
# converter reflects the shape the streaming runner would see at runtime.
# The top-level :class:`tensorzero.Text` / :class:`tensorzero.ToolCall` /
# :class:`tensorzero.Thought` content classes are needed to translate
# each block into its chunk counterpart.
from tensorzero import Text, Thought, ToolCall
from tensorzero.types import (
    ChatChunk,
    ContentBlock,
    ContentBlockChunk,
    TextChunk,
    ThoughtChunk,
    ToolCallChunk,
    Usage,
)

from t0_agents.stream.events import AgentEvent


async def fake_event_stream(
    events: Iterable[AgentEvent],
) -> AsyncIterator[AgentEvent]:
    """Wrap an iterable of events into an async iterator.

    Handy for testing anything that consumes
    ``AsyncIterator[AgentEvent]`` — TUI renderers, SSE routes,
    background tasks — without booting a real streaming gateway.
    """
    for event in events:
        yield event


def _content_block_to_chunk(block: ContentBlock) -> ContentBlockChunk:
    """Convert a terminal :class:`ContentBlock` to its chunk counterpart.

    Minimal one-chunk-per-block translation.  TensorZero's real streaming
    format emits multiple delta chunks for the same block id; tests that
    need delta-level behaviour should drive
    :class:`~t0_agents.stream.assembler.TurnStreamAssembler` directly
    with hand-crafted chunks instead.
    """
    if isinstance(block, Text):
        return TextChunk(id=f"mock-text-{uuid4().hex[:8]}", text=block.text or "")
    if isinstance(block, ToolCall):
        return ToolCallChunk(
            id=block.id,
            raw_name=block.raw_name,
            raw_arguments=block.raw_arguments,
        )
    if isinstance(block, Thought):
        return ThoughtChunk(
            id=f"mock-thought-{uuid4().hex[:8]}",
            text=block.text,
        )
    raise TypeError(
        f"MockStreamingGateway: unsupported content block type "
        f"{type(block).__name__}; extend _content_block_to_chunk to handle it"
    )


class MockStreamingGateway:
    """A tiny streaming-capable duck of :class:`AsyncTensorZeroGateway`.

    Constructor takes a list of turns; each turn is a list of
    :class:`ContentBlock` instances.  On every ``inference(stream=True)``
    the gateway emits a single :class:`ChatChunk` carrying the chunk
    forms of those blocks, followed by a second terminal chunk that
    carries the :class:`Usage`.  This is enough for
    :meth:`Agent.stream` / :class:`~t0_agents.stream.runner.StreamingAgentLoop`
    to drive a full turn loop without a real gateway, though it does NOT
    reproduce TensorZero's real delta-level streaming — tests that need
    that granularity should use :class:`~t0_agents.stream.assembler.TurnStreamAssembler`
    directly with hand-written chunks.

    Calls with ``stream=False`` are rejected with ``ValueError`` so
    misconfigurations surface immediately — use :class:`MockGateway`
    for non-streaming tests.
    """

    def __init__(
        self,
        turns: list[list[ContentBlock]],
        *,
        episode_id: UUID | None = None,
    ) -> None:
        self._turns: deque[list[ContentBlock]] = deque(turns)
        self._episode_id = episode_id or uuid4()
        self.calls: list[dict[str, Any]] = []

    async def inference(
        self, *, stream: bool = False, **kwargs: Any
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append({"stream": stream, **kwargs})
        if not stream:
            raise ValueError(
                "MockStreamingGateway: inference(stream=False) is not supported. "
                "Use MockGateway for non-streaming tests."
            )

        # Parity with ``MockGateway.script``: emit an empty Text turn when
        # the script is exhausted so the agent loop terminates with an empty
        # assistant message rather than one with no content blocks at all.
        blocks = self._turns.popleft() if self._turns else [Text(text="")]
        episode_id = self._episode_id
        inference_id = uuid4()

        async def generator() -> AsyncIterator[ChatChunk]:
            content_chunks: list[ContentBlockChunk] = [
                _content_block_to_chunk(b) for b in blocks
            ]
            yield ChatChunk(
                inference_id=inference_id,
                episode_id=episode_id,
                variant_name="mock_variant",
                content=content_chunks,
                usage=None,
            )
            yield ChatChunk(
                inference_id=inference_id,
                episode_id=episode_id,
                variant_name="mock_variant",
                content=[],
                usage=Usage(input_tokens=0, output_tokens=0),
            )

        return generator()


__all__ = [
    "MockStreamingGateway",
    "fake_event_stream",
]
