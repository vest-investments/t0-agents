"""Fold an ``AsyncIterator[ChatChunk]`` into an ``AsyncIterator[AgentEvent]``.

Strategy (validated against ``tensorzero/types.py:416-428``): each inbound
``ChatChunk`` carries a list of content-block deltas with stable ``id`` s; the
terminal chunk has ``usage`` populated.  The assembler emits delta events
immediately as chunks arrive, accumulates block state keyed by ``id``, and
flushes consolidation events (``TextEndEvent``, ``ToolCallEvent``,
``ThoughtEvent``) once the input iterator is exhausted — in the order the
blocks first appeared.

``JsonChunk`` is out of scope (``Run.events`` is chat-only) and surfaces as
an :class:`ErrorEvent`.  ``UnknownChunk`` is tolerated the same way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from tensorzero.types import (
    ChatChunk,
    JsonChunk,
    TextChunk,
    ThoughtChunk,
    ToolCallChunk,
    UnknownChunk,
)

from t0_agents.stream.events import (
    AgentEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
)

logger = logging.getLogger(__name__)


class TurnStreamAssembler:
    """Folds one turn's T0 chunk stream into the ``AgentEvent`` stream.

    A fresh assembler is built per turn inside :class:`StreamingAgentLoop`;
    the instance holds the per-block accumulators so the implementation can
    stay as a single async-generator method without module-level state.
    """

    def __init__(self) -> None:
        # Per-block accumulators keyed by the chunk's ``id`` field.
        # Insertion order is preserved (Python 3.7+ dict ordering) so the
        # final flush emits consolidation events in the order blocks first
        # appeared.
        self._text: dict[str, str] = {}
        self._tool: dict[str, dict[str, str]] = {}
        self._thought: dict[str, str] = {}
        self._invalid_args: dict[str, str] = {}

    @property
    def text_blocks(self) -> dict[str, str]:
        """Accumulated text, keyed by content-block ``id`` (read-only view).

        Consumers such as :class:`StreamingAgentLoop` read this after
        iteration ends to reconstruct the assistant's ``Text`` history.
        """
        return self._text

    @property
    def thought_blocks(self) -> dict[str, str]:
        """Accumulated reasoning text, keyed by content-block ``id``."""
        return self._thought

    @property
    def tool_calls(self) -> dict[str, dict[str, str]]:
        """Raw tool-call accumulators (``raw_name`` / ``raw_arguments``)."""
        return self._tool

    @property
    def invalid_tool_args(self) -> dict[str, str]:
        """Raw argument strings that failed JSON parsing, keyed by block ``id``.

        Populated during the final flush.  :class:`StreamingAgentLoop`
        rebuilds these blocks with ``arguments=None`` so dispatch applies
        the same no-dispatch error contract as the non-streaming loop
        (TensorZero marks invalid LLM JSON with ``arguments=None``).
        An empty ``raw_arguments`` is a legitimate no-args call, never
        recorded here.
        """
        return self._invalid_args

    async def assemble(
        self, chunks: AsyncIterator[ChatChunk]
    ) -> AsyncIterator[AgentEvent]:
        """Yield :class:`AgentEvent` s for every chunk, then flush consolidation."""

        async for chunk in chunks:
            if isinstance(chunk, JsonChunk):
                yield ErrorEvent(
                    message="JsonChunk is not supported by Run.events()",
                    detail={"variant_name": chunk.variant_name},
                )
                continue
            if not isinstance(chunk, ChatChunk):
                # Defensive: the SDK union is ``ChatChunk | JsonChunk`` today
                # but duck-typed fakes in tests may slip through.
                yield ErrorEvent(
                    message=f"unexpected chunk type: {type(chunk).__name__}",
                )
                continue

            for block in chunk.content:
                async for event in self._handle_block(block):
                    yield event

        # Final flush — emit consolidation events in first-appearance order.
        for tid, args_state in self._tool.items():
            async for event in self._flush_tool_call(tid, args_state):
                yield event
        for tid in self._text:
            yield TextEndEvent(id=tid)
        for tid, text in self._thought.items():
            yield ThoughtEvent(id=tid, text=text)

    async def _handle_block(
        self, block: object
    ) -> AsyncIterator[AgentEvent]:
        if isinstance(block, TextChunk):
            self._text[block.id] = self._text.get(block.id, "") + block.text
            yield TextDeltaEvent(id=block.id, delta=block.text)
            return
        if isinstance(block, ToolCallChunk):
            state = self._tool.setdefault(
                block.id, {"raw_name": "", "raw_arguments": ""}
            )
            state["raw_name"] += block.raw_name
            state["raw_arguments"] += block.raw_arguments
            yield ToolCallDeltaEvent(
                id=block.id,
                raw_name_delta=block.raw_name,
                raw_arguments_delta=block.raw_arguments,
            )
            return
        if isinstance(block, ThoughtChunk):
            # ThoughtChunk carries reasoning content via either ``text`` (raw
            # thought) or ``summary_text`` (provider summary, e.g. gpt-oss
            # harmony's analysis channel, OpenAI o-series reasoning summaries).
            # Fold both into a single delta — when both are present we
            # interpose a newline so the merged stream matches the
            # ``mcp/loop.py`` non-streaming fallback's ``"\n".join(...)``
            # of ``Thought.text`` + ``Thought.summary[*].text`` entries.
            text_part = block.text or ""
            summary_part = block.summary_text or ""
            separator = "\n" if text_part and summary_part else ""
            delta = text_part + separator + summary_part
            if not delta:
                return
            self._thought[block.id] = self._thought.get(block.id, "") + delta
            yield ThoughtDeltaEvent(id=block.id, delta=delta)
            return
        if isinstance(block, UnknownChunk):
            logger.warning(
                "Unknown chunk type received (id=%s, data=%.200s); ignoring",
                block.id,
                block.data,
            )
            yield ErrorEvent(
                message="unknown TensorZero chunk type",
                detail={"id": block.id},
            )
            return
        # Defensive default — foreign object types are logged but never crash.
        logger.warning("Unrecognised block type %s; skipping", type(block).__name__)
        yield ErrorEvent(
            message=f"unrecognised block type: {type(block).__name__}",
        )

    async def _flush_tool_call(
        self, tool_id: str, state: dict[str, str]
    ) -> AsyncIterator[AgentEvent]:
        """Parse accumulated raw JSON args and emit the terminal events.

        On valid JSON: yields one :class:`ToolCallEvent` with parsed args.

        On invalid JSON or non-object JSON: yields an :class:`ErrorEvent`
        describing the parse failure **and** a :class:`ToolCallEvent` with
        empty args so the downstream loop still observes the id.
        """
        name = state["raw_name"]
        raw_args = state["raw_arguments"]
        if not raw_args:
            yield ToolCallEvent(id=tool_id, name=name, arguments={})
            return
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid tool-call JSON arguments (id=%s, name=%s): %s",
                tool_id,
                name,
                exc,
            )
            self._invalid_args[tool_id] = raw_args
            yield ErrorEvent(
                message=f"invalid tool-call JSON arguments: {exc}",
                detail={"tool_call_id": tool_id, "name": name},
            )
            yield ToolCallEvent(id=tool_id, name=name, arguments={})
            return
        if not isinstance(parsed, dict):
            logger.error(
                "Tool-call JSON arguments are not an object (id=%s, type=%s)",
                tool_id,
                type(parsed).__name__,
            )
            self._invalid_args[tool_id] = raw_args
            yield ErrorEvent(
                message="tool-call JSON arguments must be an object",
                detail={"tool_call_id": tool_id, "name": name},
            )
            yield ToolCallEvent(id=tool_id, name=name, arguments={})
            return
        yield ToolCallEvent(id=tool_id, name=name, arguments=parsed)
