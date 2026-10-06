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
from typing import Literal

from tensorzero.types import (
    ChatChunk,
    JsonChunk,
    TextChunk,
    ThoughtChunk,
    ToolCallChunk,
    UnknownChunk,
)

from t0_agents.core.turn_engine import salvage_tool_arguments
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

BlockKind = Literal["text", "thought", "tool_call"]


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
        self._thought_meta: dict[str, dict[str, str | None]] = {}
        self._invalid_args: dict[str, str] = {}
        self._order: dict[tuple[BlockKind, str], None] = {}
        self._tool_ids: dict[str, list[str]] = {}

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
    def thought_metadata(self) -> dict[str, dict[str, str | None]]:
        """``signature`` / ``provider_type`` per reasoning block ``id``.

        Carried separately from the folded text because a provider may
        attach them to any chunk of the block, commonly the last.  A
        provider that verifies its own reasoning on replay rejects a
        block whose signature is missing, so this survives the rebuild
        the same way the text does.

        Also holds blocks that carry a signature and no text — Gemini's
        ``thoughtSignature`` on the part it signs, which the gateway
        splits off as its own thought.  Those never reach
        :attr:`thought_blocks`, so no empty ``ThoughtEvent`` is emitted.
        """
        return self._thought_meta

    @property
    def block_order(self) -> list[tuple[BlockKind, str]]:
        """Every block as ``(kind, id)``, in the order it first appeared.

        Ids are unique only within a kind.  The order is meaning on
        replay: the gateway folds a signature-only thought into the block
        after it, so a rebuild in any other order moves the signature
        off the call it signs.
        """
        return list(self._order)

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
        recorded here; neither is a call salvaged by
        :func:`~t0_agents.core.turn_engine.salvage_tool_arguments` — its
        recovered dict flows through as ordinary parsed arguments.
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
            self._order["text", block.id] = None
            self._text[block.id] = self._text.get(block.id, "") + block.text
            yield TextDeltaEvent(id=block.id, delta=block.text)
            return
        if isinstance(block, ToolCallChunk):
            tool_id, name_delta = self._route_tool_call(block)
            self._order["tool_call", tool_id] = None
            state = self._tool.setdefault(
                tool_id, {"raw_name": "", "raw_arguments": ""}
            )
            state["raw_name"] += name_delta
            state["raw_arguments"] += block.raw_arguments
            yield ToolCallDeltaEvent(
                id=tool_id,
                raw_name_delta=name_delta,
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
            if not delta and not block.signature:
                return
            self._order["thought", block.id] = None
            meta = self._thought_meta.setdefault(
                block.id, {"signature": None, "provider_type": None}
            )
            meta["signature"] = block.signature or meta["signature"]
            meta["provider_type"] = block.provider_type or meta["provider_type"]
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

    def _route_tool_call(self, block: ToolCallChunk) -> tuple[str, str]:
        """The accumulator ``id`` and name delta for one tool-call chunk.

        Gateway 2026.6.0 streams consecutive Gemini calls to the same tool
        under one ``id`` and names only the first: it mints an id only when
        the function name changes (``check_new_tool_call_name``).  Gemini
        sends each call as one whole JSON object, so a chunk that is a whole
        object, arriving after the current call's arguments already form
        one, starts the next call — ``"{id}.{n}"``, same name.  The ids
        stay local: Gemini pairs a call with its response by name.  A
        fragment of a second object is not a whole object and still fails
        as ambiguous, as it did before.
        """
        ids = self._tool_ids.setdefault(block.id, [block.id])
        current = self._tool.get(ids[-1])
        if (
            current is None
            or not _is_json_object(block.raw_arguments)
            or not _is_json_object(current["raw_arguments"])
        ):
            return ids[-1], block.raw_name
        ids.append(f"{block.id}.{len(ids)}")
        return ids[-1], block.raw_name or current["raw_name"]

    async def _flush_tool_call(
        self, tool_id: str, state: dict[str, str]
    ) -> AsyncIterator[AgentEvent]:
        """Parse accumulated raw JSON args and emit the terminal events.

        On valid JSON: yields one :class:`ToolCallEvent` with parsed args.

        On a complete object followed only by closing noise (``}`` / ``]``
        / whitespace): salvages it via
        :func:`~t0_agents.core.turn_engine.salvage_tool_arguments` and
        yields the same :class:`ToolCallEvent` — no :class:`ErrorEvent`,
        nothing recorded in :attr:`invalid_tool_args` — so the runtime
        dispatches the recovered arguments and the wire carries their
        canonical form.  Logged at WARNING with sizes only.

        On any other invalid JSON, or non-object JSON: yields an
        :class:`ErrorEvent` describing the parse failure **and** a
        :class:`ToolCallEvent` with empty args so the downstream loop
        still observes the id.
        """
        name = state["raw_name"]
        raw_args = state["raw_arguments"]
        if not raw_args:
            yield ToolCallEvent(id=tool_id, name=name, arguments={})
            return
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            salvaged = salvage_tool_arguments(raw_args)
            if salvaged is not None:
                logger.warning(
                    "Salvaged tool-call JSON arguments (id=%s, name=%s): "
                    "%d chars parsed, %d trailing chars discarded",
                    tool_id,
                    name,
                    salvaged.parsed,
                    salvaged.discarded,
                )
                yield ToolCallEvent(id=tool_id, name=name, arguments=salvaged.arguments)
                return
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


def _is_json_object(raw: str) -> bool:
    try:
        return isinstance(json.loads(raw), dict)
    except json.JSONDecodeError:
        return False
