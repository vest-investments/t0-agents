"""TensorZero-backed memory: read cross-turn history from ``list_inferences``.

TensorZero already persists every chat inference to ClickHouse at gateway
call time.  This module exposes that observability log as a
:class:`MemoryProvider` so an agent can recover its own conversation
history without running a parallel store.

The backend ships in the base distribution because ``tensorzero`` is
already a required dependency.  Two reads are supported:

- :meth:`TensorZeroMemory.load` — records-shaped, used by the
  ``prepended_message`` and ``system_template`` memory strategies.
- :meth:`TensorZeroMemory.load_history` — TZ-native ``list[Message]``,
  used by the recommended ``memory_strategy="messages"`` so the
  conversation is threaded through ``input.messages`` without ever
  being stringified into a summary.

``save`` is a no-op: TZ wrote the inference at the moment the call
was made, so persisting again would only duplicate the row.

``search`` returns ``[]`` by default; opt in to TensorZero's
experimental full-text endpoint via ``experimental_search=True`` if
the trade-off (full-table scans, contract may change) is acceptable.

Multi-tenant isolation rides on TZ's existing tag mechanism — pair
``TensorZeroMemory(tag_filter={"customer_id": cid})`` with
``AgentConfig(tags={"customer_id": cid})`` and reads are scoped to
that tenant by construction.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, cast

from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceFilterAnd,
    InferenceFilterTag,
    ListInferencesRequest,
    Message,
    OrderBy,
    RawText,
    Text,
    ToolCall,
    ToolResult,
)
from tensorzero.generated_types import (
    ContentBlockChatOutputText,
    InferenceFilter,
    StoredInferenceChat,
    StoredInputMessage,
    StoredInputMessageContentRawText,
    StoredInputMessageContentText,
    StoredInputMessageContentThought,
    StoredInputMessageContentToolCall,
    StoredInputMessageContentToolResult,
)

from t0_agents.core.conversation import ConversationSummary
from t0_agents.memory.base import MemoryRecord

logger = logging.getLogger(__name__)

#: Display caps for :meth:`TensorZeroMemory.list_conversations` summaries.
_TITLE_MAX_CHARS = 60
_PREVIEW_MAX_CHARS = 200


class TensorZeroMemory:
    """:class:`MemoryProvider` reading from the TensorZero gateway log.

    Works against any chat function that already runs through
    TensorZero — there is nothing to provision: the gateway has been
    persisting inferences since turn one.

    Args:
        gateway: An open :class:`AsyncTensorZeroGateway`.  Must outlive
            the memory instance — ``load`` and ``load_history`` issue
            calls on this gateway, so closing it before the memory is
            released will surface as failed reads.
        function_name: Optional TensorZero function name to filter on.
            Including it lets ClickHouse partition-prune the query and
            is strongly recommended for non-trivial workloads.  ``None``
            queries across functions, which is convenient for ad-hoc
            scripts but slow on large logs.
        tag_filter: Equality-only convenience filter.  Each
            ``key: value`` pair becomes an ``InferenceFilterTag(=)`` and
            the pairs are AND-combined.  Pair with the same tags on
            ``AgentConfig.tags`` so reads and writes line up.
        extra_filters: Raw :class:`InferenceFilter` passthrough for
            non-equality predicates (``InferenceFilterTime``,
            ``InferenceFilterOr``, ``InferenceFilterNot``, manual
            ``InferenceFilterAnd`` trees).  AND-combined with
            ``tag_filter`` when both are provided.
        load_limit: Maximum number of inferences ``load`` may return.
            Defaults to ``DEFAULT_LOAD_LIMIT`` (10), mirroring the
            renderer's ``max_records`` cap so we never fetch records
            the prompt summary would discard.
        experimental_search: When ``True``, ``search`` forwards the
            query to TZ's ``search_query_experimental``.  The gateway
            flags this endpoint as experimental and may change or
            remove it without warning; defaults to ``False``, which
            makes ``search`` a no-op returning ``[]``.

    The instance does not own the gateway lifecycle.  Wire it inside
    your existing ``async with await gateway as t0:`` block (or, in the
    serving layer, inside the per-session SessionFactory).
    """

    DEFAULT_LOAD_LIMIT = 10
    """Mirrors :func:`t0_agents.memory.base.render_memory_summary`'s
    ``max_records``.  Fetching more is wasted I/O for the records-
    returning path."""

    DEFAULT_LIST_WINDOW = 200
    """Default number of recent inferences :meth:`list_conversations` scans in
    its single ``list_inferences`` call before grouping by episode.  Bounds the
    cost (there is no server-side episode aggregation) at the price of horizon:
    conversations whose latest turn predates the window are not listed."""

    def __init__(
        self,
        gateway: AsyncTensorZeroGateway,
        *,
        function_name: str | None = None,
        tag_filter: dict[str, str] | None = None,
        extra_filters: InferenceFilter | None = None,
        load_limit: int = DEFAULT_LOAD_LIMIT,
        experimental_search: bool = False,
    ) -> None:
        self._gateway = gateway
        self._function_name = function_name
        self._tag_filter = dict(tag_filter) if tag_filter else {}
        self._extra_filters = extra_filters
        self._load_limit = load_limit
        self._experimental_search = experimental_search

    # ------------------------------------------------------------------
    # MemoryProvider Protocol
    # ------------------------------------------------------------------

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        """Return up to ``load_limit`` recent records for *episode_id*.

        One :class:`MemoryRecord` is produced per chat inference that
        emitted a non-empty text response.  Pure-tool-call inferences
        and JSON-output inferences are skipped.  The list is reversed
        from the gateway's DESC-by-timestamp order so the caller sees
        oldest-first, matching the convention of :class:`InMemoryMemory`
        and :class:`SqliteMemory`.

        ``MemoryRecord.turns`` is always ``1`` and
        ``MemoryRecord.inference_ids`` is a single-element list — this
        backend exposes one record per inference rather than one per
        agent run because that is the unit TensorZero persists.
        """
        request = ListInferencesRequest(
            function_name=self._function_name,
            episode_id=episode_id,
            filters=self._build_filters(),
            order_by=[
                OrderBy(by="timestamp", direction="descending")  # type: ignore[list-item]
            ],
            limit=self._load_limit,
        )
        response = await self._gateway.list_inferences(request=request)
        records: list[MemoryRecord] = []
        for inf in response.inferences:
            if not isinstance(inf, StoredInferenceChat):
                continue
            user_msg = _extract_last_user_message(inf.input)
            agent_resp = _extract_agent_text(inf.output)
            if not agent_resp:
                continue
            records.append(
                MemoryRecord(
                    episode_id=str(inf.episode_id),
                    user_message=user_msg,
                    agent_response=agent_resp,
                    turns=1,
                    inference_ids=[str(inf.inference_id)],
                    extra={},
                    created_at=_parse_timestamp(inf.timestamp),
                )
            )
        return list(reversed(records))

    async def save(self, record: MemoryRecord) -> None:
        """No-op.

        TensorZero is the writer — it persisted the row at the moment
        the inference was made.  Calling ``save`` again would duplicate
        the observability entry without adding information.
        """
        return None

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Substring-match across user and assistant text.

        Returns ``[]`` unless ``experimental_search=True`` was passed
        to the constructor.  When opted in, forwards the query to
        TensorZero's ``search_query_experimental`` field.  TZ documents
        that endpoint as experimental and warns it may cause full-
        table scans; it is not appropriate for low-latency request
        paths.
        """
        if not self._experimental_search:
            return []
        request = ListInferencesRequest(
            function_name=self._function_name,
            filters=self._build_filters(),
            order_by=[
                OrderBy(by="timestamp", direction="descending")  # type: ignore[list-item]
            ],
            limit=limit,
            search_query_experimental=query,
        )
        response = await self._gateway.list_inferences(request=request)
        results: list[MemoryRecord] = []
        for inf in response.inferences:
            if not isinstance(inf, StoredInferenceChat):
                continue
            user_msg = _extract_last_user_message(inf.input)
            agent_resp = _extract_agent_text(inf.output)
            if not agent_resp:
                continue
            results.append(
                MemoryRecord(
                    episode_id=str(inf.episode_id),
                    user_message=user_msg,
                    agent_response=agent_resp,
                    turns=1,
                    inference_ids=[str(inf.inference_id)],
                    extra={},
                    created_at=_parse_timestamp(inf.timestamp),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Native-history extension (used by memory_strategy="messages")
    # ------------------------------------------------------------------

    async def load_history(
        self,
        episode_id: str,
    ) -> list[Message] | None:
        """Return the prior conversation as TZ-native ``list[Message]``.

        Issues a single ``list_inferences(limit=1, order_by=DESC)`` and
        reuses the most recent inference's ``input.messages`` as the
        cumulative history (TensorZero accumulates within an episode,
        so the latest inference's input already covers every earlier
        turn).  The corresponding ``output`` is appended as the
        assistant reply so the caller sees the full exchange.

        Returns ``None`` when the episode has no inferences yet — a
        signal to the :class:`Agent` that this is a fresh session and
        no seeding is needed.

        ``StoredInputMessageContent`` variants the gateway cannot
        replay verbatim (templates with arguments, file blocks, opaque
        unknown blocks) are dropped with a single warning per call so
        the model never sees a half-decoded message.

        Assistant ``Thought`` blocks are dropped on this replay path —
        they remain in the ClickHouse inference store, but re-feeding
        them to the gateway both triggers groq's input-thought rejection
        and grows the prompt without bound as the conversation deepens.
        The drop is silent (no warning) because it is by design, not a
        decode failure. A turn whose only content was a thought yields no
        replayed message at all (it is skipped, never emitted empty).

        ``ToolCall`` blocks are dropped from the appended assistant
        message (debug-logged): it is built from the LAST inference's
        output, so a tool call there is unanswered by construction — its
        result would live in the input of a next inference that never
        ran, meaning the turn died mid-tool-loop.  Replaying it puts an
        assistant tool_call directly before the next user message, which
        OpenAI-compatible providers reject; dropping it lets the model
        re-decide the call.  A normal turn ends in text and is untouched.
        """
        request = ListInferencesRequest(
            function_name=self._function_name,
            episode_id=episode_id,
            filters=self._build_filters(),
            order_by=[
                OrderBy(by="timestamp", direction="descending")  # type: ignore[list-item]
            ],
            limit=1,
        )
        response = await self._gateway.list_inferences(request=request)
        latest: StoredInferenceChat | None = None
        for inf in response.inferences:
            if isinstance(inf, StoredInferenceChat):
                latest = inf
                break
        if latest is None:
            return None

        history: list[Message] = []
        skipped_kinds: set[str] = set()
        stored_input = latest.input
        stored_msgs: Iterable[StoredInputMessage] = (
            (stored_input.messages or []) if stored_input is not None else []
        )
        for stored in stored_msgs:
            converted = _to_native_message(stored, skipped_kinds)
            if converted is not None:
                history.append(converted)

        # Append the most recent assistant reply (TZ stores it as
        # ``output``, not as a message inside ``input.messages`` — the
        # input only carries the turn-N messages BEFORE the model
        # responded).  Tool calls here are dangling by construction —
        # see the docstring — so they never replay.
        assistant_blocks = _output_to_content_blocks(latest.output)
        dangling = sum(1 for b in assistant_blocks if isinstance(b, ToolCall))
        if dangling:
            assistant_blocks = [
                b for b in assistant_blocks if not isinstance(b, ToolCall)
            ]
            logger.debug(
                "TensorZeroMemory.load_history: dropped %d dangling "
                "tool_call block(s) from the final assistant message of "
                "episode %s — the turn died mid-tool-loop, so their "
                "results were never produced.",
                dangling,
                episode_id,
            )
        if assistant_blocks:
            history.append(
                Message(role="assistant", content=assistant_blocks),
            )

        if skipped_kinds:
            logger.warning(
                "TensorZeroMemory.load_history: skipped %d unsupported "
                "content-block kind(s): %s. The model will see the rest of "
                "the conversation but not those blocks.",
                len(skipped_kinds),
                sorted(skipped_kinds),
            )
        return history

    # ------------------------------------------------------------------
    # Conversation listing (ConversationIndex protocol)
    # ------------------------------------------------------------------

    async def list_conversations(
        self,
        *,
        tags: Mapping[str, str] | None = None,
        limit: int = 20,
        window: int = DEFAULT_LIST_WINDOW,
    ) -> list[ConversationSummary]:
        """List recent conversations, newest first — a **dev-grade** index.

        TensorZero 2026.4.0 exposes no server-side per-episode aggregation, so
        this groups a **bounded window** of the most recent inferences by
        ``episode_id`` entirely client-side.  It is honest about the resulting
        limits:

        - **O(window)** — one ``list_inferences(limit=window)`` call, not a scan
          of the whole log.
        - A conversation whose latest turn predates the window is **not listed**;
          widen ``window`` (larger scan) to extend the horizon.
        - ``title`` / ``preview`` / ``turn_count`` are read from each
          conversation's **most recent** inference input.  Under the
          recommended ``memory_strategy="messages"`` the gateway accumulates the
          whole episode into that input, so ``title`` is the first user message
          and ``turn_count`` is exact.  Without message-history accumulation the
          input carries only the latest turn, so ``title`` collapses to the
          latest message and ``turn_count`` is a lower bound (``1``).

        Scoped by ``tags`` merged over the instance ``tag_filter`` — pass e.g.
        ``tags={"customer_id": cid}`` to list one tenant's conversations.  For
        production-scale listing, back the agent with a dedicated
        :class:`~t0_agents.core.conversation.ConversationIndex` instead; this
        method satisfies that protocol so it is a drop-in for development.
        """
        request = ListInferencesRequest(
            function_name=self._function_name,
            filters=self._build_filters(tags),
            order_by=[
                OrderBy(by="timestamp", direction="descending")  # type: ignore[list-item]
            ],
            limit=window,
        )
        response = await self._gateway.list_inferences(request=request)

        # DESC by timestamp → the first inference seen for an episode is its
        # latest turn, whose input carries the most context (the whole episode
        # under memory_strategy="messages").
        latest_by_episode: dict[str, StoredInferenceChat] = {}
        for inf in response.inferences:
            if isinstance(inf, StoredInferenceChat):
                latest_by_episode.setdefault(str(inf.episode_id), inf)

        summaries: list[ConversationSummary] = []
        for episode_id, inf in latest_by_episode.items():
            users = _user_texts(inf.input)
            summaries.append(
                ConversationSummary(
                    conversation_id=episode_id,
                    title=_truncate(users[0], _TITLE_MAX_CHARS) if users else "",
                    preview=(
                        _truncate(users[-1], _PREVIEW_MAX_CHARS) if users else ""
                    ),
                    turn_count=len(users),
                    updated_at=int(
                        _parse_timestamp(inf.timestamp).timestamp() * 1000
                    ),
                )
            )

        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_filters(
        self, tags: Mapping[str, str] | None = None
    ) -> InferenceFilter | None:
        """Combine ``tag_filter`` + ``extra_filters`` into a single filter.

        Returns ``None`` when neither is set (TZ then runs an
        unfiltered query).  Single-tag with no extras returns the bare
        :class:`InferenceFilterTag`; anything more is wrapped in an
        :class:`InferenceFilterAnd`.

        When *tags* is passed it is merged over the instance ``tag_filter``
        (call-time keys win) — :meth:`list_conversations` uses this to scope a
        query without constructing a second memory instance.
        """
        tag_source = (
            self._tag_filter if tags is None else {**self._tag_filter, **tags}
        )
        tag_filters: list[InferenceFilter] = [
            cast(
                InferenceFilter,
                InferenceFilterTag(
                    key=key,
                    value=value,
                    comparison_operator="=",
                ),
            )
            for key, value in tag_source.items()
        ]
        children: list[InferenceFilter] = list(tag_filters)
        if self._extra_filters is not None:
            children.append(self._extra_filters)
        if not children:
            return None
        if len(children) == 1:
            return children[0]
        return cast(InferenceFilter, InferenceFilterAnd(children=children))


# ----------------------------------------------------------------------
# Module-level helpers (private)
# ----------------------------------------------------------------------


def _extract_last_user_message(stored_input: Any) -> str:
    """Return the most recent ``role=user`` text from a :class:`StoredInput`.

    Walks ``stored_input.messages`` in reverse until it finds a user
    message whose content yields at least one ``text`` or ``raw_text``
    block.  Tool-result-only and tool-call-only user messages (the
    intermediate steps of an agent loop) are skipped because they do
    not carry the user's "real" question.
    """
    messages = getattr(stored_input, "messages", None) or []
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, StoredInputMessageContentText):
                    if part.text:
                        texts.append(part.text)
                elif isinstance(part, StoredInputMessageContentRawText):
                    if part.value:
                        texts.append(part.value)
            joined = "\n".join(texts)
            if joined:
                return joined
    return ""


def _user_texts(stored_input: Any) -> list[str]:
    """All user-authored texts in a :class:`StoredInput`, oldest-first.

    One entry per ``role=user`` message that carried real text (``text`` /
    ``raw_text`` blocks joined with newlines); tool-result-only and
    tool-call-only user messages — the intermediate steps of an agent loop —
    are skipped.  The forward-ordered counterpart of
    :func:`_extract_last_user_message`, used by
    :meth:`TensorZeroMemory.list_conversations` to derive a title (first),
    preview (last), and turn count (length) from one cumulative input.
    """
    out: list[str] = []
    messages = getattr(stored_input, "messages", None) or []
    for msg in messages:
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            if content:
                out.append(content)
            continue
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, StoredInputMessageContentText):
                    if part.text:
                        texts.append(part.text)
                elif isinstance(part, StoredInputMessageContentRawText):
                    if part.value:
                        texts.append(part.value)
            joined = "\n".join(texts)
            if joined:
                out.append(joined)
    return out


def _truncate(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters for display, adding an ellipsis.

    Strips surrounding whitespace first; when the result exceeds ``limit`` it
    is cut to ``limit - 1`` characters (trailing whitespace removed) plus ``…``
    so the rendered string never exceeds ``limit``.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extract_agent_text(output_blocks: Any) -> str:
    """Return the joined ``text`` of every chat output block.

    Skips ``ToolCall``, ``Thought``, and any other non-text block.
    Returns ``""`` for tool-call-only or thought-only inferences so
    callers can treat them as "no user-facing reply".
    """
    if not isinstance(output_blocks, list):
        return ""
    texts: list[str] = []
    for block in output_blocks:
        if isinstance(block, ContentBlockChatOutputText):
            if block.text:
                texts.append(block.text)
        elif getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts)


def _parse_timestamp(ts: Any) -> dt.datetime:
    """Parse the ``StoredInference.timestamp`` field.

    TZ returns RFC 3339 strings with a ``Z`` UTC suffix
    (``"2026-05-01T02:20:42Z"``).  ``datetime.fromisoformat`` accepts
    ``Z`` natively in Python 3.11+.  When the gateway already gives us
    a ``datetime``, we ensure it carries UTC tzinfo to match the
    convention used by the rest of the memory subsystem.
    """
    if isinstance(ts, dt.datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.UTC)
    if isinstance(ts, str):
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return cast(dt.datetime, ts)


def _to_native_message(
    stored: StoredInputMessage,
    skipped_kinds: set[str],
) -> Message | None:
    """Convert a :class:`StoredInputMessage` to a runtime :class:`Message`.

    Each known content variant is mapped to its
    :mod:`tensorzero.types` counterpart so the result can be sent back
    to ``t0.inference()`` verbatim.  Variants the gateway cannot
    replay (Template-with-arguments, File, Unknown) are silently
    dropped and their ``type`` recorded in *skipped_kinds* for a
    single-line warning emitted by the caller.

    ``Thought`` blocks are dropped on this replay path (they stay in
    the store, never replayed) and are NOT recorded in *skipped_kinds*
    — the drop is intentional, so no warning fires.

    Returns ``None`` when no convertible content survives, so the
    message is discarded entirely (no empty-content message in the
    output).
    """
    converted: list[Any] = []
    for block in stored.content:
        if isinstance(block, StoredInputMessageContentText):
            converted.append(Text(text=block.text))
        elif isinstance(block, StoredInputMessageContentRawText):
            converted.append(RawText(value=block.value))
        elif isinstance(block, StoredInputMessageContentToolCall):
            converted.append(_stored_tool_call_to_runtime(block))
        elif isinstance(block, StoredInputMessageContentToolResult):
            converted.append(
                ToolResult(
                    id=block.id,
                    name=block.name,
                    result=block.result,
                ),
            )
        elif isinstance(block, StoredInputMessageContentThought):
            # Strip-on-replay: kept in the store, never replayed (groq rejects + unbounded growth).
            continue
        else:
            kind = getattr(block, "type", type(block).__name__)
            skipped_kinds.add(str(kind))
    if not converted:
        return None
    return Message(role=stored.role, content=converted)


def _stored_tool_call_to_runtime(
    stored: StoredInputMessageContentToolCall,
) -> ToolCall:
    """Adapt the storage-shape :class:`ToolCall` to :class:`tensorzero.ToolCall`.

    The stored form carries ``arguments`` as a JSON string and lacks
    ``raw_arguments``/``raw_name`` (those are runtime-only fields).
    We populate ``raw_arguments`` from the JSON string, re-parse
    ``arguments`` into a dict (best-effort; left as ``None`` on JSON
    errors so the model still sees ``raw_arguments``), and mirror
    ``name`` into ``raw_name``.
    """
    raw_args = stored.arguments or ""
    parsed: dict[str, Any] | None
    try:
        decoded = json.loads(raw_args) if raw_args else None
    except (TypeError, ValueError):
        parsed = None
    else:
        parsed = decoded if isinstance(decoded, dict) else None
    return ToolCall(
        id=stored.id,
        raw_arguments=raw_args,
        raw_name=stored.name,
        arguments=parsed,
        name=stored.name,
    )


def _output_to_content_blocks(output: Any) -> list[Any]:
    """Convert a :class:`StoredInferenceChat.output` list into runtime blocks.

    The gateway stores chat outputs as
    ``list[ContentBlockChatOutput*]``; we round-trip them through the
    runtime types so they match what ``run_agent`` would have appended
    on the assistant turn.  Non-chat blocks are skipped, as are
    ``Thought`` blocks — the same strip-on-replay rule as
    :func:`_to_native_message` (see :meth:`TensorZeroMemory.load_history`):
    thoughts stay in the store but are never replayed. A thought-only
    output therefore yields ``[]`` and the caller appends no assistant
    message.

    Output-side tool calls arrive as
    :class:`~tensorzero.generated_types.ContentBlockChatOutputToolCall`:
    ``arguments`` is the PARSED value (a dict — ``None`` when the model's
    JSON was invalid) and the model's raw JSON string lives in
    ``raw_arguments``; an INVALID call likewise has ``name=None`` but
    always carries ``raw_name``.  The replayed input side
    (``ToolCallWrapper``) demands ``str`` arguments in every union
    variant, so ``raw_arguments`` must be carried through as the string
    — anything else 400s every subsequent inference of the episode.
    """
    if not isinstance(output, list):
        return []
    blocks: list[Any] = []
    for item in output:
        if isinstance(item, ContentBlockChatOutputText):
            if item.text:
                blocks.append(Text(text=item.text))
        elif getattr(item, "type", None) == "tool_call":
            raw = getattr(item, "raw_arguments", None)
            args_obj = getattr(item, "arguments", None)
            raw_args: str
            if isinstance(raw, str):
                raw_args = raw
            elif isinstance(args_obj, str):
                raw_args = args_obj
            else:
                raw_args = json.dumps(args_obj) if args_obj is not None else ""
            parsed: dict[str, Any] | None
            if isinstance(args_obj, dict):
                parsed = cast("dict[str, Any]", args_obj)
            else:
                try:
                    decoded = json.loads(raw_args) if raw_args else None
                except (TypeError, ValueError):
                    parsed = None
                else:
                    parsed = decoded if isinstance(decoded, dict) else None
            blocks.append(
                ToolCall(
                    id=getattr(item, "id", ""),
                    raw_arguments=raw_args,
                    raw_name=getattr(item, "raw_name", None)
                    or getattr(item, "name", None)
                    or "",
                    arguments=parsed,
                    name=getattr(item, "name", None),
                ),
            )
    return blocks
