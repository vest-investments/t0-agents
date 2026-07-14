"""Streaming analogue of :func:`t0_agents.mcp.loop.run_agent`.

Mirrors the existing loop shape — ``list_tools`` → inference → execute tool
calls → feed results back → loop — but uses ``t0.inference(stream=True)``
and yields :class:`AgentEvent` s as chunks arrive.  The non-streaming
:func:`run_agent` is untouched; the contract's ``mcp/``-standalone invariant
(§2.1) holds.

The runner reuses:

* ``convert_tools`` / ``build_tool_result`` from :mod:`t0_agents.mcp.adapter`.
* ``execute_tool_calls`` / ``execute_tool_calls_parallel`` from
  :mod:`t0_agents.mcp.loop`.
* ``RECOVERABLE_INFERENCE_PATTERNS`` / ``RECOVERABLE_INFERENCE_STATUS_CODES``
  for parity with :func:`run_agent`'s corrective-retry contract.

Memory / guardrail scaffolding stays in :meth:`Agent.invoke` so the runner
itself is agnostic of those subsystems — identical responsibility layering
to :func:`run_agent`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, cast

from fastmcp import Client
from tensorzero import (
    AsyncTensorZeroGateway,
    Message,
    Text,
    Thought,
    ToolCall,
    ToolChoice,
)
from tensorzero.types import (
    ChatChunk,
    ChatInferenceResponse,
    InferenceChunk,
    TensorZeroError,
)

from t0_agents.core.exceptions import MaxTurnsExceededError, T0AgentsFatalError
from t0_agents.core.types import AgentResult
from t0_agents.mcp.adapter import convert_tools
from t0_agents.mcp.loop import (
    RECOVERABLE_INFERENCE_PATTERNS,
    RECOVERABLE_INFERENCE_STATUS_CODES,
    ProgressCallback,
    execute_tool_calls,
    execute_tool_calls_parallel,
)
from t0_agents.stream.assembler import TurnStreamAssembler
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
    TurnEndEvent,
)

logger = logging.getLogger(__name__)


class StreamingAgentLoop:
    """Streaming agent loop — yields :class:`AgentEvent` s as they arrive.

    Stateless across calls; one instance can drive multiple sequential
    streams against the same ``(t0, tool_provider)`` pair safely.
    """

    async def stream(
        self,
        t0: AsyncTensorZeroGateway,
        tool_provider: Client[Any] | None = None,
        user_message: str = "",
        function_name: str | None = "",
        *,
        episode_id: str | None = None,
        max_turns: int = 20,
        tool_timeout: float = 30.0,
        parallel_tools: bool = False,
        dryrun: bool = False,
        tool_choice: ToolChoice | None = None,
        tags: dict[str, str] | None = None,
        namespace: str | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None,
        prior_messages: list[Message] | None = None,
        progress_sink: Callable[[ToolProgressEvent], Awaitable[None]] | None = None,
        model_name: str | None = None,
        variant_name: str | None = None,
        params: dict[str, Any] | None = None,
        cache_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the streaming loop and yield every event.

        The final yielded event is always a :class:`DoneEvent` wrapping the
        accumulated :class:`AgentResult`.  Raises :class:`MaxTurnsExceededError`
        if ``max_turns`` is exhausted without a tool-call-free turn.

        ``tool_provider`` mirrors :func:`~t0_agents.mcp.loop.run_agent`'s
        ``tool_provider`` kwarg: an already-connected ``fastmcp.Client``
        (or ``None``).  When ``None`` (the default), no tools are
        advertised to the LLM — the loop runs a single streamed inference
        and yields the resulting text without ever calling
        ``list_tools``.  If the LLM emits a tool call anyway, a
        :class:`~t0_agents.core.exceptions.T0AgentsFatalError` is raised
        before any dispatch is attempted.  When a client is provided, it
        **must** already be inside ``async with`` for the lifetime of
        the stream; it is forwarded to ``execute_tool_calls`` /
        ``execute_tool_calls_parallel`` on every tool-dispatch turn.

        ``prior_messages`` mirrors the kwarg of the same name on
        :func:`~t0_agents.mcp.loop.run_agent` — when supplied, the loop
        seeds ``messages`` with the prior conversation before appending
        the new user prompt.  Used by the ``messages`` memory strategy.

        ``system`` mirrors :func:`~t0_agents.mcp.loop.run_agent`'s
        ``system`` kwarg: a raw string forwarded as ``input.system``.
        Mutually exclusive with ``system_arguments``; a ``ValueError``
        is raised before the first chunk is yielded when both are set.
        ``user_message`` and ``function_name`` also raise ``ValueError``
        if left at their ``""`` defaults — those defaults exist solely
        as a Python-syntax workaround that lets ``tool_provider`` default
        to ``None`` at position 2.

        ``progress_sink`` receives a :class:`ToolProgressEvent` for every
        server-side progress notification a tool emits while executing.
        A sink (not a yield) because notifications arrive while the
        generator is suspended awaiting the tool executor — there is no
        yield point to ride.  ``Agent`` passes the Run's buffer ``put``
        so progress interleaves into the ordinary event stream.
        """
        if system is not None and system_arguments is not None:
            raise ValueError(
                "Pass either `system` (raw string for input.system) or "
                "`system_arguments` (dict for template variables), not both."
            )
        if system_arguments is not None and model_name is not None:
            raise ValueError(
                "system_arguments requires a tensorzero.toml function with "
                "a system template; model_name routes via the functionless "
                "passthrough, which has none. Pass system= (a raw prompt "
                "string), or declare the function and use function_name=."
            )
        if not user_message or not (function_name or model_name):
            raise ValueError(
                "`StreamingAgentLoop.stream` requires non-empty "
                "`user_message` and a routing target: `function_name` or "
                "`model_name`. The `\"\"` defaults exist only as a "
                "Python-syntax workaround so `tool_provider` can default "
                "to `None` at position 2."
            )
        mcp_tools = (
            await tool_provider.list_tools() if tool_provider is not None else []
        )
        t0_tools = convert_tools(mcp_tools)
        messages: list[Message] = list(prior_messages) if prior_messages else []
        messages.append(Message(role="user", content=user_message))
        inference_ids: list[str] = []
        executor = (
            execute_tool_calls_parallel if parallel_tools else execute_tool_calls
        )
        progress_callback: ProgressCallback | None = None
        if progress_sink is not None:
            sink = progress_sink

            async def _forward_progress(
                tool_call_id: str,
                progress: float,
                total: float | None,
                message: str | None,
            ) -> None:
                await sink(
                    ToolProgressEvent(
                        id=tool_call_id,
                        progress=progress,
                        total=total,
                        message=message,
                    )
                )

            progress_callback = _forward_progress

        logger.info(
            "Streaming agent loop started: target=%s, max_turns=%d, tools=%d",
            function_name or model_name,
            max_turns,
            len(t0_tools),
        )

        for turn in range(max_turns):
            chunks, err = await self._open_stream(
                t0=t0,
                function_name=function_name,
                model_name=model_name,
                variant_name=variant_name,
                episode_id=episode_id,
                messages=messages,
                t0_tools=t0_tools,
                dryrun=dryrun,
                tool_choice=tool_choice,
                tags=tags,
                namespace=namespace,
                system_arguments=system_arguments,
                system=system,
                params=params,
                cache_options=cache_options,
            )
            if err is not None:
                if self._is_recoverable(err):
                    logger.warning(
                        "Turn %d: recoverable TensorZeroError caught: %s",
                        turn + 1,
                        err.text,
                    )
                    tool_names_available = [t["name"] for t in t0_tools]
                    correction = (
                        f"The previous response contained an invalid tool call "
                        f"that was rejected: {err.text}\n"
                        f"Available tools: {tool_names_available}. "
                        f"Please retry with a valid tool name, stripping any "
                        f"special tokens."
                    )
                    yield ErrorEvent(
                        message="recoverable TensorZeroError; retrying",
                        detail={"text": err.text or ""},
                    )
                    messages.append(Message(role="assistant", content=[]))
                    messages.append(Message(role="user", content=correction))
                    inference_ids.append(f"error-recoverable-turn-{turn + 1}")
                    yield TurnEndEvent(turn=turn + 1)
                    continue
                raise err

            assert chunks is not None  # recoverable branch returned above
            assembler = TurnStreamAssembler()
            assistant_blocks: list[Any] = []
            tool_calls_collected: list[ToolCallEvent] = []
            turn_episode_id: str | None = None
            turn_inference_id: str | None = None
            last_chunk_ref: list[ChatChunk | None] = [None]

            tap = self._tap_final_chunk(chunks, last_chunk_ref)
            async for event in assembler.assemble(tap):
                if isinstance(event, ToolCallEvent):
                    tool_calls_collected.append(event)
                yield event

            final_usage_chunk = last_chunk_ref[0]

            if final_usage_chunk is not None:
                turn_episode_id = str(final_usage_chunk.episode_id)
                turn_inference_id = str(final_usage_chunk.inference_id)

            if turn_episode_id is not None:
                episode_id = turn_episode_id
            if turn_inference_id is not None:
                inference_ids.append(turn_inference_id)

            assistant_blocks = self._build_assistant_blocks(
                assembler=assembler,
                tool_calls=tool_calls_collected,
            )
            messages.append(Message(role="assistant", content=assistant_blocks))

            # Fire the observability hook with a synthesised ChatInferenceResponse
            # mirror so guardrails (``max_cost``) can accumulate usage exactly
            # as they do for ``run_agent``.
            if on_inference is not None and final_usage_chunk is not None:
                synthetic = _SyntheticChatResponse(
                    content=assistant_blocks,
                    episode_id=final_usage_chunk.episode_id,
                    inference_id=final_usage_chunk.inference_id,
                    usage=final_usage_chunk.usage,
                )
                await on_inference(cast(ChatInferenceResponse, synthetic))

            logger.info(
                "Stream turn %d: inference_id=%s, episode_id=%s, "
                "blocks=%d, tool_calls=%d",
                turn + 1,
                turn_inference_id,
                turn_episode_id,
                len(assistant_blocks),
                len(tool_calls_collected),
            )

            yield TurnEndEvent(turn=turn + 1)

            if not tool_calls_collected:
                text, response_source = _final_text(assistant_blocks)
                result: AgentResult[str] = AgentResult(
                    value=text,
                    episode_id=episode_id,
                    inference_ids=list(inference_ids),
                    turns=turn + 1,
                    response_source=response_source,
                )
                yield DoneEvent(result=result)
                return

            # Dispatch tool calls via the existing executor (reuses the
            # ``_execute_single_tool_call`` error contract verbatim).
            tool_call_blocks = [
                b for b in assistant_blocks if isinstance(b, ToolCall)
            ]
            if tool_provider is None:
                raise T0AgentsFatalError(
                    "Agent loop received tool calls from the LLM but no "
                    "`tool_provider` was configured. Pass a connected "
                    "`fastmcp.Client` to "
                    "`StreamingAgentLoop.stream(..., tool_provider=...)` "
                    "if your agent is expected to dispatch tools."
                )
            tool_results = await executor(
                tool_provider,
                tool_call_blocks,
                tool_timeout,
                progress_callback=progress_callback,
            )
            for tc_event, tr in zip(tool_calls_collected, tool_results, strict=True):
                is_error = str(tr.result).startswith("ERROR:")
                yield ToolResultEvent(
                    id=tc_event.id,
                    name=tc_event.name,
                    result=str(tr.result),
                    is_error=is_error,
                )
            messages.append(Message(role="user", content=tool_results))

        logger.warning(
            "Streaming loop exceeded max_turns=%d, episode_id=%s",
            max_turns,
            episode_id,
        )
        raise MaxTurnsExceededError(max_turns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _open_stream(
        self,
        *,
        t0: AsyncTensorZeroGateway,
        function_name: str | None,
        model_name: str | None,
        variant_name: str | None,
        episode_id: str | None,
        messages: list[Message],
        t0_tools: list[dict[str, Any]],
        dryrun: bool,
        tool_choice: ToolChoice | None,
        tags: dict[str, str] | None,
        namespace: str | None,
        system_arguments: dict[str, Any] | None,
        system: str | None,
        params: dict[str, Any] | None,
        cache_options: dict[str, Any] | None,
    ) -> tuple[AsyncIterator[InferenceChunk] | None, TensorZeroError | None]:
        """Issue ``t0.inference(stream=True)`` — return the iterator or the error."""
        input_dict: dict[str, Any] = {"messages": messages}
        if system is not None:
            input_dict["system"] = system
        elif system_arguments is not None:
            input_dict["system"] = system_arguments
        try:
            raw = await t0.inference(
                function_name=function_name or None,
                model_name=model_name,
                variant_name=variant_name,
                episode_id=episode_id,
                input=input_dict,
                additional_tools=t0_tools,
                dryrun=dryrun,
                tool_choice=tool_choice,
                tags=tags,
                namespace=namespace,
                stream=True,
                params=params,
                cache_options=cache_options,
            )
        except TensorZeroError as err:
            return None, err
        return cast(AsyncIterator[InferenceChunk], raw), None

    def _is_recoverable(self, err: TensorZeroError) -> bool:
        status = err.status_code
        text = err.text
        if status not in RECOVERABLE_INFERENCE_STATUS_CODES:
            return False
        if not text:
            return False
        return any(p.search(text) for p in RECOVERABLE_INFERENCE_PATTERNS)

    # ---- Final-chunk tap ---------------------------------------------

    async def _tap_final_chunk(
        self,
        chunks: AsyncIterator[InferenceChunk],
        sink: list[ChatChunk | None],
    ) -> AsyncIterator[ChatChunk]:
        """Forward chat chunks and remember the terminal one for bookkeeping.

        The SDK signals end-of-stream by the final chunk having ``usage``
        populated; the caller owns the single-element ``sink`` into which we
        stash the last observed :class:`ChatChunk`.  Using a caller-provided
        container keeps :class:`StreamingAgentLoop` safe for concurrent
        streams on the same instance.
        """
        async for chunk in chunks:
            if isinstance(chunk, ChatChunk):
                sink[0] = chunk
                yield chunk
            # JsonChunk / other — let the assembler surface an ErrorEvent.

    def _build_assistant_blocks(
        self,
        *,
        assembler: TurnStreamAssembler,
        tool_calls: list[ToolCallEvent],
    ) -> list[Any]:
        """Reconstruct the assistant turn's ``content`` list for history.

        The history the next turn sees must carry ``Text``, ``Thought``, and
        ``ToolCall`` blocks in the same shape ``run_agent`` appends when
        non-streaming — otherwise the LLM sees a different conversation
        history between streaming and non-streaming modes.
        """
        blocks: list[Any] = []
        for _tid, text in assembler.text_blocks.items():
            blocks.append(Text(type="text", text=text))
        for _tid, thought in assembler.thought_blocks.items():
            blocks.append(Thought(type="thought", text=thought))
        for tc in tool_calls:
            blocks.append(
                ToolCall(
                    type="tool_call",
                    id=tc.id,
                    name=tc.name,
                    raw_name=tc.name,
                    arguments=tc.arguments,
                    raw_arguments=(
                        "" if not tc.arguments else _json_dumps(tc.arguments)
                    ),
                )
            )
        return blocks


class _SyntheticChatResponse:
    """Duck-typed mirror of ``ChatInferenceResponse`` for the streaming path.

    Only carries the fields the ``on_inference`` cost-tracker reads
    (``usage``, ``usage.cost``).  ``Agent.invoke`` casts the real SDK return
    value to ``ChatInferenceResponse`` and the cost-tracker never isinstance-
    checks, so this duck-type fits the contract and keeps streaming behaviour
    byte-equivalent for guardrails.
    """

    __slots__ = ("content", "episode_id", "inference_id", "usage")

    def __init__(
        self,
        *,
        content: list[Any],
        episode_id: Any,
        inference_id: Any,
        usage: Any,
    ) -> None:
        self.content = content
        self.episode_id = episode_id
        self.inference_id = inference_id
        self.usage = usage


def _final_text(
    blocks: list[Any],
) -> tuple[str, Literal["text", "thought"]]:
    """Join ``Text`` blocks; fall back to thought content (parity with ``run_agent``).

    When no ``Text`` block carries content, mine the turn's ``Thought``
    blocks so a thought-only final turn never yields an empty response.
    Reasoning models surface the answer via either ``Thought.text`` (raw)
    or ``Thought.summary[*].text`` (provider summary, e.g. gpt-oss
    harmony's analysis channel); both are mined, mirroring the
    non-streaming :func:`t0_agents.mcp.loop.run_agent` fallback. The
    source is flagged ``"thought"`` so consumers branch on provenance,
    not on model name.
    """
    text = "\n".join(b.text for b in blocks if isinstance(b, Text) and b.text)
    if text:
        return text, "text"
    thought_texts: list[str] = []
    for b in blocks:
        if not isinstance(b, Thought):
            continue
        if b.text:
            thought_texts.append(b.text)
        if b.summary:
            thought_texts.extend(s.text for s in b.summary if s.text)
    if thought_texts:
        return "\n".join(thought_texts), "thought"
    return "", "text"


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
