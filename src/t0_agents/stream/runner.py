"""Streaming analogue of :func:`t0_agents.mcp.loop.run_agent`.

Mirrors the existing loop shape — ``list_tools`` → inference → execute tool
calls → feed results back → loop — but uses ``t0.inference(stream=True)``
and yields :class:`AgentEvent` s as chunks arrive.  The non-streaming
:func:`run_agent` is untouched; the contract's ``mcp/``-standalone invariant
(§2.1) holds.

The runner reuses:

* ``convert_tools`` / ``build_tool_result`` from :mod:`t0_agents.mcp.adapter`.
* ``_execute_tool_calls_detailed`` from :mod:`t0_agents.mcp.loop` — the shared
  executor core, which also carries each tool's structured payload.
* every acquisition-independent piece of turn logic from
  :mod:`t0_agents.core.turn_engine` — the same engine :func:`run_agent`
  drives, so the two loops cannot drift apart again.

Memory / guardrail scaffolding stays in :meth:`Agent.invoke` so the runner
itself is agnostic of those subsystems — identical responsibility layering
to :func:`run_agent`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

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

from t0_agents.core.turn_engine import (
    TurnState,
    build_agent_result,
    build_inference_kwargs,
    complete_assistant_turn,
    corrective_retry,
    is_recoverable,
    is_terminal_round,
    mine_final_text,
    raise_max_turns,
    raise_missing_tool_provider,
    seed_messages,
    validate_run_inputs,
)
from t0_agents.mcp.adapter import convert_tools
from t0_agents.mcp.loop import (
    ProgressCallback,
    _execute_tool_calls_detailed,
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
        tool_timeout: float | None = 30.0,
        parallel_tools: bool = False,
        terminal_tools: tuple[str, ...] = (),
        replay_thoughts: bool = False,
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

        ``terminal_tools`` mirrors :func:`~t0_agents.mcp.loop.run_agent`'s
        kwarg of the same name: a round calling only those tools, with no
        errored result and text of its own, ends the stream right after
        its :class:`ToolResultEvent` s with the ordinary terminal
        :class:`DoneEvent` — no further inference, and no synthesised
        text block (the round streamed its own).

        ``replay_thoughts`` likewise mirrors :func:`~t0_agents.mcp.loop.run_agent`'s
        kwarg of the same name: this turn's ``Thought`` blocks stay in
        the replay transcript instead of being stripped before the next
        round's request, for providers that expect their own reasoning
        back while the turn still carries tools.  The streamed
        :class:`~t0_agents.core.events.ThoughtEvent` s are unaffected —
        the flag governs what the next request carries, not what a
        consumer sees.

        ``progress_sink`` receives a :class:`ToolProgressEvent` for every
        server-side progress notification a tool emits while executing.
        A sink (not a yield) because notifications arrive while the
        generator is suspended awaiting the tool executor — there is no
        yield point to ride.  ``Agent`` passes the Run's buffer ``put``
        so progress interleaves into the ordinary event stream.
        """
        validate_run_inputs(
            user_message=user_message,
            function_name=function_name,
            model_name=model_name,
            system=system,
            system_arguments=system_arguments,
            entry_point="StreamingAgentLoop.stream",
        )
        mcp_tools = (
            await tool_provider.list_tools() if tool_provider is not None else []
        )
        t0_tools = convert_tools(mcp_tools)
        known_tool_names = frozenset(t["name"] for t in t0_tools)
        state = TurnState(
            messages=seed_messages(prior_messages, user_message),
            episode_id=episode_id,
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
                t0,
                build_inference_kwargs(
                    function_name=function_name,
                    model_name=model_name,
                    variant_name=variant_name,
                    episode_id=state.episode_id,
                    messages=state.messages,
                    t0_tools=t0_tools,
                    dryrun=dryrun,
                    tool_choice=tool_choice,
                    tags=tags,
                    namespace=namespace,
                    system=system,
                    system_arguments=system_arguments,
                    params=params,
                    cache_options=cache_options,
                ),
            )
            if err is not None:
                if is_recoverable(err):
                    assistant_turn, correction, sentinel = corrective_retry(
                        err.text, t0_tools, turn
                    )
                    yield ErrorEvent(
                        message="recoverable TensorZeroError; retrying",
                        detail={"text": err.text or ""},
                    )
                    state.messages.append(assistant_turn)
                    state.messages.append(correction)
                    state.inference_ids.append(sentinel)
                    yield TurnEndEvent(turn=turn + 1)
                    continue
                raise err

            assert chunks is not None  # recoverable branch returned above
            assembler = TurnStreamAssembler()
            tool_calls_collected: list[ToolCallEvent] = []
            last_chunk_ref: list[ChatChunk | None] = [None]

            tap = self._tap_final_chunk(chunks, last_chunk_ref)
            async for event in assembler.assemble(tap):
                if isinstance(event, ToolCallEvent):
                    tool_calls_collected.append(event)
                yield event

            final_usage_chunk = last_chunk_ref[0]
            assistant_blocks = self._build_assistant_blocks(
                assembler=assembler,
                tool_calls=tool_calls_collected,
                known_tool_names=known_tool_names,
            )

            # The engine drives the post-inference sequence off a synthesised
            # ChatInferenceResponse mirror so guardrails (``max_cost``)
            # accumulate usage exactly as they do for ``run_agent``.  ``None``
            # (a stream that ended without a usage chunk) appends the
            # assistant turn without capturing ids or firing the hook.
            synthetic: ChatInferenceResponse | None = None
            if final_usage_chunk is not None:
                synthetic = cast(
                    ChatInferenceResponse,
                    _SyntheticChatResponse(
                        content=assistant_blocks,
                        episode_id=final_usage_chunk.episode_id,
                        inference_id=final_usage_chunk.inference_id,
                        usage=final_usage_chunk.usage,
                    ),
                )
            await complete_assistant_turn(
                state,
                content_blocks=assistant_blocks,
                response=synthetic,
                on_inference=on_inference,
                replay_thoughts=replay_thoughts,
            )

            logger.info(
                "Stream turn %d: inference_id=%s, episode_id=%s, "
                "blocks=%d, tool_calls=%d",
                turn + 1,
                str(final_usage_chunk.inference_id) if final_usage_chunk else None,
                str(final_usage_chunk.episode_id) if final_usage_chunk else None,
                len(assistant_blocks),
                len(tool_calls_collected),
            )

            yield TurnEndEvent(turn=turn + 1)

            if not tool_calls_collected:
                text, response_source = mine_final_text(assistant_blocks)
                yield DoneEvent(
                    result=build_agent_result(
                        text, response_source, state=state, turns=turn + 1
                    )
                )
                return

            # Dispatch tool calls via the existing executor (reuses the
            # ``_execute_single_tool_call`` error contract verbatim).
            tool_call_blocks = [
                b for b in assistant_blocks if isinstance(b, ToolCall)
            ]
            if tool_provider is None:
                raise_missing_tool_provider(
                    entry_point="StreamingAgentLoop.stream"
                )
            outcomes = await _execute_tool_calls_detailed(
                tool_provider,
                tool_call_blocks,
                tool_timeout,
                parallel=parallel_tools,
                progress_callback=progress_callback,
            )
            tool_results = [outcome.result for outcome in outcomes]
            for tc_event, outcome in zip(
                tool_calls_collected, outcomes, strict=True
            ):
                yield ToolResultEvent(
                    id=tc_event.id,
                    name=tc_event.name,
                    result=str(outcome.result.result),
                    is_error=outcome.is_error,
                    structured=outcome.structured,
                    duration_ms=outcome.duration_ms,
                    server=outcome.server,
                )
            state.messages.append(Message(role="user", content=tool_results))

            text, response_source = mine_final_text(assistant_blocks)
            if is_terminal_round(
                terminal_tools,
                tool_names=[b.name for b in tool_call_blocks],
                errored=[outcome.is_error for outcome in outcomes],
                text=text,
                source=response_source,
            ):
                logger.info(
                    "Stream turn %d completed on a terminal tool round: "
                    "tools=%d, episode_id=%s",
                    turn + 1,
                    len(tool_call_blocks),
                    state.episode_id,
                )
                yield DoneEvent(
                    result=build_agent_result(
                        text, response_source, state=state, turns=turn + 1
                    )
                )
                return

        raise_max_turns(max_turns, state.episode_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _open_stream(
        self,
        t0: AsyncTensorZeroGateway,
        inference_kwargs: dict[str, Any],
    ) -> tuple[AsyncIterator[InferenceChunk] | None, TensorZeroError | None]:
        """Issue ``t0.inference(stream=True)`` — return the iterator or the error.

        ``inference_kwargs`` is the engine-assembled kwargs dict
        (:func:`~t0_agents.core.turn_engine.build_inference_kwargs`);
        this helper only adds ``stream=True`` and folds the gateway
        error into the return value so the caller's generator can yield
        the recoverable-retry events.
        """
        try:
            raw = await t0.inference(**inference_kwargs, stream=True)
        except TensorZeroError as err:
            return None, err
        return cast(AsyncIterator[InferenceChunk], raw), None

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
        known_tool_names: frozenset[str],
    ) -> list[Any]:
        """Reconstruct the assistant turn's full ``content`` list.

        Returns the turn's ``Thought``, ``Text`` and ``ToolCall`` blocks — the
        complete shape read locally by the final-answer / thought fallback
        (:func:`_final_text`) and by the ``on_inference`` cost mirror.

        Reasoning precedes the answer, which is the order the
        non-streaming path receives whole from the gateway.  Every local
        reader partitions by type and is indifferent to it, but a turn
        replaying its reasoning (``replay_thoughts``) puts this list on
        the wire, and the two loops must not describe the same turn
        differently.

        The rebuild mirrors the two zeroings TensorZero applies on the
        non-streaming path but not on chunks: a name outside
        ``known_tool_names`` becomes ``name=None`` and arguments that
        failed JSON parsing become ``arguments=None`` — both route
        dispatch through ``_execute_single_tool_call``'s no-dispatch
        error branches, keeping the two loops' contracts identical.
        Arguments the assembler salvaged (a complete object plus closing
        noise) never appear in ``invalid_tool_args``, so they take the
        ordinary branch: ``arguments`` is the recovered dict and
        ``raw_arguments`` its canonical serialisation — the same bytes
        :func:`~t0_agents.wire.stages.project` puts on ``arguments_json``.
        """
        blocks: list[Any] = []
        for _tid, thought in assembler.thought_blocks.items():
            meta = assembler.thought_metadata.get(_tid, {})
            blocks.append(
                Thought(
                    type="thought",
                    text=thought,
                    signature=meta.get("signature"),
                    provider_type=meta.get("provider_type"),
                )
            )
        for _tid, text in assembler.text_blocks.items():
            blocks.append(Text(type="text", text=text))
        for tc in tool_calls:
            name = tc.name if tc.name in known_tool_names else None
            invalid_raw = assembler.invalid_tool_args.get(tc.id)
            if invalid_raw is not None:
                blocks.append(
                    ToolCall(
                        type="tool_call",
                        id=tc.id,
                        name=name,
                        raw_name=tc.name,
                        arguments=None,
                        raw_arguments=invalid_raw,
                    )
                )
                continue
            blocks.append(
                ToolCall(
                    type="tool_call",
                    id=tc.id,
                    name=name,
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


# Historical import path — the engine owns the helper since v2.2.0.
_final_text = mine_final_text


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
