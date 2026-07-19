"""Agent loop: inference -> tool calls -> execution -> result -> next inference.

The loop is intentionally async-first and stateless across calls: every invocation
of ``run_agent`` starts a fresh ``messages`` list and only carries continuity
through the optional ``episode_id`` (which TensorZero uses to group inferences
in ClickHouse for analytics and feedback).

Both ``execute_tool_calls`` (sequential) and ``execute_tool_calls_parallel``
(concurrent via ``asyncio.gather``) share the same per-call helper
``_execute_single_tool_call``, so the error contract — null arguments,
``ToolError``, ``McpError`` (including timeouts), and any other
``Exception`` — is identical across the two execution strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp import McpError
from tensorzero import (
    AsyncTensorZeroGateway,
    Message,
    Thought,
    ToolCall,
    ToolChoice,
    ToolResult,
)
from tensorzero.types import ChatInferenceResponse, TensorZeroError

from t0_agents.core.exceptions import T0AgentsFatalError
from t0_agents.core.turn_engine import (
    RECOVERABLE_INFERENCE_PATTERNS,
    RECOVERABLE_INFERENCE_STATUS_CODES,
    TurnState,
    build_agent_result,
    build_inference_kwargs,
    complete_assistant_turn,
    corrective_retry,
    is_recoverable,
    mine_final_text,
    raise_max_turns,
    raise_missing_tool_provider,
    seed_messages,
    strip_thoughts,
    validate_run_inputs,
)
from t0_agents.core.types import AgentResult, ToolProgressHandler
from t0_agents.mcp.adapter import build_tool_result, convert_tools

__all__ = [
    "RECOVERABLE_INFERENCE_PATTERNS",
    "RECOVERABLE_INFERENCE_STATUS_CODES",
    "ProgressCallback",
    "execute_tool_calls",
    "execute_tool_calls_parallel",
    "has_tool_calls",
    "run_agent",
]

# Historical import path — the engine owns the helper since v2.2.0.
_strip_thoughts = strip_thoughts

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, float | None, str | None], Awaitable[None]]
"""Per-tool-call progress relay: ``(tool_call_id, progress, total, message)``.

The loop wraps this per ``ToolCall`` block into a fastmcp
``progress_handler`` closure, so server-side ``ctx.report_progress(...)``
notifications reach the caller instead of dying inside the loop.
``tool_call_id`` is the block's id — the same id the streaming layer puts
on ``ToolCallEvent`` / ``ToolResultEvent``, so notifications correlate.
Kept as a plain callable: ``mcp/`` stays standalone and never imports the
event types built on top of it.
"""


def _progress_handler_for(
    block_id: str, progress_callback: ProgressCallback
) -> ToolProgressHandler:
    """Bind ``progress_callback`` to one tool-call id, fastmcp-handler shaped."""

    async def _forward(
        progress: float, total: float | None, message: str | None
    ) -> None:
        await progress_callback(block_id, progress, total, message)

    return _forward


def has_tool_calls(content_blocks: list[Any]) -> bool:
    """Return ``True`` if any content block in ``content_blocks`` is a ``ToolCall``.

    Used by ``run_agent`` to decide whether the current turn requires tool
    execution (continue the loop) or is the terminating text-only response
    (return). Reasoning-model artifacts like ``Thought`` blocks naturally fall
    through as ``False`` here because they aren't ``ToolCall`` instances.
    """
    return any(isinstance(b, ToolCall) for b in content_blocks)


async def _execute_single_tool_call(
    tool_provider: Client[Any],
    block: ToolCall,
    timeout: float | None,
    progress_callback: ProgressCallback | None = None,
) -> ToolResult:
    """Execute one ``ToolCall`` against FastMCP and return one ``ToolResult``.

    Implements the error contract:

    * ``name is None`` (LLM hallucinated a tool name not in the schema; T0
      validates it against ``additional_tools`` and zeroes the field) →
      ``ToolResult`` with a ``"ERROR: tool name ... is not in the schema"``
      payload, never invoking ``call_tool``.
    * ``arguments is None`` (T0 marks invalid LLM JSON with
      ``arguments=None``, but ``raw_arguments`` always carries the raw string)
      → ``ToolResult`` with an ``"ERROR: invalid JSON arguments..."`` payload
      that the LLM can read on the next turn and (potentially) retry from.
    * ``ToolError`` (the tool itself raised) → ``"ERROR: tool execution
      failed: <e>"``.
    * ``McpError`` (MCP protocol error, including timeout code
      ``-32000``) → ``"ERROR: MCP protocol error (possibly timeout): <e>"``.
    * Any other ``Exception`` → ``"ERROR: unexpected <Type>: <e>"``. Belt-and-
      braces so a single misbehaving tool can't crash the agent loop.

    The error string is always carried on the ``ToolResult.result`` field, which
    is what the LLM will see in the next inference's user message.

    The ``name`` on the returned ``ToolResult`` falls back to ``block.raw_name``
    when ``block.name`` is ``None``. ``ToolResult.name`` must be ``str`` per
    ``tensorzero/types.py:186``; this keeps mypy strict happy without an
    ``assert`` while still surfacing the raw model intent to the LLM.
    """
    name = block.name or block.raw_name
    if block.name is None:
        logger.error("Tool %r: name not in schema", block.raw_name)
        return ToolResult(
            name=name,
            result=f"ERROR: tool name {block.raw_name!r} is not in the schema",
            id=block.id,
        )
    if block.arguments is None:
        logger.error("Tool %r: invalid JSON arguments: %s", name, block.raw_arguments)
        return ToolResult(
            name=name,
            result=f"ERROR: invalid JSON arguments from LLM: {block.raw_arguments}",
            id=block.id,
        )
    logger.debug("Executing tool %r with args: %s", name, block.arguments)
    start = time.monotonic()
    try:
        # The kwarg is only passed when progress is actually wired, so
        # providers predating ``progress_handler`` keep working on every
        # progress-free path (run_agent, streaming without a sink).
        if progress_callback is None:
            mcp_result = await tool_provider.call_tool(
                block.name, block.arguments, timeout=timeout
            )
        else:
            mcp_result = await tool_provider.call_tool(
                block.name,
                block.arguments,
                timeout=timeout,
                progress_handler=_progress_handler_for(block.id, progress_callback),
            )
    except T0AgentsFatalError:
        # Guardrail-level hard block (or any future fatal marker) must
        # propagate past the belt-and-braces ``except Exception`` below,
        # so ``Agent.invoke()`` surfaces the policy violation verbatim
        # rather than stringifying it into a ToolResult.
        raise
    except ToolError as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("Tool %r failed (ToolError): %s", name, e)
        logger.debug("Tool %r completed in %.1fms", name, elapsed_ms)
        return ToolResult(
            name=name,
            result=f"ERROR: tool execution failed: {e}",
            id=block.id,
        )
    except McpError as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("Tool %r failed (McpError): %s", name, e)
        logger.debug("Tool %r completed in %.1fms", name, elapsed_ms)
        return ToolResult(
            name=name,
            result=f"ERROR: MCP protocol error (possibly timeout): {e}",
            id=block.id,
        )
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("Tool %r failed (%s): %s", name, type(e).__name__, e)
        logger.debug("Tool %r completed in %.1fms", name, elapsed_ms)
        return ToolResult(
            name=name,
            result=f"ERROR: unexpected {type(e).__name__}: {e}",
            id=block.id,
        )
    elapsed_ms = (time.monotonic() - start) * 1000
    t0_result = build_tool_result(block, mcp_result)
    logger.debug("Tool %r returned: %.200s", name, t0_result.result)
    logger.debug("Tool %r completed in %.1fms", name, elapsed_ms)
    return t0_result


async def execute_tool_calls(
    tool_provider: Client[Any],
    content_blocks: list[Any],
    timeout: float | None = 30.0,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[ToolResult]:
    """Execute every ``ToolCall`` in ``content_blocks`` sequentially.

    Non-``ToolCall`` blocks (``Text``, ``Thought``, etc.) are skipped silently.
    The output preserves the input order of the ``ToolCall`` blocks.
    ``progress_callback`` (when given) receives every server-side progress
    notification, tagged with the emitting block's id.
    """
    return [
        await _execute_single_tool_call(tool_provider, b, timeout, progress_callback)
        for b in content_blocks
        if isinstance(b, ToolCall)
    ]


async def execute_tool_calls_parallel(
    tool_provider: Client[Any],
    content_blocks: list[Any],
    timeout: float | None = 30.0,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[ToolResult]:
    """Execute ``ToolCall`` blocks concurrently via ``asyncio.gather``.

    Useful when an inference response contains multiple ``ToolCall`` blocks and
    the underlying tools are I/O-bound. The returned list preserves the input
    order of the ``ToolCall`` blocks (``asyncio.gather`` returns results in the
    order of its arguments, regardless of completion order).
    ``progress_callback`` notifications from concurrent tools interleave; the
    block id tags each one.
    """
    tool_calls = [b for b in content_blocks if isinstance(b, ToolCall)]
    if not tool_calls:
        return []
    return list(
        await asyncio.gather(
            *[
                _execute_single_tool_call(tool_provider, tc, timeout, progress_callback)
                for tc in tool_calls
            ]
        )
    )


async def run_agent(
    t0: AsyncTensorZeroGateway,
    tool_provider: Client[Any] | None = None,
    user_message: str = "",
    function_name: str | None = "",
    *,
    episode_id: str | None = None,
    max_turns: int = 20,
    tool_timeout: float | None = 30.0,
    parallel_tools: bool = False,
    dryrun: bool = False,
    tool_choice: ToolChoice | None = None,
    tags: dict[str, str] | None = None,
    namespace: str | None = None,
    system_arguments: dict[str, Any] | None = None,
    system: str | None = None,
    on_inference: Callable[[ChatInferenceResponse], Awaitable[None]] | None = None,
    prior_messages: list[Message] | None = None,
    model_name: str | None = None,
    variant_name: str | None = None,
    params: dict[str, Any] | None = None,
    cache_options: dict[str, Any] | None = None,
) -> AgentResult:
    """Run the agent loop end-to-end and return the final response.

    The loop is::

        discover tools  →  for turn in range(max_turns):
                              inference(messages, additional_tools)
                              capture episode_id + inference_id
                              append assistant message to history
                              if no tool calls → return final text
                              else execute tools → append ONE user message

    Args:
        t0: A connected ``AsyncTensorZeroGateway`` (already inside ``async with``).
        tool_provider: Optional connected ``fastmcp.Client`` (already inside
            ``async with``). When ``None`` (the default), no tools are
            advertised to the LLM — the loop runs a single inference and
            returns the resulting text without ever calling ``list_tools``.
            When provided, the client context **must** span the full loop;
            ``run_agent`` does not wrap it in its own ``async with``, and it
            is forwarded to ``execute_tool_calls`` /
            ``execute_tool_calls_parallel`` on every tool-dispatch turn.
        user_message: The user's query as a plain string.
        function_name: Name of the chat function defined in ``tensorzero.toml``.
            Empty/``None`` is coalesced to ``None`` at the gateway call so a
            ``model_name``-routed loop omits the field; exactly one of
            ``function_name``/``model_name`` must be non-empty (the gateway
            rejects both/neither).
        episode_id: Existing T0 episode id to continue, or ``None`` for a fresh
            episode (T0 will allocate one and we capture it).
        max_turns: Maximum number of inference turns before the loop aborts with
            ``MaxTurnsExceededError``.
        tool_timeout: Per-tool-call timeout in seconds, forwarded to
            ``Client.call_tool(timeout=...)``. ``None`` disables the
            per-call timeout — the call waits as long as the transport
            allows, which is what elicitation-capable tools need while a
            human deliberates. Note: FastMCP's docs flag that timeout only
            applies to foreground execution.
        parallel_tools: If ``True``, dispatch tool calls via ``asyncio.gather``
            (``execute_tool_calls_parallel``); otherwise sequential.
        dryrun: If ``True``, forwarded to every ``t0.inference(dryrun=True)``
            call so integration tests don't pollute ClickHouse. Defaults to
            ``False`` for production usage.

            **Note:** when ``dryrun=True``, inference IDs are generated but
            never stored in ClickHouse. Any subsequent ``t0.feedback()`` call
            targeting those IDs will fail with HTTP 400 (``"Episode ID: ...
            does not exist"``). Callers that send feedback after the loop must
            use ``dryrun=False`` for the inference pass.
        tool_choice: Optional tool-choice directive forwarded to every
            ``t0.inference()`` call. Accepts ``"auto"``, ``"required"``,
            ``"off"``, or ``{"specific": "<tool_name>"}`` to force a particular
            tool. ``None`` (the default) lets the model decide freely.
        tags: Free-form ``Dict[str, str]`` forwarded to every
            ``t0.inference()`` call for observability. ``None`` (the default)
            omits the parameter.
        namespace: TensorZero namespace forwarded to every ``t0.inference()``
            call for experimentation routing. Falls back to the function's
            default namespace when ``None``.
        system_arguments: Template arguments forwarded to TensorZero as
            ``input.system`` on every ``t0.inference()`` call.  When the
            function declares ``schemas.system.path``, the dict must
            match that schema directly — its keys are the
            schema-described properties.  Used by the memory subsystem
            to pass ``memory_context`` to system-template variants.
            ``None`` (the default) omits the ``system`` key from the
            input dict entirely, preserving existing behaviour.
        system: Raw system prompt forwarded to TensorZero as
            ``input.system`` (string form).  Only valid when the
            TensorZero function does **not** declare a
            ``schemas.system.path`` — passing a string to a templated
            function returns HTTP 400 from the gateway.  Mutually
            exclusive with ``system_arguments``: a ``ValueError`` is
            raised when both are set.  Use this for prompt-as-a-service
            style flows where the prompt body is itself runtime data;
            prefer ``system_arguments`` (with a configured template) for
            every other case so the dataset captured in ClickHouse stays
            structured.  ``None`` (the default) preserves existing
            behaviour.
        on_inference: Optional async callback invoked with every
            ``ChatInferenceResponse`` after it is received and cast,
            BEFORE the turn's tool calls are dispatched. Used by the
            guardrails subsystem to update ``GuardrailContext.cost_so_far``
            from ``response.usage.cost`` so ``max_cost`` can abort the
            run at the next tool call. ``None`` (the default) is a no-op
            that preserves today's behaviour byte-for-byte.
        prior_messages: Optional TensorZero-native message history to
            seed the loop with.  Used by the ``messages`` memory
            strategy (``Agent`` with ``memory_strategy="messages"`` and a
            provider that exposes ``load_history``) to thread a prior
            conversation through ``input.messages`` without
            stringifying it into the user prompt.  When ``None`` (the
            default) or empty, the loop starts with a single
            ``Message(role="user", content=user_message)``, byte-
            identical to the pre-``prior_messages`` behaviour.
        model_name: Route every inference through the gateway's built-in
            chat passthrough (``tensorzero::default``) instead of a
            declared function — either a ``{provider}::{model}``
            shorthand or a ``[models.*]`` name.  Tools, episodes, and
            tags all work unchanged on this path.
        variant_name: Pin a specific variant of ``function_name``.  The
            gateway rejects it alongside ``model_name``.
        params: Per-call inference parameters in the TensorZero shape,
            e.g. ``{"chat_completion": {"temperature": 0.2}}``,
            forwarded to every ``t0.inference()`` call.
        cache_options: TensorZero cache options
            (``{"enabled": ..., "max_age_s": ...}``) forwarded to every
            ``t0.inference()`` call.

    Returns:
        ``AgentResult`` with the joined text of the terminating turn, the
        captured ``episode_id``, the per-turn ``inference_ids``, the number
        of turns actually used, and ``response_source`` indicating whether
        the response text came from ``Text`` or ``Thought`` blocks.

    Raises:
        MaxTurnsExceededError: If the loop runs ``max_turns`` times without the
            LLM producing a tool-call-free response.
        ValueError: If both ``system`` and ``system_arguments`` are set,
            or if ``user_message`` / ``function_name`` are left at their
            empty-string defaults (those defaults exist only as a
            Python-syntax workaround for the optional ``tool_provider``).
    """
    validate_run_inputs(
        user_message=user_message,
        function_name=function_name,
        model_name=model_name,
        system=system,
        system_arguments=system_arguments,
        entry_point="run_agent",
    )

    mcp_tools = await tool_provider.list_tools() if tool_provider is not None else []
    t0_tools = convert_tools(mcp_tools)
    state = TurnState(
        messages=seed_messages(prior_messages, user_message),
        episode_id=episode_id,
    )
    executor = execute_tool_calls_parallel if parallel_tools else execute_tool_calls

    logger.info(
        "Agent loop started: target=%s, max_turns=%d, tools=%d",
        function_name or model_name, max_turns, len(t0_tools),
    )

    for turn in range(max_turns):
        # Catch recoverable TensorZeroError (e.g. malformed tool name
        # from <|channel|> artifacts) and inject a corrective user message so
        # the LLM can retry on the next turn.
        try:
            raw = await t0.inference(
                **build_inference_kwargs(
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
                )
            )
        except TensorZeroError as e:
            if is_recoverable(e):
                assistant_turn, correction, sentinel = corrective_retry(
                    e.text, t0_tools, turn
                )
                state.messages.append(assistant_turn)
                state.messages.append(correction)
                state.inference_ids.append(sentinel)
                continue
            raise
        # Chat-only loop: inference() returns Union[InferenceResponse,
        # AsyncIterator[InferenceChunk]]; we never set stream=True and we never
        # call run_agent against a JSON function, so cast to narrow for mypy.
        # We cannot isinstance-check here because unit tests use duck-typed fakes.
        response = cast(ChatInferenceResponse, raw)

        await complete_assistant_turn(
            state,
            content_blocks=response.content,
            response=response,
            on_inference=on_inference,
        )

        # Content-type breakdown per turn.
        content_types = [type(b).__name__ for b in response.content]
        logger.info(
            "Turn %d: inference_id=%s, episode_id=%s, content_types=%s",
            turn + 1, response.inference_id, state.episode_id, content_types,
        )

        if not has_tool_calls(response.content):
            # Log Thought content at DEBUG when present.
            for b in response.content:
                if isinstance(b, Thought) and b.text:
                    logger.debug("Thought content: %.500s", b.text)

            text, response_source = mine_final_text(response.content)
            if response_source == "thought":
                logger.info(
                    "Turn %d: thought-only response used as fallback",
                    turn + 1,
                )
            logger.info(
                "Agent loop completed: turns=%d, episode_id=%s",
                turn + 1, state.episode_id,
            )
            return build_agent_result(
                text, response_source, state=state, turns=turn + 1
            )

        # Log tool names in dispatch.
        tool_calls_in_turn = [
            b for b in response.content if isinstance(b, ToolCall)
        ]
        tool_names = [tc.name or tc.raw_name for tc in tool_calls_in_turn]
        logger.debug(
            "Turn %d: dispatching %d tool call(s): %s",
            turn + 1, len(tool_calls_in_turn), tool_names,
        )

        if tool_provider is None:
            raise_missing_tool_provider(entry_point="run_agent")
        tool_results = await executor(tool_provider, response.content, tool_timeout)
        # N tool results MUST go into a single user message, not N.
        state.messages.append(Message(role="user", content=tool_results))

    raise_max_turns(max_turns, state.episode_id)
