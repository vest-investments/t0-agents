"""Agent loop: inference -> tool calls -> execution -> result -> next inference.

The loop is intentionally async-first and stateless across calls: every invocation
of ``run_agent`` starts a fresh ``messages`` list and only carries continuity
through the optional ``episode_id`` (which TensorZero uses to group inferences
in ClickHouse for analytics and feedback).

Both ``execute_tool_calls`` (sequential, M3.1) and ``execute_tool_calls_parallel``
(concurrent via ``asyncio.gather``, M3.7) share the same per-call helper
``_execute_single_tool_call``, so the error contract — null arguments (EC-2),
``ToolError`` (F5.1), ``McpError`` (F5.2 / timeout F5.4), and any other
``Exception`` — is identical across the two execution strategies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp import McpError
from tensorzero import AsyncTensorZeroGateway, Message, Text, ToolCall, ToolResult
from tensorzero.types import ChatInferenceResponse

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import AgentResult
from t0_agents.mcp.adapter import build_tool_result, convert_tools

logger = logging.getLogger(__name__)


def has_tool_calls(content_blocks: list[Any]) -> bool:
    """Return ``True`` if any content block in ``content_blocks`` is a ``ToolCall``.

    Used by ``run_agent`` to decide whether the current turn requires tool
    execution (continue the loop) or is the terminating text-only response
    (return). Reasoning-model artifacts like ``Thought`` blocks naturally fall
    through as ``False`` here because they aren't ``ToolCall`` instances.
    """
    return any(isinstance(b, ToolCall) for b in content_blocks)


async def _execute_single_tool_call(
    mcp_client: Client[Any], block: ToolCall, timeout: float
) -> ToolResult:
    """Execute one ``ToolCall`` against FastMCP and return one ``ToolResult``.

    Implements the M3.1 error contract:

    * ``name is None`` (LLM hallucinated a tool name not in the schema; T0
      validates it against ``additional_tools`` and zeroes the field) →
      ``ToolResult`` with a ``"ERROR: tool name ... is not in the schema"``
      payload, never invoking ``call_tool``.
    * ``arguments is None`` (EC-2: T0 marks invalid LLM JSON with
      ``arguments=None``, but ``raw_arguments`` always carries the raw string)
      → ``ToolResult`` with an ``"ERROR: invalid JSON arguments..."`` payload
      that the LLM can read on the next turn and (potentially) retry from.
    * ``ToolError`` (F5.1: the tool itself raised) → ``"ERROR: tool execution
      failed: <e>"``.
    * ``McpError`` (F5.2 / F5.4: MCP protocol error, including timeout code
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
    try:
        mcp_result = await mcp_client.call_tool(
            block.name, block.arguments, timeout=timeout
        )
    except ToolError as e:
        logger.error("Tool %r failed (ToolError): %s", name, e)
        return ToolResult(
            name=name,
            result=f"ERROR: tool execution failed: {e}",
            id=block.id,
        )
    except McpError as e:
        logger.error("Tool %r failed (McpError): %s", name, e)
        return ToolResult(
            name=name,
            result=f"ERROR: MCP protocol error (possibly timeout): {e}",
            id=block.id,
        )
    except Exception as e:
        logger.error("Tool %r failed (%s): %s", name, type(e).__name__, e)
        return ToolResult(
            name=name,
            result=f"ERROR: unexpected {type(e).__name__}: {e}",
            id=block.id,
        )
    t0_result = build_tool_result(block, mcp_result)
    logger.debug("Tool %r returned: %.200s", name, t0_result.result)
    return t0_result


async def execute_tool_calls(
    mcp_client: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
    """Execute every ``ToolCall`` in ``content_blocks`` sequentially (M3.1).

    Non-``ToolCall`` blocks (``Text``, ``Thought``, etc.) are skipped silently.
    The output preserves the input order of the ``ToolCall`` blocks.
    """
    return [
        await _execute_single_tool_call(mcp_client, b, timeout)
        for b in content_blocks
        if isinstance(b, ToolCall)
    ]


async def execute_tool_calls_parallel(
    mcp_client: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
    """Execute ``ToolCall`` blocks concurrently via ``asyncio.gather`` (M3.7).

    Useful when an inference response contains multiple ``ToolCall`` blocks and
    the underlying tools are I/O-bound. The returned list preserves the input
    order of the ``ToolCall`` blocks (``asyncio.gather`` returns results in the
    order of its arguments, regardless of completion order).
    """
    tool_calls = [b for b in content_blocks if isinstance(b, ToolCall)]
    if not tool_calls:
        return []
    return list(
        await asyncio.gather(
            *[_execute_single_tool_call(mcp_client, tc, timeout) for tc in tool_calls]
        )
    )


async def run_agent(
    t0: AsyncTensorZeroGateway,
    mcp_client: Client[Any],
    user_message: str,
    function_name: str,
    *,
    episode_id: str | None = None,
    max_turns: int = 20,
    tool_timeout: float = 30.0,
    parallel_tools: bool = False,
    dryrun: bool = False,
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
        mcp_client: A connected ``fastmcp.Client`` (already inside ``async with``).
            Per the FastMCP guide §11.6, the client context **must** span the
            full loop; ``run_agent`` does not wrap it in its own ``async with``.
        user_message: The user's query as a plain string.
        function_name: Name of the chat function defined in ``tensorzero.toml``.
        episode_id: Existing T0 episode id to continue, or ``None`` for a fresh
            episode (T0 will allocate one and we capture it).
        max_turns: Maximum number of inference turns before the loop aborts with
            ``MaxTurnsExceededError`` (anticipates F5.7 / M5.4).
        tool_timeout: Per-tool-call timeout in seconds, forwarded to
            ``Client.call_tool(timeout=...)``. Note: FastMCP's docs flag that
            timeout only applies to foreground execution.
        parallel_tools: If ``True``, dispatch tool calls via ``asyncio.gather``
            (``execute_tool_calls_parallel``); otherwise sequential per M3.1.
        dryrun: If ``True``, forwarded to every ``t0.inference(dryrun=True)``
            call so integration tests don't pollute ClickHouse. Defaults to
            ``False`` for production usage.

    Returns:
        ``AgentResult`` with the joined text of the terminating turn, the
        captured ``episode_id``, the per-turn ``inference_ids``, and the number
        of turns actually used.

    Raises:
        MaxTurnsExceededError: If the loop runs ``max_turns`` times without the
            LLM producing a tool-call-free response.
    """
    mcp_tools = await mcp_client.list_tools()
    t0_tools = convert_tools(mcp_tools)
    messages: list[Message] = [Message(role="user", content=user_message)]
    inference_ids: list[str] = []
    executor = execute_tool_calls_parallel if parallel_tools else execute_tool_calls

    logger.info(
        "Agent loop started: function=%s, max_turns=%d, tools=%d",
        function_name, max_turns, len(t0_tools),
    )

    for turn in range(max_turns):
        raw = await t0.inference(
            function_name=function_name,
            episode_id=episode_id,
            input={"messages": messages},
            additional_tools=t0_tools,
            dryrun=dryrun,
        )
        # Chat-only loop: inference() returns Union[InferenceResponse,
        # AsyncIterator[InferenceChunk]]; we never set stream=True and we never
        # call run_agent against a JSON function, so cast to narrow for mypy.
        # We cannot isinstance-check here because unit tests use duck-typed fakes.
        response = cast(ChatInferenceResponse, raw)

        episode_id = str(response.episode_id)
        inference_ids.append(str(response.inference_id))
        messages.append(Message(role="assistant", content=response.content))

        logger.info(
            "Turn %d: inference_id=%s, episode_id=%s",
            turn + 1, response.inference_id, episode_id,
        )

        if not has_tool_calls(response.content):
            text = "\n".join(
                b.text for b in response.content
                if isinstance(b, Text) and b.text
            )
            logger.info(
                "Agent loop completed: turns=%d, episode_id=%s",
                turn + 1, episode_id,
            )
            return AgentResult(
                response=text,
                episode_id=episode_id,
                inference_ids=inference_ids,
                turns=turn + 1,
            )

        num_tool_calls = sum(
            1 for b in response.content if isinstance(b, ToolCall)
        )
        logger.debug("Turn %d: dispatching %d tool call(s)", turn + 1, num_tool_calls)

        tool_results = await executor(mcp_client, response.content, tool_timeout)
        # F3.5 — N tool results MUST go into a single user message, not N.
        messages.append(Message(role="user", content=tool_results))

    logger.warning(
        "Agent loop exceeded max_turns=%d, episode_id=%s", max_turns, episode_id,
    )
    raise MaxTurnsExceededError(max_turns)
