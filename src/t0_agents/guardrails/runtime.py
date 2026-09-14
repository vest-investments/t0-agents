"""Guardrail runtime: the ``GuardedToolProvider`` wrapper.

When :class:`AgentConfig.guardrails` is non-empty, :meth:`Agent.invoke`
constructs a :class:`GuardedToolProvider` around the real ``tool_provider``
and forwards it to ``run_agent``.  The wrapper:

- Proxies :py:meth:`list_tools` unchanged so the LLM still sees every tool.
  ``block_tools`` is a call-time gate (defense-in-depth), not a listing
  filter — the asymmetry is documented in ``docs/guardrails.md``.
- Fires every registered guardrail's ``before_tool`` hook BEFORE the inner
  ``call_tool``.  A hook may:
    * return ``None`` → pass through;
    * return ``str`` → short-circuit with an error result;
    * raise :class:`GuardrailBlockedError` → hard-abort the run (propagated
      by ``_execute_single_tool_call``'s ``T0AgentsFatalError`` branch).
  Non-fatal exceptions from user-supplied guardrails are logged and
  treated as pass so a buggy hook cannot crash the loop.
- Fires every ``after_tool`` hook once the inner call returns (or once the
  short-circuit has been prepared).  ``after_tool`` sees the serialized
  result string — identical to what the LLM will see next turn — so
  observability hooks and validators can inspect the payload uniformly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from tensorzero import ToolCall

from t0_agents.core.exceptions import T0AgentsFatalError
from t0_agents.core.types import (
    Guardrail,
    GuardrailLogEntry,
    SupportsServerAttribution,
    ToolProgressHandler,
    ToolProvider,
)
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.mcp.adapter import serialize_tool_result

logger = logging.getLogger(__name__)

__all__ = ["GuardedToolProvider"]


class GuardedToolProvider:
    """Guardrail-aware ``ToolProvider`` wrapper.

    Structurally satisfies :class:`~t0_agents.core.types.ToolProvider` —
    ``list_tools`` and ``call_tool`` are forwarded (with hook interception on
    ``call_tool``).  ``server_for`` is forwarded too: the loop probes the
    *outermost* provider for
    :class:`~t0_agents.core.types.SupportsServerAttribution`, so a wrapper that
    omits it erases attribution for everything beneath it.

    The wrapper holds a reference to the per-run :class:`GuardrailContext`
    and to the shared ``log_sink`` list that :meth:`Agent.invoke` attaches to
    :attr:`AgentResult.guardrail_log` after the run completes.
    """

    def __init__(
        self,
        inner: ToolProvider,
        guardrails: list[Guardrail],
        ctx: GuardrailContext,
        log_sink: list[GuardrailLogEntry],
    ) -> None:
        self._inner = inner
        self._guardrails = guardrails
        self._ctx = ctx
        self._log = log_sink

    async def list_tools(self) -> list[Any]:
        return await self._inner.list_tools()

    def server_for(self, name: str) -> str | None:
        """Delegate server attribution to ``inner`` when it can attribute one."""
        if isinstance(self._inner, SupportsServerAttribution):
            return self._inner.server_for(name)
        return None

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
        progress_handler: ToolProgressHandler | None = None,
    ) -> Any:
        tool_call = _synthetic_tool_call(name, arguments)

        for guardrail in self._guardrails:
            try:
                short_circuit = await guardrail.before_tool(self._ctx, tool_call)
            except T0AgentsFatalError:
                raise
            except Exception:
                logger.exception(
                    "Guardrail %r before_tool raised a non-fatal exception; "
                    "treating as pass",
                    guardrail.name,
                )
                short_circuit = None

            if short_circuit is not None:
                detail = short_circuit
                self._log.append(
                    GuardrailLogEntry(
                        guardrail_name=guardrail.name,
                        hook="before_tool",
                        kind="rejection",
                        tool_name=name,
                        detail=detail,
                    )
                )
                blocked = _blocked_result(detail)
                await self._fire_after_tool(
                    tool_call, serialize_tool_result(blocked)
                )
                return blocked

        self._log.append(
            GuardrailLogEntry(
                guardrail_name="__chain__",
                hook="before_tool",
                kind="pass",
                tool_name=name,
            )
        )

        result = await self._inner.call_tool(
            name, arguments, timeout=timeout, progress_handler=progress_handler
        )
        result_str = serialize_tool_result(result)
        await self._fire_after_tool(tool_call, result_str)
        return result

    async def _fire_after_tool(
        self, tool_call: ToolCall, result_str: str
    ) -> None:
        for guardrail in self._guardrails:
            try:
                await guardrail.after_tool(self._ctx, tool_call, result_str)
            except T0AgentsFatalError:
                raise
            except Exception:
                logger.exception(
                    "Guardrail %r after_tool raised a non-fatal exception; "
                    "continuing",
                    guardrail.name,
                )
            else:
                self._log.append(
                    GuardrailLogEntry(
                        guardrail_name=guardrail.name,
                        hook="after_tool",
                        kind="pass",
                        tool_name=tool_call.name,
                    )
                )


def _synthetic_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    """Build a ``ToolCall`` for hook introspection.

    The loop never sends this instance to TensorZero — it is only passed to
    guardrail hooks so they can inspect ``name`` and ``arguments``.  The
    ``id`` is a fixed sentinel; ``raw_*`` fields mirror the resolved values
    so hooks that log them see consistent content.
    """
    return ToolCall(
        id="guardrail-synthetic",
        raw_name=name,
        raw_arguments=json.dumps(arguments, default=str, ensure_ascii=False),
        name=name,
        arguments=arguments,
    )


def _blocked_result(detail: str) -> CallToolResult:
    """Shape a :class:`CallToolResult` for ``serialize_tool_result``.

    ``serialize_tool_result`` prepends ``"ERROR: "`` when ``is_error`` is
    ``True``; this helper strips any caller-supplied ``"ERROR: "`` prefix
    from *detail* so the final string has exactly one prefix.
    """
    text = detail.removeprefix("ERROR: ").strip() or "blocked by policy"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=None,
        meta=None,
        data=None,
        is_error=True,
    )
