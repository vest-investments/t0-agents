"""``require_approval`` — interactive approval gate over the FastMCP elicitation signature.

Rides the same ``(message, response_type, params, context) -> ElicitResult``
signature as FastMCP's ``elicitation_handler`` so the developer can plug in
the exact same callable they pass to ``Client(elicitation_handler=...)``.
No parallel HITL channel is introduced (contract §2.5).

Implementation note (why a handler kwarg).  ``fastmcp.Client`` stores its
elicitation handler in ``_session_kwargs["elicitation_callback"]`` with no
public getter.  The wrapper runs client-side and has no access to that
internal handler, so the guardrail takes its own reference at construction.
Documented as divergence D1 of the Phase C implementation log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.client.elicitation import ElicitResult
from mcp.types import ElicitRequestFormParams

from t0_agents.core.types import Guardrail
from t0_agents.guardrails.base import GuardrailContext

logger = logging.getLogger(__name__)

__all__ = ["ApprovalHandler", "require_approval"]


ApprovalHandler = Callable[
    [str, type[Any] | None, Any, Any],
    Awaitable[Any],
]
"""Structural alias for FastMCP's ``ElicitationHandler`` signature.

Returns either an ``ElicitResult`` or a raw value (the guardrail normalises
raw values into ``ElicitResult(action='accept', content=value)`` the same
way ``fastmcp.client.elicitation.create_elicitation_callback`` does).
"""


def require_approval(
    *,
    tools: list[str],
    handler: ApprovalHandler | None = None,
    prompt_template: str = "Approve calling {tool_name} with {arguments}?",
) -> Guardrail:
    """Return a :class:`Guardrail` that asks the user before running *tools*.

    The guardrail's ``before_tool`` hook:

    1. Passes through when ``tool_call.name not in tools``.
    2. When gated and ``handler is None`` → returns a soft-block error
       string to the LLM and logs a WARNING (once per run).
    3. When gated and a handler is configured → invokes
       ``handler(message, None, params, None)`` with an empty-schema
       :class:`~mcp.types.ElicitRequestFormParams` (binary approval).
       Accepts the call only when the normalised :class:`ElicitResult` has
       ``action == "accept"``; any other action — ``decline``, ``cancel``,
       raw decline dict, or an exception raised by the handler — soft-blocks.

    Args:
        tools: Exact tool names whose calls must be confirmed by the user.
        handler: ``ElicitationHandler``-shaped callable.  Pass the same
            handler wired on your ``fastmcp.Client`` for a unified UX.
            ``None`` disables approval (soft-block with a clear error).
        prompt_template: Message template rendered for the handler.
            ``{tool_name}`` and ``{arguments}`` are available as
            placeholders.
    """

    tools_set = frozenset(tools)

    class _RequireApproval:
        name = f"require_approval(tools={sorted(tools_set)!r})"
        _warned_missing_handler: bool = False

        async def before_run(self, ctx: GuardrailContext) -> None:
            return None

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            tool_name = tool_call.name
            if tool_name not in tools_set:
                return None

            if handler is None:
                if not self._warned_missing_handler:
                    logger.warning(
                        "require_approval guardrail has no handler configured; "
                        "tool %r will be soft-blocked for this run.",
                        tool_name,
                    )
                    self._warned_missing_handler = True
                return (
                    f"ERROR: approval required for {tool_name!r} but no "
                    f"handler is configured"
                )

            message = prompt_template.format(
                tool_name=tool_name,
                arguments=tool_call.arguments,
            )
            params = ElicitRequestFormParams(
                message=message,
                requestedSchema={"type": "object", "properties": {}},
            )

            try:
                raw = await handler(message, None, params, None)
            except Exception as e:
                logger.warning(
                    "require_approval handler raised for tool %r: %s",
                    tool_name, e,
                )
                return (
                    f"ERROR: approval handler failed for {tool_name!r}: {e}"
                )

            result = (
                raw
                if isinstance(raw, ElicitResult)
                else ElicitResult(action="accept", content=raw)
            )
            if result.action != "accept":
                return f"ERROR: user declined to run {tool_name!r}"
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            return None

    return _RequireApproval()
