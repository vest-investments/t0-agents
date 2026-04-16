"""Validator-style guardrails: ``block_tools``.

Defense-in-depth against the LLM invoking tools that should never run —
e.g. deleted-but-still-advertised legacy tools, or dangerous destructive
operations gated elsewhere.  Unlike :func:`~t0_agents.guardrails.limits.max_cost`
these are SOFT blocks: the tool call short-circuits with an error string to
the LLM, which can then try something else.

Asymmetry (documented in ``docs/guardrails.md``): ``block_tools`` does NOT
filter :py:meth:`ToolProvider.list_tools` output.  The LLM still sees the
tool and may attempt to call it; enforcement is at ``before_tool``.  This
keeps the wrapper's ``list_tools`` byte-identical to the inner provider
and preserves the Phase A contract §9.5 no-op equivalence when the block
list does not match any attempted call.
"""

from __future__ import annotations

from typing import Any

from t0_agents.core.types import Guardrail
from t0_agents.guardrails.base import GuardrailContext

__all__ = ["block_tools"]


def block_tools(names: list[str]) -> Guardrail:
    """Soft-block tool invocations whose name appears in *names*.

    Returns an error string the LLM can read on the next turn; does not
    abort the run.  For a hard stop, pair with :func:`max_tool_calls` or
    lift to a custom guardrail that raises :class:`GuardrailBlockedError`.
    """
    blocked = frozenset(names)

    class _BlockTools:
        name = f"block_tools({sorted(blocked)!r})"

        async def before_run(self, ctx: GuardrailContext) -> None:
            return None

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            if tool_call.name in blocked:
                return f"ERROR: tool {tool_call.name!r} is blocked by policy"
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            return None

    return _BlockTools()
