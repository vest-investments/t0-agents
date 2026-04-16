"""Guardrail subsystem: declarative policy layer on top of ``Agent``.

Exports the :class:`Guardrail` Protocol (from ``core/types.py`` to avoid a
circular import), the per-run :class:`GuardrailContext`, the audit-log
entry type :class:`GuardrailLogEntry`, and the four built-in factories:

- :func:`require_approval` — interactive approval gate.
- :func:`max_cost` — cumulative cost ceiling per run.
- :func:`max_tool_calls` — tool-call cap per run.
- :func:`block_tools` — hard-deny specific tool names.

Write-your-own: any object that structurally satisfies the
``Guardrail`` Protocol plugs in via ``AgentConfig(guardrails=[...])``.
"""

from t0_agents.core.types import Guardrail, GuardrailLogEntry
from t0_agents.guardrails.approval import ApprovalHandler, require_approval
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.limits import max_cost, max_tool_calls
from t0_agents.guardrails.runtime import GuardedToolProvider
from t0_agents.guardrails.validators import block_tools

__all__ = [
    "ApprovalHandler",
    "GuardedToolProvider",
    "Guardrail",
    "GuardrailContext",
    "GuardrailLogEntry",
    "block_tools",
    "max_cost",
    "max_tool_calls",
    "require_approval",
]
