"""Core types for the guardrails subsystem.

``Guardrail`` (the Protocol) lives in :mod:`t0_agents.core.types` so that
:class:`~t0_agents.core.config.AgentConfig` can reference it without pulling
``guardrails/`` into the ``core/`` layer.  This module only defines the
concrete runtime context and re-exports :class:`Guardrail`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from t0_agents.core.types import Guardrail, GuardrailLogEntry

__all__ = ["Guardrail", "GuardrailContext", "GuardrailLogEntry"]


@dataclass(slots=True)
class GuardrailContext:
    """Mutable per-run state threaded through every guardrail hook.

    A fresh ``GuardrailContext`` is constructed inside :meth:`Agent.invoke` and
    passed to every ``before_run`` / ``before_tool`` / ``after_tool`` /
    ``after_run`` call for that run.  Sharing one context across runs would
    leak counters (``tool_calls_made``) and cost accumulation between them,
    so callers should never construct it directly — ``Agent`` owns the
    lifecycle.

    Attributes:
        agent_name: Mirror of :attr:`AgentConfig.name` for log messages.
        function_name: Mirror of :attr:`AgentConfig.function_name`.
        user_message: The user's original query for this run.
        turn: Incremented by the ``on_inference`` callback after each
            ``t0.inference()`` response.  Starts at ``0``.
        tool_calls_made: Incremented pre-dispatch by
            :func:`~t0_agents.guardrails.limits.max_tool_calls`.  Counts
            attempts, not completions, so the cap is race-safe under
            ``parallel_tools=True``.
        cost_so_far: Cumulative ``response.usage.cost`` across every
            inference in the run.  ``None`` costs are treated as ``0.0``.
        extra: Free-form dict for developer-defined guardrails that need
            their own per-run state.  The framework never reads this.
    """

    agent_name: str
    function_name: str
    user_message: str
    turn: int = 0
    tool_calls_made: int = 0
    cost_so_far: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
