"""Middleware extension surface for the agent execution chain.

The package surfaces the protocol (:class:`Middleware`), the per-run
:class:`RunContext`, the :func:`build_pipeline` helper, and the six
built-in middlewares that materialise v0.12's hard-coded cross-cutting
concerns.

The default chain materialised by :meth:`AgentConfig.__post_init__`
when ``middleware is None`` is::

    Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal

``middleware[0]`` is outermost; an ``AgentConfig.usage_ledger`` adds
:class:`~t0_agents.usage.UsageLedgerMiddleware` between ``CostTracking``
and ``JsonRetry``.  See :mod:`t0_agents.core.middleware.builtins`
for each layer's responsibility and the load-bearing order invariant
(``SubAgentMiddleware`` wraps ``tool_provider`` inside
``GuardrailsMiddleware``).

Re-exports the public surface so callers write::

    from t0_agents.core.middleware import (
        Middleware, RunContext, MemoryMiddleware, GuardrailsMiddleware,
        CostTrackingMiddleware, SubAgentMiddleware, JsonRetryMiddleware,
        ObservabilityMiddleware, build_pipeline,
    )

or — more commonly — via the package root: ``from t0_agents import ...``.
"""

from __future__ import annotations

from t0_agents.core.middleware.base import (
    Middleware,
    Next,
    OnInferenceCallback,
    RunContext,
    build_pipeline,
)
from t0_agents.core.middleware.builtins import (
    CostTrackingMiddleware,
    GuardrailsMiddleware,
    JsonRetryMiddleware,
    MemoryMiddleware,
    ObservabilityMiddleware,
    SubAgentMiddleware,
)

__all__ = [
    "CostTrackingMiddleware",
    "GuardrailsMiddleware",
    "JsonRetryMiddleware",
    "MemoryMiddleware",
    "Middleware",
    "Next",
    "ObservabilityMiddleware",
    "OnInferenceCallback",
    "RunContext",
    "SubAgentMiddleware",
    "build_pipeline",
]
