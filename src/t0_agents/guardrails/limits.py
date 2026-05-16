"""Budget-style guardrails: ``max_cost``, ``max_tool_calls``.

Both raise :class:`GuardrailBlockedError` — a ``T0AgentsFatalError`` —
from ``before_tool`` when a limit is crossed.  ``_execute_single_tool_call``
re-raises ``T0AgentsFatalError`` past its broad ``except Exception`` belt-
and-braces trap so the run aborts cleanly.

Cost tracking is populated by :meth:`Agent.invoke`'s ``on_inference`` callback,
which reads ``response.usage.cost`` after every ``t0.inference()`` response
and mutates :attr:`GuardrailContext.cost_so_far` BEFORE any tool call is
dispatched.  That ordering lets the ``before_tool`` check abort a run the
moment the previous turn's inference pushed cumulative cost past the cap.
"""

from __future__ import annotations

from typing import Any

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import Guardrail
from t0_agents.guardrails.base import GuardrailContext

__all__ = ["max_cost", "max_tool_calls"]


def max_cost(*, per_episode: float, currency: str = "USD") -> Guardrail:
    """Abort the run when cumulative cost crosses *per_episode*.

    The cap is checked in ``before_tool``; ``cost_so_far`` is updated by the
    Agent's ``on_inference`` callback BEFORE tools run, so the limit can
    fire on the very next tool call after a budget-breaking inference.

    Caveat.  TensorZero returns cost post-call (after the gateway sees the
    provider response), so the guardrail is not a predictive budget — it
    catches overruns at the next turn boundary rather than preventing them
    from happening in the first place.

    Args:
        per_episode: Cumulative cost ceiling, in *currency*, for a single
            ``Agent.invoke()`` invocation.
        currency: Label used in the error message.  Purely cosmetic — no
            FX conversion is performed.
    """
    if per_episode <= 0:
        raise ValueError(
            f"max_cost(per_episode={per_episode}) must be positive"
        )

    class _MaxCost:
        name = f"max_cost({per_episode} {currency})"

        async def before_run(self, ctx: GuardrailContext) -> None:
            return None

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            if ctx.cost_so_far >= per_episode:
                raise GuardrailBlockedError(
                    f"Cost limit {per_episode} {currency} exceeded "
                    f"(spent {ctx.cost_so_far:.4f} {currency})"
                )
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            return None

    return _MaxCost()


def max_tool_calls(*, per_run: int) -> Guardrail:
    """Cap total tool calls per ``Agent.invoke()``.

    Increments :attr:`GuardrailContext.tool_calls_made` at dispatch time
    (inside ``before_tool``), not after completion.  This makes the counter
    race-safe under ``parallel_tools=True``: asyncio coroutines run to their
    first await point without yielding, so the ``+=`` completes before the
    next concurrent call enters ``before_tool``.

    Args:
        per_run: Maximum number of ``call_tool`` invocations allowed in a
            single run.  Calls attempted beyond the cap raise
            :class:`GuardrailBlockedError`.  ``0`` is a valid "no tool
            calls allowed" configuration (any tool dispatch aborts the
            run); negative values are rejected.
    """
    if per_run < 0:
        raise ValueError(
            f"max_tool_calls(per_run={per_run}) must be non-negative"
        )

    class _MaxToolCalls:
        name = f"max_tool_calls({per_run})"

        async def before_run(self, ctx: GuardrailContext) -> None:
            return None

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            if ctx.tool_calls_made >= per_run:
                raise GuardrailBlockedError(
                    f"Tool-call cap {per_run} hit "
                    f"(tool {tool_call.name!r} would have been #{ctx.tool_calls_made + 1})"
                )
            ctx.tool_calls_made += 1
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            return None

    return _MaxToolCalls()
