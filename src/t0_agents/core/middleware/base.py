"""Middleware protocol, :class:`RunContext`, and pipeline composition.

The middleware chain is the v1.0 extension surface for cross-cutting
concerns.  Memory, guardrails, cost tracking, sub-agent orchestration,
and JSON retry — all hard-coded inline in v0.12's ``core/agent.py`` —
become composable middlewares that wrap a single terminal callable.

A middleware is a Starlette-style coroutine::

    async def my_middleware(ctx: RunContext, next_: Next) -> AgentResult:
        # ... setup ...
        result = await next_(ctx)
        # ... teardown ...
        return result

:func:`build_pipeline` folds a list of middlewares around a terminal
producing an ``async (ctx) -> AgentResult`` callable.  ``middleware[0]``
is **outermost** — it sees the run first on the way in and last on the
way out.

The terminal is selected by :class:`~t0_agents.core.agent.Agent` per
``(output.expects_tool_loop, is_streaming)`` and corresponds to the
inner inference loop (``run_agent`` for chat,
``StreamingAgentLoop.stream`` for streaming chat,
``Agent._single_shot_terminal`` for every non-chat
:class:`~t0_agents.core.output.OutputMode`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from t0_agents.core.exceptions import T0AgentsError

if TYPE_CHECKING:
    from tensorzero import AsyncTensorZeroGateway

    from t0_agents.core.config import AgentConfig
    from t0_agents.core.output import OutputMode
    from t0_agents.core.run import Run
    from t0_agents.core.types import (
        AgentResult,
        MemoryProvider,
        ToolCallUsage,
        ToolProvider,
    )


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------

Next = Callable[["RunContext"], Awaitable["AgentResult[Any]"]]
"""Callable that advances the pipeline to the next layer.

Each middleware receives one of these and must call it at most once.
Calling it zero times short-circuits the chain (the middleware returns
a synthesised :class:`AgentResult` without invoking the terminal).
Calling it twice raises :class:`~t0_agents.core.exceptions.T0AgentsError`.
"""


OnInferenceCallback = Callable[[Any], Awaitable[None]]
"""Inference observation callback.

Both :class:`tensorzero.types.ChatInferenceResponse` and
:class:`tensorzero.types.JsonInferenceResponse` expose ``usage``,
``inference_id``, and ``episode_id`` attributes but they are not related
by inheritance.  Callbacks accept either via duck typing — the
parameter is typed as :class:`Any` to keep the call-site free of
unions while preserving strict typing inside each callback body.
"""


OnToolRoundCallback = Callable[[tuple["ToolCallUsage", ...]], Awaitable[None]]
"""Tool-round observation callback.

Receives the round's :class:`~t0_agents.core.types.ToolCallUsage` values
in dispatch order, after the results were appended to the transcript.
"""


@runtime_checkable
class Middleware(Protocol):
    """Composable execution-chain element.

    Implementers expose a ``name`` (used in observability span labels
    and in error messages) and an ``async __call__`` that receives the
    current :class:`RunContext` and a :data:`Next` callable.  The body
    decides whether and when to call ``next_(ctx)`` — setup before,
    teardown after, short-circuit by returning early, or replay by
    awaiting twice (rejected at runtime).
    """

    name: str

    async def __call__(
        self,
        ctx: RunContext,
        next_: Next,
    ) -> AgentResult[Any]: ...


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Mutable per-Run state threaded through the middleware chain.

    Created at the dispatch boundary of :class:`~t0_agents.core.run.Run`
    and consumed by every middleware and the terminal.  Fields fall into
    four groups:

    1. **Identity / request** — the ``config`` snapshot and the
       caller-supplied invocation kwargs.  ``user_message`` is mutable
       so middlewares like a PII redactor or a memory-prepended-message
       strategy can replace it before the terminal runs; the immutable
       original is always reachable via ``ctx.run.input.user_message``.
    2. **Gateway + tools** — the live ``t0`` gateway handle and the
       current ``tool_provider``.  Middlewares wrap or replace
       ``tool_provider`` via :meth:`replace_tool_provider`; the previous
       provider stack is recorded on ``extras`` for audit.
    3. **Per-run scratch** — ``on_inference`` and ``on_tool_round`` carry
       the chained callbacks assembled by middlewares calling
       :meth:`chain_on_inference` / :meth:`chain_on_tool_round`.  ``prior_messages`` and
       ``system_arguments`` start as the caller's explicit values and
       may be augmented by :class:`MemoryMiddleware`.  ``extras`` is a
       free-form dict for middleware-private state (e.g.,
       ``"cumulative_usage"`` for cost tracking).
    4. **Outputs** — the terminal writes ``result``; middlewares may
       observe it on the way out.  ``run`` is the live
       :class:`~t0_agents.core.run.Run` (used by
       :class:`CostTrackingMiddleware` to push cost / usage totals onto
       the read-side properties).
    """

    # 1. Identity / request
    config: AgentConfig
    user_message: str
    episode_id: str | None
    tags: dict[str, str] | None
    system: str | None
    system_arguments: dict[str, Any] | None
    output: OutputMode[Any] | None

    # 2. Gateway + tools
    t0: AsyncTensorZeroGateway
    tool_provider: ToolProvider | None

    # 3. Per-run scratch
    on_inference: OnInferenceCallback | None = None
    on_tool_round: OnToolRoundCallback | None = None
    prior_messages: list[Mapping[str, Any]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    # Per-run override state threaded from ``Agent.invoke``:
    # ``tool_provider_overridden`` marks that ``tool_provider`` above came
    # from ``invoke(tool_provider=…)`` rather than the agent (the JSON
    # tool-mode safety check must then verify it un-latched);
    # ``memory_override`` swaps the provider :class:`MemoryMiddleware`
    # uses for this run only.
    tool_provider_overridden: bool = False
    memory_override: MemoryProvider | None = None

    # 4. Outputs
    result: AgentResult[Any] | None = None
    run: Run[Any] | None = None

    def chain_on_inference(self, cb: OnInferenceCallback) -> None:
        """Compose ``cb`` after the existing ``on_inference`` callback.

        Multiple middlewares can observe inferences without overwriting
        each other.  The chain fires in registration order — the first
        middleware to register fires first per inference.  Each callback
        receives the raw inference response (a
        :class:`~tensorzero.types.ChatInferenceResponse` or
        :class:`~tensorzero.types.JsonInferenceResponse` depending on
        the terminal); use ``getattr`` to read fields portably.

        Example::

            async def my_tracker(response):
                cost = getattr(response.usage, "cost", 0.0) or 0.0
                self.total += cost

            ctx.chain_on_inference(my_tracker)
        """
        existing = self.on_inference

        if existing is None:
            self.on_inference = cb
            return

        async def composed(response: Any) -> None:
            await existing(response)
            await cb(response)

        self.on_inference = composed

    def chain_on_tool_round(self, cb: OnToolRoundCallback) -> None:
        """Compose ``cb`` after the existing ``on_tool_round`` callback.

        Same composition rule as :meth:`chain_on_inference`: registration
        order is firing order, and each callback receives the whole
        round.  The terminal fires it once per dispatched tool round,
        after the results are appended to the transcript.
        """
        existing = self.on_tool_round

        if existing is None:
            self.on_tool_round = cb
            return

        async def composed(tool_calls: tuple[ToolCallUsage, ...]) -> None:
            await existing(tool_calls)
            await cb(tool_calls)

        self.on_tool_round = composed

    def replace_tool_provider(self, tp: ToolProvider) -> None:
        """Replace ``ctx.tool_provider``, stacking the prior provider for audit.

        Middlewares that wrap the tool provider (e.g.,
        :class:`GuardrailsMiddleware` and :class:`SubAgentMiddleware`)
        record the prior provider on ``ctx.extras["_tool_provider_stack"]``
        so chains can introspect the wrapping order — useful for
        debugging order invariants and for tests that need to assert the
        wrapping sequence.

        The stack grows outermost-first (the original provider is the
        first element; the most recent wrap is the last).
        """
        stack = self.extras.setdefault("_tool_provider_stack", [])
        stack.append(self.tool_provider)
        self.tool_provider = tp


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


class _NextGuard:
    """Wraps a :data:`Next` to detect double-invocation.

    A middleware that calls ``next_()`` twice is almost always a bug —
    the second call re-enters the inner chain with potentially mutated
    state, producing surprising results.  The guard raises a clear
    :class:`T0AgentsError` so the misuse surfaces as soon as it
    happens instead of producing silent corruption downstream.

    Single-call usage is byte-identical to the unguarded callable —
    the guard adds only a boolean flag check on entry.
    """

    __slots__ = ("_called", "_inner", "_mw_name")

    def __init__(self, inner: Next, mw_name: str) -> None:
        self._inner = inner
        self._mw_name = mw_name
        self._called = False

    async def __call__(self, ctx: RunContext) -> AgentResult[Any]:
        if self._called:
            raise T0AgentsError(
                f"Middleware {self._mw_name!r} called next_() more than once. "
                "Each middleware must call next_(ctx) at most once per run; "
                "to replay inference, return a fresh Run from a higher layer."
            )
        self._called = True
        return await self._inner(ctx)


def build_pipeline(
    middleware: list[Middleware],
    terminal: Next,
) -> Next:
    """Fold a middleware list around the terminal.

    Composition order: ``middleware[0]`` becomes the **outermost** layer
    (entered first, exited last).  Each layer's ``next_`` is wrapped in
    a :class:`_NextGuard` so double-calls raise a clear error — *unless*
    the middleware sets the class attribute ``allow_retry = True`` (per
    :class:`~t0_agents.core.middleware.builtins.JsonRetryMiddleware`,
    which inherently re-invokes the inner chain on parse failures).
    Retry-enabled middlewares receive the raw inner :class:`Next`
    without guarding.

    When ``middleware`` is empty the terminal is returned unchanged — a
    valid pipeline that just runs the inner inference loop with no
    cross-cutting layers.  When ``middleware`` contains a single entry
    the result is a callable that wraps the terminal in one layer.

    The fold is a right-fold by reversed iteration: at each step we
    capture the current ``chain`` (the not-yet-wrapped inner) and the
    middleware ``mw`` in default-argument values so the closure does
    not pick up the loop variable's final binding (Python closure
    capture pitfall).
    """

    chain = terminal
    for mw in reversed(middleware):
        inner_next: Next
        if getattr(mw, "allow_retry", False):
            inner_next = chain
        else:
            inner_next = _NextGuard(chain, mw.name)

        async def wrapped(
            ctx: RunContext,
            _mw: Middleware = mw,
            _next: Next = inner_next,
        ) -> AgentResult[Any]:
            return await _mw(ctx, _next)

        chain = wrapped
    return chain
