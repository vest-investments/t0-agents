"""Built-in :class:`Composer` topologies for v1.0.

Nine factories cover the industrial / military patterns called out by
the integration contract §3.4:

- :func:`sequential` — pipeline; each step's result feeds the next.
- :func:`parallel` — fan-out + optional reducer with two error policies.
- :func:`race` — first-completed wins; loser cancellation optional.
- :func:`consensus` — N propose, ``merge`` decides.
- :func:`map_reduce` — scatter over an iterable input with bounded
  concurrency, then reduce.
- :func:`conditional` — synchronous router picks a sub-Composable.
- :func:`saga` — sequential with compensation on failure.
- :func:`retry_chain` — provider escalation; fall through nodes on
  failure.
- :func:`circuit_breaker` — state-machine wrapper protecting a single
  child.

Each factory returns a concrete ``@dataclass`` Composer subclass.  All
of them satisfy the :class:`~t0_agents.compose.topology.Composer`
protocol and therefore :class:`~t0_agents.compose.topology.Composable`,
which is what nesting depends on: ``sequential(race(a, b),
conditional(routes={...}))`` type-checks at every layer.

The narrow ``Composer.invoke(input, *, eager, context, cancel_timeout)``
surface is deliberate — composers do not forward Agent-level kwargs to
their children.  Each child owns its own configuration; a composer
that injected kwargs into sub-runs would silently override the
configuration the child was constructed with.  ``parent_episode_id``
correlation is threaded automatically by the runtime via the
:func:`~t0_agents.compose.orchestrator.bind_parent_episode_id`
ContextVar and the TZ-native ``tags={"t0_agents.parent_episode_id":
...}`` kwarg on Agent children.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from t0_agents.compose._runtime import (
    _default_sequential_chain,
    _drive_circuit_breaker,
    _drive_conditional,
    _drive_consensus,
    _drive_map_reduce,
    _drive_parallel,
    _drive_race,
    _drive_retry_chain,
    _drive_saga,
    _drive_sequential,
)
from t0_agents.compose.topology import (
    DEFAULT_CANCEL_TIMEOUT,
    Composable,
    Composer,
    CompositeRun,
)
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult

__all__ = [
    "CircuitBreakerComposer",
    "ConditionalComposer",
    "ConsensusComposer",
    "MapReduceComposer",
    "ParallelComposer",
    "RaceComposer",
    "RetryChainComposer",
    "SagaComposer",
    "SequentialComposer",
    "circuit_breaker",
    "conditional",
    "consensus",
    "map_reduce",
    "parallel",
    "race",
    "retry_chain",
    "saga",
    "sequential",
]


# ----------------------------------------------------------------------
# sequential
# ----------------------------------------------------------------------


@dataclass
class SequentialComposer(Composer):
    """Pipeline composer: each child's result feeds the next via ``chain``."""

    children: tuple[Composable[Any, Any], ...]
    chain: Callable[[AgentResult[Any], Any], Any]
    name: str = "sequential"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_sequential(
                composite, self, input, self.children, self.chain,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def sequential(
    *nodes: Composable[Any, Any],
    chain: Callable[[AgentResult[Any], Any], Any] | None = None,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> SequentialComposer:
    """Pipeline of Composables; each node's result feeds the next.

    The default ``chain`` coerces non-string values to JSON via
    :func:`json.dumps` (with ``default=str``) so the next Agent's
    ``user_message: str`` always gets a deterministic textual form.
    Custom chains receive ``(prev_result: AgentResult, original_input:
    Any)`` and return the next node's input — useful for routing on
    a field of a structured prior result or threading the original
    input alongside.

    Cancellation: cancelling the returned :class:`Run` cancels the
    current sub-Run and skips remaining nodes.  The first sub-Run's
    ``episode_id`` becomes the root parent for the rest.
    """
    return SequentialComposer(
        children=nodes,
        chain=chain if chain is not None else _default_sequential_chain,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# parallel
# ----------------------------------------------------------------------


@dataclass
class ParallelComposer(Composer):
    """Fan-out composer: every child runs on the same input concurrently."""

    children: tuple[Composable[Any, Any], ...]
    reducer: Callable[[Any], Any] | None
    error_policy: Literal["fail_fast", "continue"]
    name: str = "parallel"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_parallel(
                composite, self, input, self.children,
                self.reducer, self.error_policy,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def parallel(
    *nodes: Composable[Any, Any],
    reducer: Callable[[Any], Any] | None = None,
    error_policy: Literal["fail_fast", "continue"] = "fail_fast",
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> ParallelComposer:
    """Run every child on the same input; aggregate via ``reducer``.

    With ``error_policy="fail_fast"`` (default), the first sub-Run
    failure cancels siblings (best-effort under ``cancel_timeout``)
    and propagates as :class:`~t0_agents.compose.topology.ComposerError`
    with the failing child as ``node``.  The ``reducer`` then receives
    ``list[AgentResult[T]]`` of all successful results.

    With ``error_policy="continue"``, every child runs to terminal
    state; the ``reducer`` receives ``list[AgentResult[T] |
    BaseException]`` so callers can ``isinstance(item, BaseException)``
    to detect failures.  Reducer exceptions are wrapped in
    :class:`ComposerError` with the parent composer as ``node``.

    The default ``reducer`` (when ``None``) extracts every
    :attr:`AgentResult.value` into a list.
    """
    return ParallelComposer(
        children=nodes,
        reducer=reducer,
        error_policy=error_policy,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# race
# ----------------------------------------------------------------------


@dataclass
class RaceComposer(Composer):
    """First-completed-wins composer."""

    children: tuple[Composable[Any, Any], ...]
    cancel_others: bool = True
    name: str = "race"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_race(
                composite, self, input, self.children,
                self.cancel_others, timeout,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def race(
    *nodes: Composable[Any, Any],
    cancel_others: bool = True,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> RaceComposer:
    """Race children; whoever finishes first (success or failure) is returned.

    **Deprecated since v1.0** — removal scheduled for v1.1.  Use
    :func:`parallel` with ``reducer=..., error_policy="continue"`` for
    fan-out + merge, or write
    ``asyncio.wait(..., return_when=FIRST_COMPLETED)`` over the leaf
    coroutines in user code if you genuinely need first-finisher-wins.
    See `docs/composition.md` ("Advanced / not recommended") for the
    rationale.

    With ``cancel_others=True`` (default) the losers receive
    :meth:`Run.cancel` under ``cancel_timeout``.  With
    ``cancel_others=False`` the losers run in the background
    detached; ``race`` returns immediately on first completion.  If
    the winner failed, the exception is wrapped in
    :class:`ComposerError`.
    """
    warnings.warn(
        "`race(...)` is deprecated and will be removed in v1.1; use "
        "`parallel(*nodes, reducer=..., error_policy='continue')` for "
        "fan-out + merge or "
        "`asyncio.wait(..., return_when=FIRST_COMPLETED)` in user code "
        "for first-finisher-wins.",
        DeprecationWarning,
        stacklevel=2,
    )
    return RaceComposer(
        children=nodes,
        cancel_others=cancel_others,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# consensus
# ----------------------------------------------------------------------


@dataclass
class ConsensusComposer(Composer):
    """N-propose / merge-decides composer."""

    children: tuple[Composable[Any, Any], ...]
    merge: Callable[[list[Any]], Any]
    name: str = "consensus"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_consensus(
                composite, self, input, self.children, self.merge,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def consensus(
    *nodes: Composable[Any, Any],
    merge: Callable[[list[Any]], Any],
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> ConsensusComposer:
    """Every child proposes; ``merge`` produces the consensus value.

    ``merge`` receives ``list[T]`` of every successful child's
    :attr:`AgentResult.value` (the canonical ``T`` of the
    :class:`OutputMode`).  Any sub-Run failure propagates as
    :class:`ComposerError`; ``merge`` exceptions wrap with the
    parent composer as ``node``.  Tie handling is ``merge``'s job
    — common patterns: most-common with stable tiebreaker, weighted
    average, structured vote count.
    """
    return ConsensusComposer(
        children=nodes,
        merge=merge,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# map_reduce
# ----------------------------------------------------------------------


@dataclass
class MapReduceComposer(Composer):
    """Scatter-gather composer over an iterable input."""

    child: Composable[Any, Any]
    reducer: Callable[[list[AgentResult[Any]]], Any]
    concurrency: int | None = None
    name: str = "map_reduce"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    @property
    def children(self) -> tuple[Composable[Any, Any], ...]:
        return (self.child,)

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_map_reduce(
                composite, self, input, self.child,
                self.reducer, self.concurrency,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def map_reduce(
    node: Composable[Any, Any],
    *,
    reducer: Callable[[list[AgentResult[Any]]], Any],
    concurrency: int | None = None,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> MapReduceComposer:
    """Scatter over an iterable; invoke ``node`` for each item; reduce.

    The input to :meth:`MapReduceComposer.invoke` must be iterable
    (list, tuple, generator — anything supporting ``__iter__`` except
    ``str``/``bytes``, which are explicitly rejected to avoid the
    common bug of mapping over a string's characters).  Each item
    is invoked on ``node`` independently; the ``reducer`` receives
    every result in invocation order.  ``concurrency=None`` runs all
    items concurrently; ``concurrency=N`` bounds active calls to ``N``
    via :class:`asyncio.Semaphore`.
    """
    return MapReduceComposer(
        child=node,
        reducer=reducer,
        concurrency=concurrency,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# conditional
# ----------------------------------------------------------------------


@dataclass
class ConditionalComposer(Composer):
    """Router-driven dispatch composer."""

    router: Callable[[Any], str]
    routes: dict[str, Composable[Any, Any]]
    default: Composable[Any, Any] | None = None
    name: str = "conditional"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    @property
    def children(self) -> tuple[Composable[Any, Any], ...]:
        items: list[Composable[Any, Any]] = list(self.routes.values())
        if self.default is not None:
            items.append(self.default)
        return tuple(items)

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_conditional(
                composite, self, input,
                self.router, self.routes, self.default,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def conditional(
    *,
    router: Callable[[Any], str],
    routes: dict[str, Composable[Any, Any]],
    default: Composable[Any, Any] | None = None,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> ConditionalComposer:
    """Route input to one of N Composables via a synchronous ``router``.

    The ``router`` is a plain Python function (``Callable[[Input],
    str]``) that returns a route key.  If you need LLM-driven
    routing, compose explicitly with a preceding classifier:
    ``sequential(classifier, conditional(router=lambda r: r.value,
    routes={...}))``.

    Unknown keys fall through to ``default`` if supplied;
    otherwise :class:`ComposerError` with a ``ValueError`` cause is
    raised.  Router exceptions are also wrapped in
    :class:`ComposerError`.
    """
    return ConditionalComposer(
        router=router,
        routes=routes,
        default=default,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# saga
# ----------------------------------------------------------------------


@dataclass
class SagaComposer(Composer):
    """Sequential composer with compensation on failure."""

    steps: tuple[tuple[Composable[Any, Any], Composable[Any, Any] | None], ...]
    name: str = "saga"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    @property
    def children(self) -> tuple[Composable[Any, Any], ...]:
        # Both action and compensate live in ``children`` for
        # introspection — handy for "how many composables are in this
        # saga?" but no caller code keys off the structure.
        items: list[Composable[Any, Any]] = []
        for action, compensate in self.steps:
            items.append(action)
            if compensate is not None:
                items.append(compensate)
        return tuple(items)

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_saga(
                composite, self, input, self.steps,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def saga(
    *steps: tuple[Composable[Any, Any], Composable[Any, Any] | None],
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> SagaComposer:
    """Sequential pipeline with reverse-order compensation on failure.

    Each step is a ``(action, compensate)`` pair; ``compensate`` may
    be ``None`` to skip rollback for that step.  When ``action_N``
    fails, ``compensate_{N-1}``, ``compensate_{N-2}``, ...,
    ``compensate_0`` run in reverse order (best-effort: a raising
    compensate logs a warning but doesn't block earlier compensations).

    Compensations receive the **same input** that their paired action
    received (the chained value flowing into ``action_N``).  This is
    the locally consistent contract — encode whatever uniqueness the
    compensate needs (resource ids, etc.) into the action's input.

    Compensations must be **idempotent** (user responsibility, Q10
    resolution); a partial failure may invoke them more than once.
    """
    return SagaComposer(
        steps=steps,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# retry_chain
# ----------------------------------------------------------------------


@dataclass
class RetryChainComposer(Composer):
    """Provider-escalation composer: try children in order until one succeeds."""

    children: tuple[Composable[Any, Any], ...]
    on_error: Callable[[BaseException], bool] | None = None
    name: str = "retry_chain"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_retry_chain(
                composite, self, input, self.children, self.on_error,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def retry_chain(
    *nodes: Composable[Any, Any],
    on_error: Callable[[BaseException], bool] | None = None,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> RetryChainComposer:
    """Cascade through nodes; on failure of node K, try node K+1.

    ``on_error`` filters which exceptions trigger fallback: return
    ``True`` to retry with the next node, ``False`` to propagate the
    current exception immediately without trying any further nodes.
    Default (``None``) retries on every exception.

    If every node fails, the **original** last exception is re-raised
    un-wrapped (not :class:`ComposerError`) so callers' ``except
    SpecificError`` blocks keep working — common pattern is
    ``retry_chain(primary, backup, on_error=lambda e: isinstance(e,
    RateLimitError))``.
    """
    return RetryChainComposer(
        children=nodes,
        on_error=on_error,
        cancel_timeout=cancel_timeout,
    )


# ----------------------------------------------------------------------
# circuit_breaker
# ----------------------------------------------------------------------


@dataclass
class CircuitBreakerComposer(Composer):
    """State-machine wrapper protecting a single child Composable.

    State is per-Composer-instance (per the Q3/Q10 spec
    recommendation): ``_state``, ``_consecutive_failures``,
    ``_opened_at``.  Construct a fresh composer per tenant /
    per request scope if you need isolated state.  The fields are
    intentionally private (``_``-prefixed); test code that wants to
    observe state should use the public behaviour (raising
    :class:`ComposerError` while open).
    """

    child: Composable[Any, Any]
    threshold: int = 5
    reset_timeout: float = 30.0
    name: str = "circuit_breaker"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT
    _state: Literal["closed", "open", "half_open"] = field(
        default="closed", init=False, repr=False, compare=False,
    )
    _consecutive_failures: int = field(
        default=0, init=False, repr=False, compare=False,
    )
    _opened_at: float | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @property
    def children(self) -> tuple[Composable[Any, Any], ...]:
        return (self.child,)

    def invoke(  # type: ignore[override]
        self,
        input: Any,
        /,
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        cancel_timeout: float | None = None,
    ) -> Run[Any]:
        timeout = (
            cancel_timeout if cancel_timeout is not None else self.cancel_timeout
        )

        async def _driver(composite: CompositeRun) -> AgentResult[Any]:
            return await _drive_circuit_breaker(
                composite, self, input, self.child,
            )

        return CompositeRun(
            self.name, input, _driver,
            cancel_timeout=timeout, eager=eager, context=context,
        )


def circuit_breaker(
    node: Composable[Any, Any],
    *,
    threshold: int = 5,
    reset_timeout: float = 30.0,
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
) -> CircuitBreakerComposer:
    """Wrap a Composable with a state-machine breaker.

    Closed → call ``node``.  On ``threshold`` consecutive failures
    the breaker opens.  Open → refuse calls (raise
    :class:`ComposerError` synthesised as ``RuntimeError("circuit
    open")``) without invoking ``node``.  After ``reset_timeout``
    seconds the next call transitions to half-open and attempts
    once; success closes, failure re-opens.

    State is **per-Composer-instance**, not per-call.  Two
    invocations of the same ``circuit_breaker(...)`` instance share
    the same state machine — that's the point of the pattern.
    Construct fresh composers per tenant when isolation is required.
    """
    return CircuitBreakerComposer(
        child=node,
        threshold=threshold,
        reset_timeout=reset_timeout,
        cancel_timeout=cancel_timeout,
    )
