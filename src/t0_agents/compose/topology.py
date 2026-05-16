"""Core abstractions for the v1.0 ``Composer`` topology primitive.

This module hosts the protocol types every concrete topology (in
:mod:`t0_agents.compose.builtins`) implements, plus the
:class:`CompositeRun` subclass that satisfies the same public surface
as :class:`~t0_agents.core.run.Run` for compositions of Composables.

The contract (``00-integration-contract.md`` §3.4) requires
``Composer.invoke()`` to return ``Run[T]`` so an Agent and a Composer
nest uniformly — ``sequential(race(a, b), conditional(routes=...))``
type-checks regardless of how deep the tree gets.

Cancellation cascades naturally: :meth:`CompositeRun.cancel` is the
parent's :meth:`Run.cancel` override; its ``_drive`` body re-raises
``asyncio.CancelledError`` after asking each tracked sub-Run to cancel
under a bounded timeout (``DEFAULT_CANCEL_TIMEOUT`` by default, per
Q3 resolution).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from t0_agents.core.exceptions import T0AgentsError
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.run import Run, RunInput
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import DoneEvent

logger = logging.getLogger(__name__)

#: Default cap on per-Composer cancellation cascade (seconds).
#:
#: Each topology factory accepts a ``cancel_timeout=`` kwarg that
#: overrides this default for that specific Composer instance.
#: Overshoot logs a single ``WARNING`` and lets sub-Run cancellation
#: continue best-effort.  Resolved per Q3 in
#: ``02-open-questions.md``.
DEFAULT_CANCEL_TIMEOUT: float = 30.0


Input = TypeVar("Input", contravariant=True)
T = TypeVar("T")


@runtime_checkable
class Composable(Protocol[Input, T]):
    """An object whose :meth:`invoke` produces a :class:`Run`.

    Both :class:`~t0_agents.core.agent.Agent` and :class:`Composer`
    satisfy this protocol — an Agent matches ``Composable[str, Any]``
    structurally because ``Agent.invoke(user_message: str, ...)``
    accepts a positional string, and Composers match
    ``Composable[Any, Any]`` because they consume the prior driver's
    chained value.

    Composers compose Composables; nesting is uniform.

    The Protocol uses *invariant* type variables (not contravariant
    Input / covariant T) because :class:`~typing.Protocol` requires
    invariance for runtime-checkable Protocols with method signatures
    that read both ``Input`` and ``T``.  In practice this is a no-op
    constraint — Composables are nested by passing the outer's input
    into the inner's ``invoke`` directly, with no implicit variance.
    """

    def invoke(self, input: Input, /, **kwargs: Any) -> Run[T]: ...


class Composer(Composable[Any, Any]):
    """Base class for every built-in topology.

    Concrete topologies (in :mod:`t0_agents.compose.builtins`) are
    ``@dataclass`` subclasses that carry the topology-specific
    parameters (children, reducers, breakers state) and implement
    :meth:`invoke` by constructing a :class:`CompositeRun` whose
    driver coroutine calls the matching ``_drive_*`` helper.

    The base class does not enforce a structural attribute set —
    each topology declares its own (``children: tuple[...]`` for
    multi-child topologies, ``child: Composable`` for the
    single-child ``map_reduce`` and ``circuit_breaker``).  Every
    concrete topology exposes ``name: str`` for diagnostics.
    """

    name: str

    def invoke(self, input: Any, /, **kwargs: Any) -> Run[Any]:
        raise NotImplementedError(
            "Composer subclasses must implement invoke()"
        )


class ComposerError(T0AgentsError):
    """Raised when a sub-Run inside a Composer fails with topology context.

    The ``node`` attribute references the failing :class:`Composable`
    (an :class:`~t0_agents.core.agent.Agent` or another
    :class:`Composer`) and ``cause`` is the original exception.  Each
    topology decides when to wrap — ``parallel(error_policy="fail_fast")``
    wraps the first failure, ``conditional`` wraps router exceptions,
    ``circuit_breaker`` wraps the synthetic "circuit open" refusal.
    ``retry_chain`` is the exception: it re-raises the last node's
    original exception un-wrapped so upstream ``except SpecificError``
    blocks keep working.
    """

    def __init__(self, node: Composable[Any, Any], cause: BaseException) -> None:
        super().__init__(f"Composer node {node!r} failed: {cause!r}")
        self.node = node
        self.cause = cause


def _coerce_user_message(input_value: Any) -> str:
    """Render an arbitrary input value into the synthetic ``user_message``.

    :class:`RunInput.user_message` is typed ``str`` — a Composer's
    input may be anything, so we coerce: strings pass through,
    everything else uses ``repr`` and truncates to keep the snapshot
    light.  Only consumed by the synthetic :class:`RunInput`; the real
    chained values flow through each driver via the ``chain`` callback.
    """
    if isinstance(input_value, str):
        return input_value
    rendered = repr(input_value)
    if len(rendered) > 256:
        return rendered[:253] + "..."
    return rendered


_COMPOSITE_OUTPUT = ChatOutput()
"""Placeholder :class:`~t0_agents.core.outputs.chat.ChatOutput` for every
:class:`CompositeRun`.

The composite ``_drive`` never calls ``OutputMode.shape_inference`` or
``materialize`` — those run inside each sub-Run's own pipeline — so
the choice of ``_output`` is cosmetic.  ``ChatOutput`` is convenient
because ``supports_streaming=True`` lets callers iterate
``composer.invoke(...).events()`` and receive the terminal
:class:`DoneEvent` without raising the v1.0 streaming gate.  The
shared instance is fine: ``ChatOutput`` is stateless.
"""


class CompositeRun(Run[Any]):
    """Concrete :class:`Run` produced by :meth:`Composer.invoke`.

    Subclasses :class:`Run` rather than duck-typing it so the
    ``Composable.invoke -> Run[T]`` Protocol holds nominally
    (``isinstance(composite, Run)``).  The base class is constructed
    with ``agent=None`` — :meth:`Run.__init__` widens to allow
    composite Runs — and the composite-mode ``_drive`` override
    delegates to the topology driver coroutine instead of touching the
    Agent middleware chain.

    Sub-Runs spawned by the driver are appended to ``_sub_runs`` so
    :meth:`cancel` can cascade.  The driver is the only writer; the
    field is otherwise treated as private.

    What is *not* supported on composite Runs today: :meth:`Run.feedback`,
    :meth:`Run.replay`, :meth:`Run.fork`.  These are well-defined on
    Agent-backed Runs and would require either threading an Agent into
    the CompositeRun or designing per-topology semantics — both are
    deferred.  They raise ``NotImplementedError`` with a pointer at
    the underlying sub-Run.
    """

    def __init__(
        self,
        topology_name: str,
        input_value: Any,
        driver: Callable[[CompositeRun], Awaitable[AgentResult[Any]]],
        *,
        cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        synthetic_input = RunInput(
            user_message=_coerce_user_message(input_value),
            function_name=f"compose:{topology_name}",
        )
        # Defer the eager start until composite-specific fields are
        # populated — ``_drive`` reads ``self._driver`` /
        # ``self._sub_runs`` immediately.
        super().__init__(
            None,
            synthetic_input,
            _COMPOSITE_OUTPUT,
            eager=False,
            context=context,
            middleware=None,
        )
        self._topology_name = topology_name
        self._driver = driver
        self._cancel_timeout = cancel_timeout
        self._sub_runs: list[Run[Any]] = []
        if eager:
            self._start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _drive(self) -> None:
        """Run the topology driver, capture the composite result.

        Mirrors the single-shot branch of :meth:`Run._drive`: no
        ``_mode_chosen`` wait, no per-sub-Run event aggregation.
        Sub-Runs maintain their own buffers; consumers can introspect
        them via ``composite._sub_runs`` (private — a public accessor
        will arrive when there is a concrete consumer ask).
        """
        try:
            result = await self._driver(self)
            self._capture_result(result)
            await self._buffer.put(DoneEvent(result=result))
            self._state = "succeeded"
        except asyncio.CancelledError:
            # Cascade to children before the CancelledError unwinds —
            # by the time ``cancel()`` awaits the task, every sub-Run
            # should already be terminal.
            await self._cancel_sub_runs()
            self._state = "cancelled"
            raise
        except BaseException as exc:
            self._exception = exc
            self._state = "failed"
        finally:
            try:
                await self._buffer.close()
            except Exception:
                logger.debug(
                    "EventBuffer.close raised during composite drive",
                    exc_info=True,
                )

    async def _cancel_sub_runs(self) -> None:
        """Best-effort cascade: cancel every tracked sub-Run.

        Bounded by ``self._cancel_timeout``.  Overshoot logs a WARN
        and lets remaining children dangle — the alternative (block
        the caller forever) is worse.  ``return_exceptions=True``
        keeps a single failing ``cancel()`` from short-circuiting the
        rest of the cascade.
        """
        if not self._sub_runs:
            return
        try:
            async with asyncio.timeout(self._cancel_timeout):
                await asyncio.gather(
                    *(r.cancel() for r in self._sub_runs),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning(
                "Composite '%s' cancel exceeded %.1fs; %d sub-Runs may leak",
                self._topology_name,
                self._cancel_timeout,
                len(self._sub_runs),
            )

    # ------------------------------------------------------------------
    # Unsupported on composite Runs (deferred until a concrete consumer)
    # ------------------------------------------------------------------

    async def feedback(
        self,
        name: str,
        value: float | str | bool,
        *,
        target: Literal["episode", "inference"] = "episode",
        inference_id: str | None = None,
    ) -> None:
        raise NotImplementedError(
            "Run.feedback() is not implemented on CompositeRun in v1.0. "
            "Send feedback against a specific sub-Run via "
            "`composite._sub_runs[i].feedback(...)`."
        )

    async def replay(
        self,
        *,
        agent: Any | None = None,
        variant: str | None = None,
        tags: Mapping[str, str] | None = None,
        episode_id: str | None = None,
    ) -> Run[Any]:
        raise NotImplementedError(
            "Run.replay() is not implemented on CompositeRun in v1.0. "
            "Re-invoke the Composer directly: `composer.invoke(input)`."
        )

    async def fork(
        self,
        extra_message: str,
        *,
        agent: Any | None = None,
    ) -> Run[Any]:
        raise NotImplementedError(
            "Run.fork() is not implemented on CompositeRun in v1.0. "
            "Compose explicitly: `sequential(composer, follow_up_agent)`."
        )
