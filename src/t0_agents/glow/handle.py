"""``FluentRun[T]`` — lazy, fluent wrapper around ``Run``.

Defers OutputMode selection until ``__await__`` (or ``events()``) so the
caller can apply per-call overrides — ``.text()``, ``.json()``,
``.json(M)``, ``.as_(mode)`` — without rebuilding the underlying
:class:`~t0_agents.core.run.Run`.  Construction stores a build closure
that takes the chosen mode and produces a Run; M2's ``@app.agent`` will
supply that closure as ``lambda mode: agent.invoke(msg, output=mode,
eager=False)``.

``__await__`` returns the *unwrapped* ``T`` directly (the Pythonic fast
path: ``answer = await tutor(query)``).  :meth:`result` is the escape
hatch for the full :class:`AgentResult` when the caller needs
``episode_id`` / ``inference_ids`` / cost / usage.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Generator
from typing import Any, Generic, TypeVar, cast, overload

from pydantic import BaseModel

from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.stream.events import AgentEvent

__all__ = ["FluentRun"]

logger = logging.getLogger(__name__)


T = TypeVar("T")
U = TypeVar("U")
M = TypeVar("M", bound=BaseModel)


class FluentRun(Generic[T]):
    """Lazy, fluent wrapper over :class:`~t0_agents.core.run.Run`.

    Construction stores a build closure (``Callable[[OutputMode[Any]],
    Run[Any]]``) and a default :class:`OutputMode`.  Each fluent
    override (:meth:`text`, :meth:`json`, :meth:`as_`) returns a *new*
    ``FluentRun`` with the override applied; the prior instance is
    unchanged.  When the same FluentRun is overridden a second time
    (e.g. ``.json(A).json(B)``), the second override logs DEBUG so
    overlap is observable.

    Consumption is lazy: the build closure does not fire until
    :meth:`__await__`, :meth:`result`, or :meth:`events` is called; the
    resulting :class:`Run` is cached so the second consumer joins the
    first's drive task (standard ``Run`` first-consumer-wins semantics).
    """

    def __init__(
        self,
        build: Callable[[OutputMode[Any]], Run[Any]],
        default_output: OutputMode[T],
        *,
        value_transform: Callable[[Any], T] | None = None,
    ) -> None:
        self._build: Callable[[OutputMode[Any]], Run[Any]] = build
        self._chosen_output: OutputMode[Any] = default_output
        self._has_override: bool = False
        self._run: Run[Any] | None = None
        # ``value_transform`` is applied to ``AgentResult.value`` when the
        # default ``OutputMode`` is in effect (i.e. the user did not call
        # a per-call override).  Decorator-driven path: ``@app.agent``
        # passes ``lambda v: v.items`` when the function's return type is
        # ``list[BaseModel]`` (M1 synthesises a single-field wrapper model
        # whose ``items: list[T]`` is the actual user-visible payload).
        # Per-call overrides (.text / .json / .as_) deliberately reset
        # the transform — the user has opted out of the default shape.
        self._value_transform: Callable[[Any], T] | None = value_transform

    # ------------------------------------------------------------------
    # Fluent overrides — every override returns a fresh ``FluentRun``
    # ------------------------------------------------------------------

    def text(self) -> FluentRun[str]:
        """Override the output to :class:`ChatOutput` (T=str)."""
        return cast(FluentRun[str], self._spawn(ChatOutput()))

    @overload
    def json(self) -> FluentRun[dict[str, Any]]: ...

    @overload
    def json(self, model: type[M]) -> FluentRun[M]: ...

    def json(self, model: type[M] | None = None) -> FluentRun[Any]:
        """Override to :class:`JsonOutput` or :class:`StructuredOutput(model)`.

        Calling without a model gives a ``dict[str, Any]`` result;
        passing a Pydantic ``BaseModel`` subclass narrows ``T`` to that
        model.
        """
        if model is None:
            return self._spawn(JsonOutput())
        return self._spawn(cast(OutputMode[Any], StructuredOutput(model)))

    def as_(self, mode: OutputMode[U]) -> FluentRun[U]:
        """Override to the supplied :class:`OutputMode`."""
        return cast(FluentRun[U], self._spawn(mode))

    def _spawn(self, new_output: OutputMode[Any]) -> FluentRun[Any]:
        """Construct the next FluentRun in the chain.

        Logs DEBUG when an already-overridden FluentRun is overridden
        again, matching the spec's "last-wins, DEBUG-logged" semantics
        for ``.json(A).json(B)``-style chains.  First-overrides on a
        default-output FluentRun are silent because that is the normal
        per-call shape (``await tutor(msg).json(Model)``).
        """
        if self._has_override:
            logger.debug(
                "FluentRun: overriding output %r with %r (last-wins).",
                self._chosen_output.name,
                new_output.name,
            )
        new: FluentRun[Any] = FluentRun(self._build, new_output)
        new._has_override = True
        return new

    # ------------------------------------------------------------------
    # Consumption — builds the underlying Run lazily
    # ------------------------------------------------------------------

    def events(self, *selectors: Any) -> AsyncIterator[Any]:
        """Iterate the underlying ``Run.events()`` stream, optionally filtered.

        Builds the ``Run`` synchronously at call time so subsequent
        ``__await__`` joins the same cached buffer (standard
        first-consumer-wins semantics).

        When no selectors are passed, yields every :class:`AgentEvent`
        from the underlying stream (matches the v1.0 ``Run.events()``
        contract).  When one or more selectors are passed, an event
        passes if **any** selector's :meth:`Selector.matches` returns
        ``True``; the first such selector's ``transform`` is applied to
        the yielded value (first-match-wins, mirroring isinstance
        dispatch ergonomics).
        """
        if self._run is None:
            self._run = self._build(self._chosen_output)
        run = self._run
        if not selectors:
            return run.events()
        return _filter_with_selectors(run.events(), selectors)

    async def result(self) -> AgentResult[T]:
        """Await the full :class:`AgentResult`.

        Use this when the caller needs ``episode_id``, ``inference_ids``,
        ``cost()`` / ``usage()``, or the diagnostics envelope.  The
        Pythonic fast path is plain ``await fluent_run``, which returns
        the unwrapped ``.value`` directly.

        If a ``value_transform`` was installed at construction time and
        the user has not applied a per-call override (.text / .json /
        .as_), the returned :class:`AgentResult` has ``value`` replaced
        with ``value_transform(original.value)``.  Everything else
        (``episode_id``, ``inference_ids``, ``turns``, etc.) is
        preserved.
        """
        if self._run is None:
            self._run = self._build(self._chosen_output)
        raw = await self._run.result()
        if self._value_transform is None:
            return cast(AgentResult[T], raw)
        return AgentResult(
            value=self._value_transform(raw.value),
            episode_id=raw.episode_id,
            inference_ids=list(raw.inference_ids),
            turns=raw.turns,
            response_source=raw.response_source,
            guardrail_log=list(raw.guardrail_log),
            diagnostics=raw.diagnostics,
        )

    def __await__(self) -> Generator[Any, None, T]:
        """``await fluent_run`` returns the unwrapped ``.value`` of type ``T``."""

        async def _unwrap() -> T:
            result = await self.result()
            return result.value

        return _unwrap().__await__()


async def _filter_with_selectors(
    source: AsyncIterator[AgentEvent],
    selectors: tuple[Any, ...],
) -> AsyncIterator[Any]:
    """Filter + transform an event stream through one or more selectors.

    First-match-wins semantics: every event is tried against the
    selectors in order; the first whose :meth:`Selector.matches`
    returns ``True`` has its :attr:`transform` applied to the event,
    and the result is yielded.  Events matching no selector are
    skipped silently.  This mirrors how :func:`isinstance`-style
    dispatch works in vanilla Python.

    Kept as a module-level coroutine generator (rather than nested
    inside :meth:`FluentRun.events`) so it can be unit-tested directly
    against a fake stream without ``FluentRun`` plumbing.
    """
    async for event in source:
        for selector in selectors:
            if selector.matches(event):
                yield selector.transform(event)
                break
