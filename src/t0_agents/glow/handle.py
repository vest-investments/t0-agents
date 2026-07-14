"""``FluentRun[T]`` — lazy, fluent wrapper around ``Run``.

Defers OutputMode selection until ``__await__`` (or ``events()``) so the
caller can apply per-call overrides — ``.text()``, ``.json()``,
``.json(M)``, ``.as_(mode)`` for output shape and ``.with_(...)`` for the
per-call ``Agent.invoke`` surface (``episode_id``, ``tags``,
``system_arguments``, ``prior_messages``, …) — without rebuilding the
underlying :class:`~t0_agents.core.run.Run`.  Construction stores a build
closure that takes the chosen mode plus those per-call kwargs and produces
a Run; the ``@app.agent`` decorator supplies it (see ``App._invoke``).

``__await__`` returns the *unwrapped* ``T`` directly (the Pythonic fast
path: ``answer = await tutor(query)``).  :meth:`result` is the escape
hatch for the full :class:`AgentResult` when the caller needs
``episode_id`` / ``inference_ids`` / cost / usage.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Generator
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

from pydantic import BaseModel

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow.selectors import Selector
from t0_agents.stream.events import AgentEvent

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent
    from t0_agents.core.types import RunUsage

__all__ = ["FluentRun"]

logger = logging.getLogger(__name__)

# Per-call invoke kwargs reachable via ``.with_()`` — the ``Agent.invoke``
# keyword surface minus ``output`` (use ``.text()`` / ``.json()`` / ``.as_()``)
# and ``eager`` (glow always defers).  Per-agent config knobs (``max_turns``,
# ``namespace``, ``memory``, ``memory_strategy``, ``tool_timeout``, …) are set
# on ``@app.agent(...)``, not here.
_WITH_KWARGS = frozenset(
    {
        "episode_id",
        "tags",
        "system_arguments",
        "system",
        "prior_messages",
        "middleware",
        "context",
    }
)


T = TypeVar("T")
U = TypeVar("U")
M = TypeVar("M", bound=BaseModel)


class FluentRun(Generic[T]):
    """Lazy, fluent wrapper over :class:`~t0_agents.core.run.Run`.

    Construction stores a build closure (``Callable[..., Run[Any]]``,
    taking the chosen mode plus per-call kwargs) and a default
    :class:`OutputMode`.  Each fluent override (:meth:`text`, :meth:`json`,
    :meth:`as_`, :meth:`with_`) returns a *new* ``FluentRun`` with the
    override applied; the prior instance is unchanged.  When the same
    FluentRun's output is overridden a second time (e.g. ``.json(A).json(B)``),
    the second override logs DEBUG so overlap is observable.

    Consumption is lazy: the build closure does not fire until
    :meth:`__await__`, :meth:`result`, or :meth:`events` is called; the
    resulting :class:`Run` is cached so the second consumer joins the
    first's drive task (standard ``Run`` first-consumer-wins semantics).
    """

    def __init__(
        self,
        build: Callable[..., Run[Any]],
        default_output: OutputMode[T],
        *,
        value_transform: Callable[[Any], T] | None = None,
    ) -> None:
        self._build: Callable[..., Run[Any]] = build
        self._chosen_output: OutputMode[Any] = default_output
        self._has_override: bool = False
        self._run: Run[Any] | None = None
        # Per-call ``Agent.invoke`` overrides accumulated via ``.with_()``
        # (``episode_id``, ``tags``, ``system_arguments``, ``prior_messages``,
        # …).  Forwarded to the build closure at consumption time and
        # propagated across ``.text()`` / ``.json()`` / ``.as_()`` overrides
        # (unlike ``value_transform``, which those deliberately reset).
        self._call_kwargs: dict[str, Any] = {}
        # ``value_transform`` is applied to ``AgentResult.value`` when the
        # default ``OutputMode`` is in effect (i.e. the user did not call
        # a per-call override).  Decorator-driven path: ``@app.agent``
        # passes ``lambda v: v.items`` when the function's return type is
        # ``list[BaseModel]`` — output inference synthesises a single-field
        # wrapper model whose ``items: list[T]`` is the actual user-visible
        # payload.  Per-call overrides (.text / .json / .as_) deliberately
        # reset the transform — the user has opted out of the default shape.
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

    def with_(self, **call_kwargs: Any) -> FluentRun[T]:
        """Attach per-call :meth:`Agent.invoke` overrides, returning a new FluentRun.

        Accepts the per-call invoke surface — ``episode_id``, ``tags``,
        ``system_arguments``, ``prior_messages``, ``middleware``,
        ``context``, ``system`` — enabling multi-turn exchanges
        (``episode_id`` + ``prior_messages``), tenancy (``tags``), and
        server-side prompt templates (``system_arguments``) from glow.

        Unlike :meth:`text` / :meth:`json` / :meth:`as_`, ``.with_()`` is
        NOT an output override: it preserves ``value_transform`` and the
        chosen :class:`OutputMode`, so ``tutor(q).with_(episode_id=eid)``
        and ``.json(Model)`` compose in either order.  Repeated ``.with_()``
        calls merge, the last value winning per key.

        ``system_arguments`` and a raw ``system`` prompt are mutually
        exclusive in the gateway; supplying both raises at consumption
        time.  Per-agent config knobs (``max_turns``, ``namespace``,
        ``memory``, …) belong on ``@app.agent(...)`` — passing one here
        raises :class:`ConfigurationError`.
        """
        unknown = set(call_kwargs) - _WITH_KWARGS
        if unknown:
            raise ConfigurationError(
                f"FluentRun.with_(): unknown per-call override(s) "
                f"{sorted(unknown)!r}.  Valid keys: {sorted(_WITH_KWARGS)}.  "
                f"Per-agent config knobs (max_turns, tool_timeout, namespace, "
                f"memory, memory_strategy, …) are set on @app.agent(...)."
            )
        new: FluentRun[T] = FluentRun(
            self._build,
            self._chosen_output,
            value_transform=self._value_transform,
        )
        new._has_override = self._has_override
        new._call_kwargs = {**self._call_kwargs, **call_kwargs}
        return new

    def _spawn(self, new_output: OutputMode[Any]) -> FluentRun[Any]:
        """Construct the next FluentRun in the chain.

        Logs DEBUG when an already-overridden FluentRun is overridden
        again, matching the spec's "last-wins, DEBUG-logged" semantics
        for ``.json(A).json(B)``-style chains.  First-overrides on a
        default-output FluentRun are silent because that is the normal
        per-call shape (``await tutor(msg).json(Model)``).

        ``value_transform`` is deliberately dropped (a per-call output
        override opts out of the default shape), but any ``.with_()``
        call kwargs are propagated so ``.with_(...).json(M)`` keeps them.
        """
        if self._has_override:
            logger.debug(
                "FluentRun: overriding output %r with %r (last-wins).",
                self._chosen_output.name,
                new_output.name,
            )
        new: FluentRun[Any] = FluentRun(self._build, new_output)
        new._has_override = True
        new._call_kwargs = dict(self._call_kwargs)
        return new

    # ------------------------------------------------------------------
    # Consumption — builds the underlying Run lazily
    # ------------------------------------------------------------------

    def events(self, *selectors: Selector[Any]) -> AsyncIterator[Any]:
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

        Each positional argument must be a :class:`~t0_agents.glow.selectors.Selector`
        instance (typically one of the module-level constants like
        ``Thought``, ``ToolCall``, ``Text`` — possibly chained with
        ``.where(...)`` / ``.map(...)``).  Passing the bare event class
        instead (e.g. ``Thought`` is a ``Selector[ThoughtEvent]``, not
        ``ThoughtEvent``) raises :class:`TypeError` eagerly with a
        clear message rather than failing with an opaque
        ``AttributeError`` mid-stream.
        """
        if self._run is None:
            self._run = self._build(self._chosen_output, **self._call_kwargs)
        run = self._run
        if not selectors:
            return run.events()
        for s in selectors:
            if not isinstance(s, Selector):
                raise TypeError(
                    f"FluentRun.events() expected Selector instances; got "
                    f"{type(s).__name__!r}.  Did you pass the event class "
                    f"(e.g. `ThoughtEvent`) instead of the selector "
                    f"constant (e.g. `Thought`)?  Import selectors from "
                    f"`t0_agents.glow` and chain with `.where(...)` / "
                    f"`.map(...)` if you need filters."
                )
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
            self._run = self._build(self._chosen_output, **self._call_kwargs)
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

    # ------------------------------------------------------------------
    # Escape hatch — delegate to the underlying ``Run``
    # ------------------------------------------------------------------

    def _require_run(self) -> Run[Any]:
        """Return the built :class:`Run`, or raise if not yet consumed.

        ``cost`` / ``usage`` / ``feedback`` / ``fork`` operate on the
        underlying Run, which exists only once this FluentRun has been
        consumed (``await`` it, iterate ``.events()``, or call
        ``.result()``).  Reaching for them earlier is a programming
        error — raise clearly rather than silently kick off a hidden run.
        """
        if self._run is None:
            raise ConfigurationError(
                "FluentRun: cost()/usage()/feedback()/fork() are available "
                "only after the run has been consumed — `await` the run, "
                "iterate `.events()`, or call `.result()` first."
            )
        return self._run

    def cost(self) -> float:
        """Aggregate cost (USD) of the underlying Run.  See :meth:`Run.cost`."""
        return self._require_run().cost()

    def usage(self) -> RunUsage | None:
        """Aggregate token usage of the underlying Run.  See :meth:`Run.usage`."""
        return self._require_run().usage()

    async def feedback(
        self,
        name: str,
        value: float | str | bool,
        *,
        target: Literal["episode", "inference"] = "episode",
        inference_id: str | None = None,
    ) -> None:
        """Forward outcome feedback to TensorZero.  See :meth:`Run.feedback`."""
        await self._require_run().feedback(
            name, value, target=target, inference_id=inference_id
        )

    async def fork(self, extra_message: str, *, agent: Agent | None = None) -> Run[T]:
        """Continue this Run with another user turn.  See :meth:`Run.fork`."""
        return cast(Run[T], await self._require_run().fork(extra_message, agent=agent))


async def _filter_with_selectors(
    source: AsyncIterator[AgentEvent],
    selectors: tuple[Selector[Any], ...],
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
