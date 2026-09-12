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
hatch for the full :class:`AgentResult` (``episode_id`` /
``inference_ids`` / diagnostics); :meth:`cost` / :meth:`usage` /
:meth:`feedback` / :meth:`fork` / :meth:`cancel` reach the run-level
surface directly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from dataclasses import replace
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
from t0_agents.stream.events import AgentEvent, DoneEvent
from t0_agents.stream.status import Status, synthesize_statuses

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent
    from t0_agents.core.types import RunUsage

__all__ = ["FluentRun"]

logger = logging.getLogger(__name__)

# Per-call invoke kwargs reachable via ``.with_()`` — the ``Agent.invoke``
# keyword surface minus ``output`` (use ``.text()`` / ``.json()`` / ``.as_()``)
# and ``eager`` (glow always defers).  Per-agent config knobs (``max_turns``,
# ``namespace``, ``memory``, ``memory_strategy``, ``tool_timeout``, …) are set
# on ``@app.agent(...)``, not here.  ``prompt_vars`` is glow-level: the
# App's build closure consumes it (str.format on the effective system
# prompt) and it never reaches ``Agent.invoke``.
_WITH_KWARGS = frozenset(
    {
        "episode_id",
        "tags",
        "system_arguments",
        "system",
        "prior_messages",
        "middleware",
        "context",
        "prompt_vars",
    }
)


T = TypeVar("T")
U = TypeVar("U")
M = TypeVar("M", bound=BaseModel)
R1 = TypeVar("R1")
R2 = TypeVar("R2")
R3 = TypeVar("R3")
R4 = TypeVar("R4")


class _RunLineage:
    """The one mutable cell shared by every ``FluentRun`` derived from a root.

    Fluent overrides each return a *new* handle, but all handles in the
    chain stand for the same logical run — whichever member is consumed
    builds THE :class:`Run` for the family.  The lineage records it so an
    observer holding any member can see the consumption and reach the built
    run: glow's :class:`~t0_agents.glow.conversation.Conversation` holds the
    root ``send()`` returned while the caller typically drives a derived
    handle (``send(...).with_(...)`` / ``.json(...)``).
    """

    __slots__ = ("run",)

    def __init__(self) -> None:
        self.run: Run[Any] | None = None


class FluentRun(Generic[T]):
    """Lazy, fluent wrapper over :class:`~t0_agents.core.run.Run`.

    Construction stores a build closure (``Callable[..., Run[Any]]``,
    taking the chosen mode plus per-call kwargs) and a default
    :class:`OutputMode`.  Each fluent override (:meth:`text`, :meth:`json`,
    :meth:`as_`, :meth:`with_`) returns a *new* ``FluentRun`` with the
    override applied; the prior instance is unchanged — but root and
    derived handles share one :class:`_RunLineage`, so consumption is
    observable from any of them (see :attr:`consumed`).  When the same
    FluentRun's output is overridden a second time (e.g. ``.json(A).json(B)``),
    the second override logs DEBUG so overlap is observable.

    Consumption is lazy: the build closure does not fire until the
    handle is actually consumed — ``await``, :meth:`result`, or the
    first iteration of :meth:`events` / :meth:`stream_text` (all via
    the async :meth:`_ensure_run` seam, which also awaits glow's
    ``prepare`` auto-build hook once).  The resulting :class:`Run` is
    cached so the second consumer joins the first's drive task
    (standard ``Run`` first-consumer-wins semantics).
    """

    def __init__(
        self,
        build: Callable[..., Run[Any]],
        default_output: OutputMode[T],
        *,
        value_transform: Callable[[Any], T] | None = None,
        known_servers: tuple[str, ...] = (),
        prepare: Callable[[], Awaitable[None]] | None = None,
        on_result: Callable[[AgentResult[Any]], None] | None = None,
    ) -> None:
        self._build: Callable[..., Run[Any]] = build
        # Awaited once before the build closure fires (see _ensure_run).
        # ``App._invoke`` injects the auto-build hook here; direct
        # constructions (tests, custom integrations) may leave it None.
        self._prepare: Callable[[], Awaitable[None]] | None = prepare
        self._chosen_output: OutputMode[Any] = default_output
        self._has_override: bool = False
        self._run: Run[Any] | None = None
        # The family's consumption record.  Fresh here (a construction is a
        # root); ``_spawn()`` / ``with_()`` overwrite the derived handle's
        # cell with the parent's, so one send's whole derivation chain
        # shares it.
        self._lineage: _RunLineage = _RunLineage()
        # Mounted MCP namespaces, injected by ``App._invoke`` so
        # :meth:`statuses` decomposes prefixed tool names without
        # reaching back into the App.
        self._known_servers: tuple[str, ...] = known_servers
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
        # Fired exactly once with the raw terminal ``AgentResult`` when the run
        # completes successfully, via whichever consumption path drains it
        # (result()/await, stream_text(), events()).  glow's Conversation
        # installs it to append the boundary turn to ``conv.transcript`` and
        # release the sequential-send guard; a failed / cancelled run never
        # fires it (the failure surfaces to the consumer instead).  Propagated
        # across .with_()/.text()/.json()/.as_() like ``prepare``.
        self._on_result: Callable[[AgentResult[Any]], None] | None = on_result
        self._fired: bool = False

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
        return self._spawn(StructuredOutput(model))

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
        ``prompt_vars`` fills ``{placeholders}`` in the effective system
        prompt (the docstring, or an explicit ``system=``) client-side
        via ``str.format`` — mutually exclusive with
        ``system_arguments`` (TZ server-side templates).

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
            known_servers=self._known_servers,
            prepare=self._prepare,
            on_result=self._on_result,
        )
        new._has_override = self._has_override
        new._call_kwargs = {**self._call_kwargs, **call_kwargs}
        new._lineage = self._lineage
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
        The ``on_result`` hook rides along too (like ``prepare``) so a
        ``conv.send(x).json(M)`` chain still records the turn.
        """
        if self._has_override:
            logger.debug(
                "FluentRun: overriding output %r with %r (last-wins).",
                self._chosen_output.name,
                new_output.name,
            )
        new: FluentRun[Any] = FluentRun(
            self._build,
            new_output,
            known_servers=self._known_servers,
            prepare=self._prepare,
            on_result=self._on_result,
        )
        new._has_override = True
        new._call_kwargs = dict(self._call_kwargs)
        new._lineage = self._lineage
        return new

    # ------------------------------------------------------------------
    # Consumption — builds the underlying Run lazily
    # ------------------------------------------------------------------

    async def _ensure_run(self) -> Run[Any]:
        """Build (once) and return the underlying :class:`Run`.

        The async seam of the boot collapse: when a ``prepare`` hook was
        injected (``App._invoke`` passes the App's auto-build), it is
        awaited exactly once — before the build closure fires — so the
        first consumption pays the App boot (gateway resolution, MCP
        mounts, client connection, ``tools=`` filters) instead of the
        handle construction.  The built Run is cached; concurrent and
        subsequent consumers join it (first-consumer-wins).
        """
        if self._run is None:
            if self._prepare is not None:
                await self._prepare()
            self._run = self._build(self._chosen_output, **self._call_kwargs)
            self._lineage.run = self._run
        return self._run

    def _fire(self, result: AgentResult[Any]) -> None:
        """Invoke the ``on_result`` hook exactly once, if one is installed.

        Called from every *successful* terminal consumption path —
        :meth:`result`, the tail of :meth:`stream_text`, and on the
        :class:`~t0_agents.stream.events.DoneEvent` seen by :meth:`events` — so
        a streamed send records identically to an awaited one.  The ``_fired``
        latch makes it fire-once regardless of how many paths (or how many
        repeat calls) drain the same cached Run.  ``result`` is the raw core
        :class:`AgentResult` (before any glow ``value_transform``); a failed or
        cancelled run never reaches here, so the hook only ever sees successes.
        """
        if self._on_result is None or self._fired:
            return
        self._fired = True
        self._on_result(result)

    @property
    def consumed(self) -> bool:
        """Whether consumption has begun — on this handle or any derived one.

        ``True`` once ``await`` / :meth:`result` / the first iteration of
        :meth:`events` / :meth:`stream_text` has fired the build closure (via
        :meth:`_ensure_run`) of this ``FluentRun`` **or of any handle derived
        from it** via :meth:`text` / :meth:`json` / :meth:`as_` /
        :meth:`with_` — the chain shares one :class:`_RunLineage`.  glow's
        :class:`~t0_agents.glow.conversation.Conversation` reads it to enforce
        sequential sends: ``send()`` returns the root handle, and the send
        counts as consumed whichever derived handle the caller drives.
        """
        return self._lineage.run is not None

    @overload
    def events(
        self, *, on_error: Literal["raise", "event"] = ...
    ) -> AsyncIterator[AgentEvent]: ...
    @overload
    def events(
        self, s1: Selector[Any, R1], /, *, on_error: Literal["raise", "event"] = ...
    ) -> AsyncIterator[R1]: ...
    @overload
    def events(
        self,
        s1: Selector[Any, R1],
        s2: Selector[Any, R2],
        /,
        *,
        on_error: Literal["raise", "event"] = ...,
    ) -> AsyncIterator[R1 | R2]: ...
    @overload
    def events(
        self,
        s1: Selector[Any, R1],
        s2: Selector[Any, R2],
        s3: Selector[Any, R3],
        /,
        *,
        on_error: Literal["raise", "event"] = ...,
    ) -> AsyncIterator[R1 | R2 | R3]: ...
    @overload
    def events(
        self,
        s1: Selector[Any, R1],
        s2: Selector[Any, R2],
        s3: Selector[Any, R3],
        s4: Selector[Any, R4],
        /,
        *,
        on_error: Literal["raise", "event"] = ...,
    ) -> AsyncIterator[R1 | R2 | R3 | R4]: ...

    def events(
        self,
        *selectors: Selector[Any, Any],
        on_error: Literal["raise", "event"] = "event",
    ) -> AsyncIterator[Any]:
        """Iterate the underlying ``Run.events()`` stream, optionally filtered.

        Selector arguments are validated eagerly at call time (a bad
        argument raises here, not mid-stream); the ``Run`` itself is
        built on first iteration — via :meth:`_ensure_run`, so glow's
        auto-build ``prepare`` hook runs first — and cached, so a
        subsequent ``__await__`` joins the same buffer (standard
        first-consumer-wins semantics).

        When no selectors are passed, yields every :class:`AgentEvent`
        from the underlying stream.  When one or more selectors are
        passed, an event passes if **any** selector's
        :meth:`Selector.matches` returns ``True``; the first such
        selector's ``transform`` is applied to the yielded value
        (first-match-wins, mirroring isinstance dispatch ergonomics).

        ``on_error`` defaults to ``"event"`` in glow — a UI-facing
        stream should never raise mid-iteration, so a fatal failure
        arrives as a single terminal
        :class:`~t0_agents.stream.events.RunErrorEvent` instead (v2
        breaking change: core's ``Run.events()`` keeps ``"raise"``).
        Note that a selector filter applies to the terminal event too —
        include ``RunError`` (or use no selectors) if you need to
        observe it.  Pass ``on_error="raise"`` to restore the raising
        contract.  Cancellation always raises in both modes.

        Each positional argument must be a :class:`~t0_agents.glow.selectors.Selector`
        instance (typically one of the module-level constants like
        ``Thought``, ``ToolCall``, ``Text`` — possibly chained with
        ``.where(...)`` / ``.map(...)``).  Passing the bare event class
        instead (e.g. ``Thought`` is a ``Selector[ThoughtEvent, ThoughtEvent]``, not
        ``ThoughtEvent``) raises :class:`TypeError` eagerly with a
        clear message rather than failing with an opaque
        ``AttributeError`` mid-stream.
        """
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

        async def _stream() -> AsyncIterator[Any]:
            run = await self._ensure_run()
            captured: AgentResult[Any] | None = None

            async def _tap() -> AsyncIterator[AgentEvent]:
                # Capture the terminal AgentResult off the DoneEvent so the
                # on_result hook fires from the streamed path too — read here,
                # before any selector can filter DoneEvent out.  A failed
                # stream yields RunErrorEvent (no DoneEvent), leaving
                # ``captured`` None, so the hook does not fire on failure.
                nonlocal captured
                async for ev in run.events(on_error=on_error):
                    if isinstance(ev, DoneEvent):
                        captured = ev.result
                    yield ev

            source: AsyncIterator[Any] = _tap()
            if selectors:
                source = _filter_with_selectors(source, selectors)
            try:
                async for item in source:
                    yield item
            finally:
                # Fire even if the consumer breaks early (GeneratorExit skips
                # the code after the loop): once the terminal DoneEvent was
                # tapped the turn completed and must record — the vest-ai
                # record-in-finally pattern.  ``_fire`` is latched (fire-once).
                if captured is not None:
                    self._fire(captured)

        return _stream()

    def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]:
        """Stream the assistant text — the UI one-liner, reasoning-safe.

        Delegates to :meth:`Run.stream_text`, building the ``Run``
        lazily exactly like :meth:`events` (on first iteration, through
        :meth:`_ensure_run`).  ``fallback=True`` (default) covers
        reasoning models that route the answer through thought
        summaries and stream no text deltas: the final
        ``result.response`` is yielded once after the drain.  Fatal
        errors raise out of the iterator — use
        ``events(on_error="event")`` when you need a terminal event.
        """

        async def _stream() -> AsyncIterator[str]:
            run = await self._ensure_run()
            async for chunk in run.stream_text(fallback=fallback):
                yield chunk
            # Reached only when no fatal error raised out of the iterator
            # (stream_text raises on failure) ⇒ the run succeeded ⇒ the
            # captured result is available for the on_result hook.
            self._fire(await run.result())

        return _stream()

    def statuses(self) -> AsyncIterator[AgentEvent | Status]:
        """Iterate events with user-facing :class:`Status` hints interleaved.

        Wraps ``events(on_error="event")`` in
        :func:`~t0_agents.stream.status.synthesize_statuses`, using the
        App's mounted MCP namespaces to decompose prefixed tool names —
        ``Status(text="Calling fred_get_series…", server="fred",
        tool="get_series")``.  Original events re-emit first, each
        synthesized hint follows its trigger; composition happens here
        at the consumer boundary, never on a bus or in a worker.
        """
        return synthesize_statuses(
            self.events(), known_servers=self._known_servers
        )

    async def result(self) -> AgentResult[T]:
        """Await the full :class:`AgentResult`.

        Use this when the caller needs ``episode_id``, ``inference_ids``,
        or the diagnostics envelope.  Aggregate cost / token usage live on
        the run handle itself (:meth:`cost` / :meth:`usage`).  The Pythonic
        fast path is plain ``await fluent_run``, which returns the
        unwrapped ``.value`` directly.

        If a ``value_transform`` was installed at construction time and
        the user has not applied a per-call override (.text / .json /
        .as_), the returned :class:`AgentResult` has ``value`` replaced
        with ``value_transform(original.value)`` via
        :func:`dataclasses.replace`; every other field (present and
        future) is carried over untouched.
        """
        run = await self._ensure_run()
        raw = await run.result()
        self._fire(raw)
        if self._value_transform is None:
            return cast(AgentResult[T], raw)
        # ``replace`` (vs a field-by-field copy) carries over every
        # AgentResult field, so fields added later are not silently dropped.
        return cast(AgentResult[T], replace(raw, value=self._value_transform(raw.value)))

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
        """Return the family's built :class:`Run`, or raise if not yet consumed.

        ``cost`` / ``usage`` / ``feedback`` / ``fork`` operate on the
        underlying Run, which exists only once this FluentRun — or any
        handle derived from it — has been consumed (``await`` it, iterate
        ``.events()``, or call ``.result()``).  Reaching for them earlier
        is a programming error — raise clearly rather than silently kick
        off a hidden run.
        """
        if self._lineage.run is None:
            raise ConfigurationError(
                "FluentRun: cost()/usage()/feedback()/fork() are available "
                "only after the run has been consumed — `await` the run, "
                "iterate `.events()`, or call `.result()` first."
            )
        return self._lineage.run

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

    async def cancel(self) -> None:
        """Cancel the family's underlying Run.  See :meth:`Run.cancel`.

        Awaiting this stops the *drive*, not merely the consumption: the
        drive task is cancelled and awaited, so by the time ``cancel()``
        returns the run is settled at state ``"cancelled"``, no further
        tool call can start, and a tool blocked mid-dispatch has already
        observed the cancellation at its own await point.  Cancelling the
        consumer task instead would leave the drive running unobserved —
        which is why this reaches the run rather than the iterator.

        Every consumption channel then reports it:
        :meth:`events` raises
        :class:`~t0_agents.core.exceptions.RunCancelledError` after
        yielding the events emitted up to the cancellation point — in
        **both** ``on_error`` modes, since cancellation is never converted
        into a terminal event — and :meth:`result` / ``await`` raise it
        directly.  Idempotent: cancelling a settled run is a no-op.

        The run is reached through the shared lineage, so any handle in a
        fluent chain cancels the one run the family stands for: the holder
        of the root handle can cancel a run consumed through a derived one.

        **No-op before consumption.**  Unlike :meth:`cost` / :meth:`usage`
        / :meth:`feedback` / :meth:`fork`, which raise until the run
        exists, cancelling an unconsumed handle succeeds and does nothing —
        there is no run to cancel, which is the same answer
        :meth:`Run.cancel` gives for an already-settled one.  It does not
        arm a pending cancellation, so a caller racing a cancel against the
        first consumption needs its own "cancel requested" flag; this
        method cannot pre-empt a run that was never built.
        """
        run = self._lineage.run
        if run is None:
            return
        await run.cancel()


async def _filter_with_selectors(
    source: AsyncIterator[AgentEvent],
    selectors: tuple[Selector[Any, Any], ...],
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
