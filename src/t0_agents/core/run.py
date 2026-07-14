"""First-class invocation handle for :meth:`Agent.invoke`.

:class:`Run` represents one live or completed agent invocation.  It
is the value returned by :meth:`t0_agents.core.agent.Agent.invoke`
and the type Middleware and OutputMode build on.

The Run drives in one of two modes: the first consumer's call
(``result()`` vs ``events()``) picks it.  ``result()`` runs the
non-streaming chat path (or the single-shot path); ``events()`` runs
the streaming chat path.  Output modes whose
:attr:`~OutputMode.supports_streaming` is ``False`` reject ``events()``
with :class:`~t0_agents.core.exceptions.ConfigurationError`.

The Run holds a :class:`weakref.ref` to its :class:`Agent`, so a
discarded Run does not extend the gateway's lifetime.  ``replay()``
and ``fork()`` accept an explicit ``agent=`` for the case where the
original Agent has been garbage-collected.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterator, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from t0_agents.core._runtime import EventBuffer
from t0_agents.core.exceptions import (
    ConfigurationError,
    MaxTurnsExceededError,
    RunCancelledError,
    RunFailedError,
)
from t0_agents.core.middleware.base import RunContext
from t0_agents.core.output import OutputMode
from t0_agents.core.types import AgentResult, RunUsage, Usage
from t0_agents.stream.events import (
    AgentEvent,
    DoneEvent,
    RunErrorEvent,
    TextDeltaEvent,
)

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent

logger = logging.getLogger(__name__)

T = TypeVar("T")

RunState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
"""Lifecycle state of a :class:`Run`.

- ``"pending"`` — lazy-mode Run not yet started.
- ``"running"`` — task scheduled (may still be awaiting first-consumer signal for chat).
- ``"succeeded"`` — terminal: ``result`` populated.
- ``"failed"`` — terminal: ``exception`` populated; ``await run.result()`` raises
  :class:`RunFailedError`.
- ``"cancelled"`` — terminal: ``await run.result()`` raises :class:`RunCancelledError`.
"""


def _as_run_error(exc: BaseException) -> RunErrorEvent:
    """Map a drive-time failure to its terminal :class:`RunErrorEvent`.

    ``message`` prefers the exception's ``text`` attribute when one is
    set — ``TensorZeroError`` carries the provider's detailed error
    fragments there while ``str(e)`` is often just a generic status
    line — then falls back to ``str(exc)``, then to the class name
    (bare exceptions like ``TimeoutError()`` stringify empty).  Known
    loop-guard exceptions contribute structured ``detail`` so wire
    consumers don't have to parse the human message.
    """
    detail: dict[str, Any] = {}
    if isinstance(exc, MaxTurnsExceededError):
        detail["max_turns"] = exc.max_turns
    text = getattr(exc, "text", None)
    message = (text if isinstance(text, str) else None) or str(exc) or exc.__class__.__name__
    return RunErrorEvent(
        error_type=exc.__class__.__name__,
        message=message,
        detail=detail,
        exception=exc,
    )


@dataclass(frozen=True, slots=True)
class RunInput:
    """Immutable snapshot of one ``Agent.invoke(...)`` call.

    Frozen so :meth:`Run.replay` and :meth:`Run.fork` can derive new
    inputs by :func:`dataclasses.replace` without mutating the source.
    Per-call kwargs land here; the :class:`OutputMode` is stored
    separately on the :class:`Run` itself (so each Run knows how to
    materialise its result without re-deriving from config).
    """

    user_message: str
    function_name: str | None
    episode_id: str | None = None
    tags: Mapping[str, str] | None = None
    system: str | None = None
    system_arguments: Mapping[str, Any] | None = None
    prior_messages: tuple[Mapping[str, Any], ...] = ()


class Run(Generic[T]):
    """Live or completed handle for one ``Agent.invoke(...)`` call.

    Two consumption channels exist and both work concurrently for
    streaming-capable agents: ``await run.result()`` returns the final
    :class:`AgentResult`, and ``async for ev in run.events()`` yields
    every :class:`AgentEvent` as the streaming runner emits it.  Agents
    whose :class:`OutputMode` does not support streaming (e.g.
    :class:`JsonOutput`, :class:`StructuredOutput`) only support
    ``result()`` — calling ``events()`` raises
    :class:`ConfigurationError` (TZ does not emit ``output.parsed``
    mid-stream).

    The first call to ``result()`` or ``events()`` picks the drive
    mode for streaming-capable agents.  Subsequent calls share that
    mode — calling ``events()`` after ``result()`` is well-defined but
    the buffer will only have the synthetic ``DoneEvent`` that the
    non-streaming drive emits at the end of the run.
    """

    def __init__(
        self,
        agent: Agent | None,
        input_: RunInput,
        output: OutputMode[Any],
        *,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
        middleware: Sequence[Any] | None = None,
    ) -> None:
        # ``agent=None`` is supported for third-party Run-subclass
        # authors whose ``_drive()`` implementation does not touch
        # ``self._agent_ref``.  Every Agent-backed Run continues to use
        # ``weakref.ref(agent)`` so the gateway lifetime is not extended.
        self._agent_ref: Callable[[], Agent | None] = (
            weakref.ref(agent) if agent is not None else (lambda: None)
        )
        self._input = input_
        self._output: OutputMode[Any] = output
        self._context: dict[str, Any] = dict(context) if context else {}
        self._middleware: Sequence[Any] | None = middleware
        self._state: RunState = "pending"
        self._buffer = EventBuffer()
        self._result: AgentResult[Any] | None = None
        self._exception: BaseException | None = None
        self._task: asyncio.Task[None] | None = None
        self._mode: Literal["result", "events"] | None = None
        self._mode_chosen = asyncio.Event()
        self._episode_id: str | None = input_.episode_id
        self._inference_ids: tuple[str, ...] = ()
        self._final_response: str = ""
        # Cost / usage totals — written by ``CostTrackingMiddleware`` via
        # ``_set_cost`` / ``_set_usage``.  ``_usages`` is the per-inference
        # list (written via ``_set_usages``) that backs the live context
        # footprint, distinct from the summing ``_usage`` aggregate.
        self._cost: float = 0.0
        self._usage: RunUsage | None = None
        self._usages: tuple[Usage, ...] = ()
        if eager:
            self._start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def input(self) -> RunInput:
        """The frozen :class:`RunInput` snapshot for this Run."""
        return self._input

    @property
    def output(self) -> OutputMode[Any]:
        """The :class:`OutputMode` driving this Run."""
        return self._output

    @property
    def context(self) -> dict[str, Any]:
        """User-supplied metadata dict (mutable)."""
        return self._context

    @property
    def state(self) -> RunState:
        """Lifecycle state."""
        return self._state

    @property
    def episode_id(self) -> str | None:
        """TensorZero episode id (populated mid-run on the first inference)."""
        return self._episode_id

    @property
    def inference_ids(self) -> tuple[str, ...]:
        """Tuple of TensorZero inference ids produced by this Run, in order."""
        return self._inference_ids

    @property
    def agent(self) -> Agent | None:
        """The originating :class:`Agent`, or ``None`` if it has been GC'd."""
        return self._agent_ref()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Schedule the drive task — idempotent."""
        if self._task is not None:
            return
        self._state = "running"
        self._task = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        """The asyncio task body.  Dispatches on ``output.expects_tool_loop``.

        Builds a :class:`RunContext` from the frozen :class:`RunInput`
        snapshot and dispatches to :meth:`Agent._drive` which wraps the
        configured middleware chain around the appropriate terminal.
        For streaming chat the terminal pushes events to
        ``self._buffer`` mid-iteration; this method pushes the synthetic
        terminal :class:`DoneEvent` AFTER the chain unwinds so outer
        middlewares (e.g. :class:`GuardrailsMiddleware`) have a chance
        to mutate ``result.guardrail_log`` first.
        """
        try:
            agent = self._agent_ref()
            if agent is None:
                raise RuntimeError(
                    "Agent was garbage-collected before its Run could start. "
                    "Keep a reference to the Agent for the lifetime of the Run."
                )

            if not self._output.expects_tool_loop:
                # Single-shot: only result() is valid (events() raises eagerly).
                # No need to wait on _mode_chosen.
                ctx = self._build_run_context(agent, is_streaming=False)
                result = await agent._drive(
                    ctx,
                    is_streaming=False,
                    middleware=self._middleware,
                )
                self._capture_result(result)
                await self._buffer.put(DoneEvent(result=result))
                self._state = "succeeded"
                return

            # Chat / tool-loop: wait for first consumer to choose drive mode.
            await self._mode_chosen.wait()
            is_streaming = self._mode == "events"
            ctx = self._build_run_context(agent, is_streaming=is_streaming)
            result = await agent._drive(
                ctx,
                is_streaming=is_streaming,
                middleware=self._middleware,
            )
            self._capture_result(result)
            # Streaming mode: the terminal pushed every event except the
            # DoneEvent.  Non-streaming: no events were pushed.  Either
            # way, surface the terminal DoneEvent now with the final,
            # outer-middleware-mutated result.
            await self._buffer.put(DoneEvent(result=result))
            self._state = "succeeded"
        except asyncio.CancelledError:
            self._state = "cancelled"
            raise
        except BaseException as exc:
            self._exception = exc
            self._state = "failed"
        finally:
            # Always close so any waiting events() consumer exits cleanly.
            try:
                await self._buffer.close()
            except Exception:
                logger.debug("EventBuffer.close raised; suppressed", exc_info=True)

    def _build_run_context(
        self,
        agent: Agent,
        *,
        is_streaming: bool,
    ) -> RunContext:
        """Snapshot the :class:`RunInput` into a mutable :class:`RunContext`.

        Called at the dispatch boundary.  Per-call kwargs land on the
        ``RunContext`` fields; the immutable original is always reachable
        via ``ctx.run.input.<field>``.  Memory middleware may modify
        ``ctx.user_message`` / ``ctx.system_arguments`` /
        ``ctx.prior_messages`` before the terminal runs.

        ``is_streaming`` is accepted for forward-compat: future
        middleware may want to know the drive mode at chain build time.
        """
        del is_streaming  # not used today; future middleware may key on it.
        return RunContext(
            config=agent.config,
            user_message=self._input.user_message,
            episode_id=self._input.episode_id,
            tags=self._dict_or_none(self._input.tags),
            system=self._input.system,
            system_arguments=self._dict_or_none(self._input.system_arguments),
            output=self._output,
            t0=agent._t0,
            tool_provider=agent._tool_provider,
            prior_messages=(
                list(self._input.prior_messages)
                if self._input.prior_messages
                else None
            ),
            run=self,
        )

    def _capture_result(self, result: AgentResult[Any]) -> None:
        """Record the final AgentResult on the Run's read-side properties."""
        self._result = result
        if result.episode_id:
            self._episode_id = result.episode_id
        self._inference_ids = tuple(result.inference_ids)
        self._final_response = result.response

    @staticmethod
    def _dict_or_none(
        m: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Convert a Mapping snapshot to a fresh dict for the legacy callees.

        The legacy ``_execute_*`` helpers expect ``dict | None``; cast at
        the Run boundary so the frozen ``RunInput`` Mappings stay
        immutable.
        """
        return dict(m) if m is not None else None

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    async def result(self) -> AgentResult[T]:
        """Await the final :class:`AgentResult`.

        For chat agents this selects the non-streaming drive mode when
        called before :meth:`events`.  For single-shot agents the drive
        mode is fixed at construction; ``result()`` returns the
        materialised result.
        """
        if self._mode is None:
            self._mode = "result"
            self._mode_chosen.set()
        if self._task is None:
            # Lazy mode: start now.
            self._start()
        assert self._task is not None
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # Two cases share this branch:
            # (a) THIS coroutine was cancelled by the caller — the
            #     drive task survives via ``shield``; propagate up.
            # (b) the drive task itself was cancelled (via run.cancel)
            #     and shield surfaces its CancelledError on us; the
            #     contract is to translate that into
            #     :class:`RunCancelledError`.
            if self._state == "cancelled":
                raise RunCancelledError(
                    "Run was cancelled before completion",
                ) from None
            raise
        if self._state == "cancelled":
            raise RunCancelledError("Run was cancelled before completion")
        if self._state == "failed":
            # Surface the original exception directly so callers can
            # ``except SpecificError`` with standard Python idioms — same
            # behaviour as :meth:`events`.  :class:`RunFailedError` stays
            # in the type system as a defensive fallback for the case
            # where the drive ended in "failed" state without a captured
            # exception (shouldn't normally happen).
            if self._exception is not None:
                raise self._exception
            raise RunFailedError(
                Exception("Run failed with no captured exception"),
            )
        assert self._result is not None
        return self._result

    async def events(
        self,
        *,
        replay: bool = False,
        on_error: Literal["raise", "event"] = "raise",
    ) -> AsyncIterator[AgentEvent]:
        """Yield every :class:`AgentEvent` produced by this Run.

        For streaming-capable agents the first call to ``events()``
        selects the streaming drive mode.  For non-streaming output
        modes the call raises :class:`ConfigurationError`.

        Args:
            replay: When ``True``, yield from the earliest still-
                buffered event.  When ``False`` (default), yield from
                the current live cursor — events emitted before the
                subscription are skipped.
            on_error: ``"raise"`` (default) re-raises drive-time
                failures out of the iterator — same behaviour as
                :meth:`result`.  ``"event"`` converts the failure into
                a single terminal :class:`RunErrorEvent` instead, so
                the stream upholds the invariant *every stream ends
                with exactly one ``DoneEvent | RunErrorEvent``*.
                Cancellation is never converted: it raises
                :class:`RunCancelledError` in both modes.
        """
        if not self._output.supports_streaming:
            raise ConfigurationError(
                f"Agent.invoke(...).events() is not supported for "
                f"output={self._output.name!r}.  Output mode does not "
                "support streaming; use `await run.result()` instead."
            )
        if self._mode is None:
            self._mode = "events"
            self._mode_chosen.set()
        if self._task is None:
            self._start()
        start_cursor = 0 if replay else self._buffer.total_pushed
        saw_done = False
        async for ev in self._buffer.iter_from(start_cursor=start_cursor):
            if isinstance(ev, DoneEvent):
                saw_done = True
            yield ev
        # Buffer is now closed.  Drive-time failures raise the
        # ORIGINAL exception (``GuardrailBlockedError``,
        # ``MaxTurnsExceededError``, ...) on the iterator — same
        # behaviour as :meth:`result` — unless ``on_error="event"``
        # asked for a terminal event.  Cancellation surfaces as
        # :class:`RunCancelledError` to give the iterator a uniform
        # terminal exception.
        if self._state == "failed" and self._exception is not None:
            if on_error == "raise":
                raise self._exception
            if not saw_done:
                yield _as_run_error(self._exception)
            return
        if self._state == "cancelled":
            raise RunCancelledError("Run was cancelled before completion")

    async def stream_text(self, *, fallback: bool = True) -> AsyncIterator[str]:
        """Yield only the assistant-text deltas as they arrive.

        Convenience wrapper around :meth:`events` that filters down to
        :class:`TextDeltaEvent` and yields the delta string.  Useful
        for UI surfaces that just want streaming text without the
        full event taxonomy.

        Reasoning models (OpenAI o-series, ``gpt-oss`` via Harmony)
        route the final answer through thought summaries and emit no
        ``TextDeltaEvent`` at all.  With ``fallback=True`` (default) a
        turn that streamed no non-blank delta yields the final
        ``result.response`` once, after the stream drains — the
        one-liner never renders an empty answer for a turn that
        produced one.  ``fallback=False`` restores the raw filter.

        Fatal errors raise out of the iterator; use :meth:`events`
        with ``on_error="event"`` when you need a terminal event
        instead.
        """
        saw_text = False
        result: AgentResult[Any] | None = None
        async for ev in self.events():
            if isinstance(ev, TextDeltaEvent):
                if ev.delta.strip():
                    saw_text = True
                yield ev.delta
            elif isinstance(ev, DoneEvent):
                result = ev.result
        if fallback and not saw_text and result is not None:
            response = result.response
            if response.strip():
                yield response

    def __await__(self) -> Generator[Any, None, AgentResult[T]]:
        """Awaiting a Run is an alias for ``await run.result()``."""
        return self.result().__await__()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def cancel(self) -> None:
        """Cancel the drive task and transition to ``"cancelled"``.

        Idempotent: calling ``cancel()`` on a terminal Run is a no-op.
        After cancellation, ``await run.result()`` raises
        :class:`RunCancelledError`; ``events()`` consumers see the
        events emitted up to the cancellation point and then a
        terminal :class:`RunCancelledError`.
        """
        if self._state in ("succeeded", "failed", "cancelled"):
            return
        if self._task is None:
            # Lazy Run that was never started — transition directly.
            self._state = "cancelled"
            await self._buffer.close()
            return
        # cancel() raises CancelledError at the task's current await
        # point regardless of whether the task is waiting on
        # _mode_chosen or already inside the drive body — no need to
        # set _mode_chosen first.
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, RunCancelledError, RunFailedError):
            pass
        except BaseException:
            logger.debug("Run drive task raised during cancel", exc_info=True)
        self._state = "cancelled"
        if not self._buffer.closed:
            await self._buffer.close()

    async def feedback(
        self,
        name: str,
        value: float | str | bool,
        *,
        target: Literal["episode", "inference"] = "episode",
        inference_id: str | None = None,
    ) -> None:
        """Forward outcome feedback to TensorZero.

        Maps to :meth:`tensorzero.AsyncTensorZeroGateway.feedback`.  By
        default the feedback is attached to the run's episode;
        ``target="inference"`` attaches it to a specific inference
        (defaulting to the most recent one when ``inference_id`` is
        omitted).

        Args:
            name: TensorZero metric name (must be pre-declared in
                ``tensorzero.toml``).
            value: Metric value — float / str / bool depending on
                metric type.
            target: ``"episode"`` (default) or ``"inference"``.
            inference_id: Specific inference id when
                ``target="inference"``; defaults to the latest if
                omitted.
        """
        agent = self._agent_ref()
        if agent is None:
            raise RuntimeError(
                "Agent was garbage-collected; cannot send feedback. "
                "Hold a reference to the Agent until feedback is sent."
            )
        t0 = agent._t0
        if target == "episode":
            if self._episode_id is None:
                raise RuntimeError(
                    "Run has no episode_id yet — await run.result() before "
                    "sending feedback."
                )
            await t0.feedback(
                metric_name=name, value=value, episode_id=self._episode_id,
            )
            return
        # target == "inference"
        iid = inference_id or (
            self._inference_ids[-1] if self._inference_ids else None
        )
        if iid is None:
            raise RuntimeError(
                "Run has no inference_ids yet — await run.result() before "
                "sending inference-targeted feedback."
            )
        await t0.feedback(metric_name=name, value=value, inference_id=iid)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    async def replay(
        self,
        *,
        agent: Agent | None = None,
        variant: str | None = None,
        tags: Mapping[str, str] | None = None,
        episode_id: str | None = None,
    ) -> Run[T]:
        """Replay this Run's input on the same or a different agent.

        By default mints a new TensorZero episode (Q7 resolution).
        Pass ``episode_id=original.episode_id`` to preserve the
        correlation; useful when treating replay as "re-execute under
        same conversation."

        ``variant`` flows through TensorZero's tags as
        ``"t0_agents.variant"`` so operators can audit replay
        provenance without an SDK change.
        """
        target = agent if agent is not None else self._agent_ref()
        if target is None:
            raise RuntimeError(
                "Original agent was garbage-collected; pass agent= to "
                "replay against a different Agent."
            )
        merged_tags: dict[str, str] | None
        base_tags: dict[str, str] = (
            dict(self._input.tags) if self._input.tags else {}
        )
        extra: dict[str, str] = dict(tags) if tags else {}
        if variant is not None:
            extra.setdefault("t0_agents.variant", variant)
        merged_tags = {**base_tags, **extra} or None
        return target.invoke(
            self._input.user_message,
            episode_id=episode_id,
            tags=merged_tags,
            system_arguments=self._dict_or_none(self._input.system_arguments),
            system=self._input.system,
            prior_messages=(
                list(self._input.prior_messages)
                if self._input.prior_messages
                else None
            ),
            middleware=self._middleware,
            output=self._output,
            context=self._context,
        )

    async def fork(
        self,
        extra_message: str,
        *,
        agent: Agent | None = None,
    ) -> Run[T]:
        """Continue from this Run's transcript with an additional user turn.

        Awaits the original Run to completion (if not yet terminal),
        then invokes the agent with the recorded transcript as
        ``prior_messages`` and ``extra_message`` as the new user turn.
        The forked Run runs against a fresh episode by default — the
        original and the fork are independent thereafter (cancelling
        one does not affect the other).
        """
        target = agent if agent is not None else self._agent_ref()
        if target is None:
            raise RuntimeError(
                "Original agent was garbage-collected; pass agent= to "
                "fork against a different Agent."
            )
        # Make sure the original ran (need transcript + final response).
        await self.result()
        transcript = self.transcript()
        return target.invoke(
            extra_message,
            tags=self._dict_or_none(self._input.tags),
            system_arguments=self._dict_or_none(self._input.system_arguments),
            system=self._input.system,
            prior_messages=transcript,
            middleware=self._middleware,
            output=self._output,
            context=self._context,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def transcript(self) -> list[Mapping[str, Any]]:
        """Return the recorded conversation as a list of TZ-shaped messages.

        The transcript is any ``prior_messages`` the Run was seeded
        with (so a fork-of-a-fork preserves the full conversation
        chain) followed by the boundary pair — this Run's user turn plus
        the agent's final response.  Intermediate tool turns are not
        captured today; a future version may extend the transcript with
        the full middleware-level message history.
        """
        if self._result is None:
            return []
        return [
            *(dict(m) for m in self._input.prior_messages),
            {"role": "user", "content": self._input.user_message},
            {"role": "assistant", "content": self._final_response},
        ]

    def cost(self) -> float:
        """Aggregate cost (USD) across every inference in this Run.

        Populated by
        :class:`~t0_agents.core.middleware.CostTrackingMiddleware`
        on each inference and snapshotted onto the Run after the
        middleware chain unwinds.  ``0.0`` until the Run produces an
        inference (and forever when running against a provider that
        omits pricing — local models, for instance).
        """
        return self._cost

    def usage(self) -> RunUsage | None:
        """Aggregate token usage across every inference in this Run.

        Populated by
        :class:`~t0_agents.core.middleware.CostTrackingMiddleware` and
        returned as a frozen :class:`RunUsage` aggregate.  ``None``
        until the Run produces an inference (or when the middleware
        was excluded via ``middleware=[]``).

        This *sums* usage across turns; for the current full-conversation
        prompt size (the compaction signal) use :meth:`context_footprint`
        instead, and :meth:`usages` for the per-inference breakdown.
        """
        return self._usage

    def usages(self) -> tuple[Usage, ...]:
        """Per-inference usage for this Run, in order.

        One :class:`~t0_agents.core.types.Usage` per inference (chat
        turns + JSON retries), **including** cache-served turns that
        report ``input_tokens == 0`` — kept for transparency.  Empty
        until the Run produces an inference (or when
        :class:`~t0_agents.core.middleware.CostTrackingMiddleware` was
        excluded via ``middleware=[]``).

        Contrast with :meth:`usage` (the cumulative
        :class:`~t0_agents.core.types.RunUsage` sum) and
        :meth:`context_footprint` (the single latest non-empty turn).
        """
        return self._usages

    def context_footprint(self) -> Usage | None:
        """The current full-conversation prompt size, as a live signal.

        Returns the **latest** inference whose ``input_tokens > 0`` — for
        a multi-turn chat that is approximately the whole conversation
        re-sent on the most recent turn, i.e. the number to compare
        against the model's context window when deciding whether to
        compact or rotate the conversation.

        This is deliberately **not** :meth:`usage`:
        ``usage().input_tokens`` *sums* every turn's prompt, double-
        counting the growing prefix each turn re-sends, so it overstates
        the live footprint.

        Cache-served turns report all-zero usage (a whole-response cache
        hit) and would read as "empty context"; they are skipped, so the
        footprint carries forward over them.  Returns ``None`` until the
        Run produces a non-zero inference.
        """
        for usage in reversed(self._usages):
            if usage.input_tokens > 0:
                return usage
        return None

    def context_remaining(self, window: int) -> int | None:
        """Tokens left in a caller-supplied context ``window``.

        Computed as ``window - context_footprint().input_tokens``.  The
        window is a caller argument because the model / variant — and so
        its context size — is resolved by TensorZero and is not knowable
        in-process.

        May be **negative** when the current prompt already exceeds the
        window: the overflow magnitude is informative for the outer layer
        that owns the compaction / rotation decision, so it is not
        clamped.  Returns ``None`` when there is no footprint yet (no
        non-zero inference).
        """
        footprint = self.context_footprint()
        if footprint is None:
            return None
        return window - footprint.input_tokens

    def _set_cost(self, cost: float) -> None:
        """Private setter used by :class:`CostTrackingMiddleware`."""
        self._cost = cost

    def _set_usage(self, usage: RunUsage) -> None:
        """Private setter used by :class:`CostTrackingMiddleware`."""
        self._usage = usage

    def _set_usages(self, usages: tuple[Usage, ...]) -> None:
        """Private setter for the per-inference usage list.

        Written by :class:`CostTrackingMiddleware` after the chain
        unwinds; read by :meth:`usages` / :meth:`context_footprint`.
        """
        self._usages = usages


def build_run_input(
    *,
    user_message: str,
    function_name: str | None,
    episode_id: str | None = None,
    tags: Mapping[str, str] | None = None,
    system: str | None = None,
    system_arguments: Mapping[str, Any] | None = None,
    prior_messages: Sequence[Mapping[str, Any]] | None = None,
) -> RunInput:
    """Build a :class:`RunInput` from :meth:`Agent.invoke` kwargs.

    Centralised constructor so the Agent does not have to know the
    frozen-dataclass field shape.  Mappings are accepted by reference
    (``RunInput`` is frozen, so callers should pass immutable shapes
    or accept that the snapshot reflects the dict at call time);
    ``prior_messages`` is materialised into a tuple to make the
    snapshot fully hashable / safely shareable across replays.
    """
    return RunInput(
        user_message=user_message,
        function_name=function_name,
        episode_id=episode_id,
        tags=tags,
        system=system,
        system_arguments=system_arguments,
        prior_messages=tuple(prior_messages) if prior_messages else (),
    )
