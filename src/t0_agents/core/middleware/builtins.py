"""Six built-in middlewares that materialise the v0.12 cross-cutting concerns.

Each middleware reproduces a slice of behaviour that v0.12's
``Agent._execute_*_legacy`` orchestrated inline.  The default chain
materialised by :meth:`AgentConfig.__post_init__` is::

    Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal

`middleware[0]` (``Memory``) is outermost; ``JsonRetry`` is innermost
right before the terminal.  Each layer's responsibility:

- :class:`MemoryMiddleware` — load prior records, inject summary or
  ``prior_messages``, save the new record in ``finally``.
- :class:`SubAgentMiddleware` — wrap ``tool_provider`` with
  :class:`_OrchestrationToolProvider`, bind ``parent_episode_id``
  ContextVar, refresh it on the first inference when the orchestrator
  did not pre-allocate an episode.
- :class:`GuardrailsMiddleware` — wrap ``tool_provider`` with
  :class:`GuardedToolProvider`, fire ``before_run``/``after_run`` hooks
  (with ``T0AgentsFatalError`` propagation), populate
  ``result.guardrail_log``, register a cost-tracker into
  ``ctx.on_inference`` so cost-budget guardrails (``max_cost``) keep
  working.
- :class:`CostTrackingMiddleware` — always-on accumulator independent of
  guardrails; writes :class:`RunUsage` totals onto the live
  :class:`Run` so :meth:`Run.cost` / :meth:`Run.usage` read live data.
- :class:`JsonRetryMiddleware` — **M3 activated**.  Drives the
  parse-failure retry loop for single-shot output modes
  (:class:`JsonOutput`, :class:`StructuredOutput`).  Calls the terminal
  in a loop, inspects ``result.diagnostics.parse_strategy``, and on
  ``"none"`` consults ``ctx.output.recover(...)`` for retry / accept /
  fail.
- :class:`ObservabilityMiddleware` — wraps the run in
  ``tracer.start_as_current_span(f"agent.invoke/{name}")``; ``None``
  tracer falls back to a no-op span.

The order is load-bearing: ``SubAgentMiddleware`` must wrap
``tool_provider`` **inside** ``GuardrailsMiddleware`` so orchestrator-
level guardrails observe sub-agent delegations as tool calls.  The
default chain enforces this; custom chains that rearrange them must
preserve the wrapping invariant manually.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from t0_agents.core.exceptions import T0AgentsFatalError
from t0_agents.core.middleware.base import Next, RunContext
from t0_agents.core.output import (
    Attempt,
    OutputDiagnostics,
    ParseFailure,
)
from t0_agents.core.types import (
    AgentResult,
    Guardrail,
    GuardrailLogEntry,
    MemoryProvider,
    RunUsage,
    ToolProvider,
)
from t0_agents.guardrails.base import GuardrailContext
from t0_agents.guardrails.runtime import GuardedToolProvider
from t0_agents.memory.base import MemoryRecord

if TYPE_CHECKING:
    from t0_agents.core.agent import Agent


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryMiddleware
# ---------------------------------------------------------------------------


class MemoryMiddleware:
    """Hydrate prior context before the run, persist a record after.

    Dispatches on :attr:`AgentConfig.memory_strategy` (resolved via the
    agent's :meth:`_resolve_memory_strategy`):

    - ``"prepended_message"`` (default) — prepends a rendered summary
      of prior records to ``ctx.user_message``.
    - ``"system_template"`` — surfaces the summary under
      ``ctx.system_arguments["memory_context"]``.  Caller-explicit
      ``system_arguments`` win on key collisions.
    - ``"messages"`` — populates ``ctx.prior_messages`` via
      ``memory.load_history(episode_id)``.  Caller-explicit
      ``prior_messages`` (e.g. from :meth:`Run.fork`) take precedence
      and skip the load entirely.

    Save runs in ``finally`` so the record is persisted even on
    exception paths when the run produced a result; failures of either
    load or save are logged but never propagate (memory is a
    best-effort layer).

    The memory record stores the **original** user message
    (``ctx.run.input.user_message``), not the memory-injected effective
    one, so future ``load`` calls see clean prior turns.
    """

    name = "memory"

    def __init__(
        self,
        provider: MemoryProvider,
        strategy: str,
    ) -> None:
        self._provider = provider
        self._strategy = strategy

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        agent = ctx.run.agent if ctx.run is not None else None
        original_user_message = (
            ctx.run.input.user_message
            if ctx.run is not None
            else ctx.user_message
        )

        if agent is None:
            # Agent GC'd before middleware ran — should be impossible
            # under normal lifecycle (Run holds a weakref but the
            # caller of invoke owns the Agent strongly until the Run
            # resolves).  Fall through without hydration so we surface
            # the underlying failure at the terminal rather than masking
            # it with a memory-layer error.
            return await next_(ctx)

        strategy = agent._resolve_memory_strategy()

        # --- 1. Load prior records ---
        # ``messages`` strategy uses load_history (called from
        # _prepare_memory_injection); records are unused on that path,
        # so we skip the load() round-trip entirely.  Skip also when
        # episode_id is None — we never persist under an empty key.
        memory_records: list[MemoryRecord] = []
        if ctx.episode_id is not None and strategy != "messages":
            try:
                raw_records = await self._provider.load(ctx.episode_id)
                memory_records = cast(list[MemoryRecord], raw_records)
            except Exception:
                logger.warning(
                    "Memory load failed for episode %s; proceeding without context",
                    ctx.episode_id,
                    exc_info=True,
                )

        # --- 2. Prepare injection ---
        # Reuse the existing helper on Agent so middleware behaviour
        # tracks any future refinement to the strategy dispatch
        # without two places drifting apart.
        effective_message, extra_kwargs = await agent._prepare_memory_injection(
            original_user_message, memory_records, episode_id=ctx.episode_id,
        )
        ctx.user_message = effective_message

        # Merge memory-derived system_arguments under caller-provided ones —
        # caller keys win on collision (parity with the v0.12
        # ``_merge_caller_system_arguments`` precedence).
        mem_sys_args = extra_kwargs.get("system_arguments")
        if mem_sys_args:
            merged = dict(mem_sys_args)
            if ctx.system_arguments:
                merged.update(ctx.system_arguments)
            ctx.system_arguments = merged or None

        # Caller-explicit prior_messages (Run.fork) win over the
        # memory-strategy-derived history: fork is by design the user's
        # explicit intent to continue from this exact transcript.
        if ctx.prior_messages is None:
            mem_prior = extra_kwargs.get("prior_messages")
            if mem_prior:
                ctx.prior_messages = list(mem_prior)

        # --- 3. Call next + save in finally ---
        result: AgentResult[Any] | None = None
        try:
            result = await next_(ctx)
        finally:
            # Skip when episode_id is missing so we never persist under
            # an empty-string key (which would collide across runs and
            # be unretrievable).
            if result is not None and result.episode_id:
                record = MemoryRecord(
                    episode_id=result.episode_id,
                    user_message=original_user_message,
                    agent_response=result.response,
                    turns=result.turns,
                    inference_ids=list(result.inference_ids),
                )
                try:
                    await self._provider.save(record)
                except Exception:
                    logger.warning(
                        "Memory save failed for episode %s; run result is still valid",
                        result.episode_id,
                        exc_info=True,
                    )

        assert result is not None
        return result


# ---------------------------------------------------------------------------
# GuardrailsMiddleware
# ---------------------------------------------------------------------------


class GuardrailsMiddleware:
    """Run-level + tool-level policy hooks; cost-budget accumulator.

    For each invocation:

    1. Construct a fresh :class:`GuardrailContext` keyed on the
       **original** ``user_message`` (so guardrails see the user's
       words, not the memory-prepended effective message).
    2. Wrap ``ctx.tool_provider`` with :class:`GuardedToolProvider` so
       every ``call_tool`` fires the ``before_tool`` / ``after_tool``
       hooks and tracks ``tool_calls_made`` on the context.
    3. Register a cost-tracker on ``ctx.on_inference`` that updates
       ``guard_ctx.cost_so_far`` and ``guard_ctx.turn`` per inference —
       this is what cost-budget guardrails (``max_cost``) read.
    4. Fire every ``before_run`` hook; ``T0AgentsFatalError`` aborts
       the run immediately, non-fatal exceptions are logged and the
       chain continues.
    5. Call ``next_(ctx)``.  In ``finally``, fire every ``after_run``
       hook (same fatal/non-fatal split) and attach the populated
       ``guardrail_log`` to the result.  The ``after_run`` block runs
       even when the inner chain raised — the guardrail subsystem's
       contract is "fires once around every run" regardless of outcome.

    For JSON runs the tool-provider wrap is a no-op (JSON functions do
    not dispatch tools), but the run-level hooks still fire and the
    cost-tracker still accumulates across retries.
    """

    name = "guardrails"

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._guardrails = guardrails

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        original_user_message = (
            ctx.run.input.user_message
            if ctx.run is not None
            else ctx.user_message
        )

        guard_ctx = GuardrailContext(
            agent_name=ctx.config.name,
            function_name=ctx.config.function_name,
            user_message=original_user_message,
        )
        guard_log: list[GuardrailLogEntry] = []

        # Wrap the tool provider so per-tool hooks fire on every call.
        # JSON agents have no tool_provider (or an empty one); the
        # GuardedToolProvider is still constructed but never invoked.
        if ctx.tool_provider is not None:
            ctx.replace_tool_provider(
                cast(
                    ToolProvider,
                    GuardedToolProvider(
                        ctx.tool_provider,
                        list(self._guardrails),
                        guard_ctx,
                        guard_log,
                    ),
                ),
            )

        # Compose the cost-tracker into ctx.on_inference so
        # cost-budget guardrails see usage per inference.
        async def _cost_tracker(response: Any) -> None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                guard_ctx.cost_so_far += getattr(usage, "cost", 0.0) or 0.0
            guard_ctx.turn += 1

        ctx.chain_on_inference(_cost_tracker)

        # --- before_run + run + after_run ---
        # ``before_run`` lives INSIDE the try block so a fatal abort
        # (T0AgentsFatalError raised from a hook) still fires
        # ``after_run`` via the finally clause — the docs/guardrails.md
        # contract promises ``after_run`` fires once around every run,
        # success or fatal abort.
        result: AgentResult[Any] | None = None
        try:
            for guardrail in self._guardrails:
                try:
                    await guardrail.before_run(guard_ctx)
                except T0AgentsFatalError:
                    # Hard-abort propagates past the broad except.
                    # ``after_run`` still fires via the outer finally.
                    raise
                except Exception:
                    logger.exception(
                        "Guardrail %r before_run raised a non-fatal exception; "
                        "continuing",
                        guardrail.name,
                    )
                else:
                    guard_log.append(
                        GuardrailLogEntry(
                            guardrail_name=guardrail.name,
                            hook="before_run",
                            kind="pass",
                        ),
                    )

            result = await next_(ctx)
        finally:
            # --- after_run --- fires even when before_run or next_ raised.
            for guardrail in self._guardrails:
                try:
                    await guardrail.after_run(guard_ctx, result)
                except T0AgentsFatalError:
                    raise
                except Exception:
                    logger.exception(
                        "Guardrail %r after_run raised a non-fatal exception; "
                        "continuing",
                        guardrail.name,
                    )
                else:
                    guard_log.append(
                        GuardrailLogEntry(
                            guardrail_name=guardrail.name,
                            hook="after_run",
                            kind="pass",
                        ),
                    )
            if result is not None:
                result.guardrail_log = list(guard_log)

        assert result is not None
        return result


# ---------------------------------------------------------------------------
# CostTrackingMiddleware
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CumulativeUsage:
    """Per-run accumulator stored on ``ctx.extras['cumulative_usage']``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class CostTrackingMiddleware:
    """Always-on accumulator wiring :meth:`Run.cost` / :meth:`Run.usage`.

    v0.12 only tracked cost when guardrails were configured (the
    tracker was attached on the ``GuardrailContext``).  v1.0 makes cost
    + usage accumulation always-on so :meth:`Run.cost` and
    :meth:`Run.usage` return live data for every Run — a strict
    superset of v0.12 behaviour.

    The accumulator lives on ``ctx.extras['cumulative_usage']`` so
    other middlewares (or user observability hooks) can read it
    mid-run.  On the way out (``finally`` after ``next_()``) the
    middleware snapshots the totals onto the live
    :class:`~t0_agents.core.run.Run` via the private
    :meth:`_set_cost` / :meth:`_set_usage` setters.
    """

    name = "cost_tracking"

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        usage_acc = _CumulativeUsage()
        ctx.extras["cumulative_usage"] = usage_acc

        async def _tracker(response: Any) -> None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            usage_acc.input_tokens += getattr(usage, "input_tokens", 0) or 0
            usage_acc.output_tokens += getattr(usage, "output_tokens", 0) or 0
            usage_acc.cost += getattr(usage, "cost", 0.0) or 0.0

        ctx.chain_on_inference(_tracker)

        result: AgentResult[Any] | None = None
        try:
            result = await next_(ctx)
            return result
        finally:
            # Always flush totals to Run, even on exception paths — gives
            # observers (e.g. ``await run.cost()`` after a failure) an
            # accurate cumulative count.  Skip when the Run weakref is
            # gone (caller dropped the Agent).
            if ctx.run is not None:
                snapshot = RunUsage(
                    input_tokens=usage_acc.input_tokens,
                    output_tokens=usage_acc.output_tokens,
                    cost=usage_acc.cost,
                )
                ctx.run._set_cost(usage_acc.cost)
                ctx.run._set_usage(snapshot)


# ---------------------------------------------------------------------------
# SubAgentMiddleware
# ---------------------------------------------------------------------------


class SubAgentMiddleware:
    """Wrap ``tool_provider`` for inline sub-agent dispatch.

    Mirror of v0.12's inline orchestrator setup
    (``_execute_chat_legacy:664-680, 728-759``):

    1. Wrap ``ctx.tool_provider`` with
       :class:`_OrchestrationToolProvider` so sub-agent tool calls
       dispatch in-process via the cached ``tool_name → Agent`` map.
    2. Bind ``parent_episode_id`` ContextVar to the orchestrator's
       ``episode_id`` so sub-agent inferences tag
       ``t0_agents.parent_episode_id=<orchestrator-id>`` for
       ClickHouse correlation.
    3. When the orchestrator did not pre-allocate an episode
       (``episode_id is None`` at entry), compose an ``on_inference``
       hook that refreshes the ContextVar from the first inference's
       ``response.episode_id``.  Without this refresh every sub-agent
       call within the orchestrator's first turn would tag
       ``parent_episode_id=unknown``.

    Order invariant: must run **inside** :class:`GuardrailsMiddleware`
    in the chain so orchestrator-level guardrails see sub-agent
    delegations as tool calls.  Enforced by the default chain order
    (``Guardrails`` registers before ``SubAgents``).
    """

    name = "sub_agents"

    def __init__(self, sub_agents: list[Agent]) -> None:
        self._sub_agents = sub_agents

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        from t0_agents.compose.orchestrator import (
            _OrchestrationToolProvider,
            bind_parent_episode_id,
            update_parent_episode_id,
        )

        agent = ctx.run.agent if ctx.run is not None else None
        if agent is None or ctx.tool_provider is None:
            # No live agent (GC'd) or no tool provider — SubAgent
            # orchestration is impossible.  Fall through to surface the
            # real failure at the terminal.
            return await next_(ctx)

        sub_agents_by_tool = agent._sub_agents_by_tool_name
        ctx.replace_tool_provider(
            cast(
                ToolProvider,
                _OrchestrationToolProvider(ctx.tool_provider, sub_agents_by_tool),
            ),
        )

        captured_episode_id = ctx.episode_id

        # Compose a refresh hook into ctx.on_inference when the
        # orchestrator did not pre-allocate an episode — the first
        # inference populates ``response.episode_id`` and the
        # ContextVar refresh ensures every sub-agent call dispatched
        # mid-turn correlates against the orchestrator's real id.
        if captured_episode_id is None:

            async def _refresh_parent_episode_id(response: Any) -> None:
                episode = getattr(response, "episode_id", None)
                if episode:
                    update_parent_episode_id(str(episode))

            ctx.chain_on_inference(_refresh_parent_episode_id)

        with bind_parent_episode_id(captured_episode_id):
            return await next_(ctx)


# ---------------------------------------------------------------------------
# JsonRetryMiddleware (M2 placeholder)
# ---------------------------------------------------------------------------


class JsonRetryMiddleware:
    """Drives the parse-failure retry loop for single-shot output modes.

    Placed at the innermost position in the default chain (Q13 resolution)
    so memory, guardrails, and cost tracking see only the final
    successful inference rather than every retry attempt.  Audit of
    retries surfaces via ``result.diagnostics.attempts`` and TensorZero
    tags.

    Chat agents bypass the retry logic entirely
    (``ctx.output.expects_tool_loop == True`` short-circuits to a single
    ``next_(ctx)`` call).  Single-shot modes drive through the loop:

    1. Seed ``ctx.extras["_output_attempts"]`` with the attempts list so
       the single-shot terminal can pass it to
       :meth:`OutputMode.materialize`.
    2. Call ``next_(ctx)`` — the terminal materialises a result.
    3. If ``result.diagnostics.parse_strategy != "none"`` → success;
       return.
    4. Otherwise build a :class:`ParseFailure`, append an
       :class:`Attempt` to the buffer, and call
       :meth:`ctx.output.recover` to decide what to do next.
       - :meth:`Recovery.fail` → return the failed result as-is so v0.12
         byte-identity is preserved (``parsed=None``,
         ``parse_strategy="none"`` flow through the
         :attr:`AgentResult.parsed` property).
       - :meth:`Recovery.accept` → synthesise a new :class:`AgentResult`
         carrying the coerced value with ``parse_strategy="permissive"``.
       - :meth:`Recovery.retry` → append the prior user/assistant turn
         pair to ``ctx.prior_messages``, set ``ctx.user_message`` to
         the corrective hint, and loop.

    This middleware needs to call ``next_(ctx)`` more than once per
    invocation.  The ``allow_retry`` class attribute signals
    :func:`build_pipeline` to skip the :class:`_NextGuard` for this
    layer so re-invocation does not raise.
    """

    name = "json_retry"
    allow_retry = True

    async def __call__(
        self, ctx: RunContext, next_: Next
    ) -> AgentResult[Any]:
        if ctx.output is None or ctx.output.expects_tool_loop:
            # Chat / tool-loop path — no retry orchestration, single
            # call through.
            return await next_(ctx)

        attempts: list[Attempt] = []
        ctx.extras["_output_attempts"] = attempts

        while True:
            result = await next_(ctx)
            # Latch the episode_id from the first response so retries
            # reuse it (mirrors v0.12 ``_json_terminal`` semantics —
            # the whole retry chain belongs to one episode).
            if ctx.episode_id is None and result.episode_id is not None:
                ctx.episode_id = result.episode_id
            diag = result.diagnostics
            if diag is None or diag.parse_strategy != "none":
                # Success path: terminal materialised a valid result.
                return result

            failure = ParseFailure(
                raw_text=diag.raw_text or "",
                inference_id=(
                    result.inference_ids[-1]
                    if result.inference_ids
                    else ""
                ),
                reason=(
                    "validation_failed"
                    if diag.failure_detail is not None
                    else "parse_failed"
                ),
                detail=(
                    diag.failure_detail
                    if diag.failure_detail is not None
                    else "JSON parsing failed after permissive attempt"
                ),
            )
            attempts.append(
                Attempt(
                    inference_id=failure.inference_id,
                    raw_text=failure.raw_text,
                    parsed=None,
                    usage=None,
                ),
            )
            recovery = await ctx.output.recover(failure, attempts)

            if recovery.kind == "fail":
                # Preserve v0.12 byte-identity: failed JSON parse
                # returns the failed AgentResult with parsed=None and
                # parse_strategy="none".  Stamp the final attempts
                # count on the diagnostics.
                result.diagnostics = OutputDiagnostics(
                    parse_strategy="none",
                    attempts=len(attempts) - 1,
                    raw_text=diag.raw_text,
                    failure_detail=diag.failure_detail,
                )
                return result

            if recovery.kind == "accept":
                return AgentResult(
                    value=recovery.value,
                    episode_id=result.episode_id,
                    inference_ids=list(result.inference_ids),
                    turns=result.turns,
                    diagnostics=OutputDiagnostics(
                        parse_strategy="permissive",
                        # ``attempts`` includes the just-failed attempt;
                        # ``json_parse_attempts`` counts EXTRA inferences
                        # past the first, so subtract one to match the
                        # fail branch (line 595) and the v0.12 semantics.
                        attempts=len(attempts) - 1,
                        raw_text=diag.raw_text,
                    ),
                )

            # recovery.kind == "retry"
            hint = recovery.hint or ""
            prior = list(ctx.prior_messages or [])
            prior.append(
                {"role": "user", "content": ctx.user_message},
            )
            prior.append(
                {"role": "assistant", "content": failure.raw_text},
            )
            ctx.prior_messages = prior
            ctx.user_message = hint
            # Loop: re-call next_(ctx).


# ---------------------------------------------------------------------------
# ObservabilityMiddleware
# ---------------------------------------------------------------------------


class ObservabilityMiddleware:
    """Span-wrap the run for tracing backends.

    Wraps ``next_(ctx)`` in ``tracer.start_as_current_span(name)`` where
    ``name = f"agent.invoke/{ctx.config.name}"``.  Accepts any tracer
    that exposes ``start_as_current_span`` — OpenTelemetry's
    ``trace.Tracer``, Datadog's ``ddtrace.Tracer``, and any custom
    structural-typed tracer all work.

    Not in the default chain — opt-in only.  Pass it explicitly via
    ``AgentConfig(middleware=[ObservabilityMiddleware(my_tracer), ...])``
    or include it in a custom chain composed with
    :meth:`AgentConfig.default_middleware_for`.

    When ``tracer is None`` (the default) the middleware uses
    :class:`contextlib.nullcontext` so the chain stays callable for
    tests and for users who include the middleware in their default
    list without configuring tracing.
    """

    name = "observability"

    def __init__(self, tracer: Any = None) -> None:
        self._tracer = tracer

    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult[Any]:
        span_name = f"agent.invoke/{ctx.config.name}"
        if self._tracer is None:
            cm: Any = contextlib.nullcontext()
        else:
            cm = self._tracer.start_as_current_span(span_name)
        with cm:
            return await next_(ctx)


__all__ = [
    "CostTrackingMiddleware",
    "GuardrailsMiddleware",
    "JsonRetryMiddleware",
    "MemoryMiddleware",
    "ObservabilityMiddleware",
    "SubAgentMiddleware",
]
