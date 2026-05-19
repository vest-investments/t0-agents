"""Topology driver coroutines for :mod:`t0_agents.compose.builtins`.

One private ``_drive_*`` function per built-in topology.  Each driver:

1. Pulls the ambient ``parent_episode_id`` from
   :func:`~t0_agents.delegation.orchestrator.get_parent_episode_id`, so
   nested composers inherit correlation transparently.
2. Invokes each child :class:`Composable` and appends the resulting
   :class:`Run` to ``composite._sub_runs`` for cancellation cascade.
3. Tags Agent children via the TZ ``tags={"t0_agents.parent_episode_id":
   ...}`` kwarg; binds the ContextVar for Composer children (which
   don't accept ``tags=`` — the narrow ``Composer.invoke`` surface
   resolution).
4. Aggregates ``inference_ids`` (and ``turns``, and the first
   ``episode_id``) into a composite :class:`AgentResult`.

Errors that come out of a sub-Run are wrapped in :class:`ComposerError`
with the failing child as ``node``, except in
:func:`_drive_retry_chain` where the policy is to re-raise unchanged so
callers can ``except SpecificError``.

Reducer / merge / router callbacks that raise are also wrapped in
:class:`ComposerError`, with ``node`` set to the composer itself
(``composer.children[0]`` is wrong — it's the *user code* on the
composer instance that failed).  The dev-log records the rationale.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal

from t0_agents.compose.topology import (
    Composable,
    Composer,
    ComposerError,
    CompositeRun,
)
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.delegation.orchestrator import (
    PARENT_EPISODE_ID_TAG,
    bind_parent_episode_id,
    get_parent_episode_id,
    update_parent_episode_id,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _default_sequential_chain(
    prev_result: AgentResult[Any], original_input: Any
) -> Any:
    """Default ``chain`` for :func:`sequential` and :func:`saga`.

    Coerces non-string values to JSON so the next Agent (whose
    ``user_message`` must be ``str``) sees a deterministic textual
    representation.  ``default=str`` keeps non-JSON-serialisable
    objects from breaking the chain — they round-trip through ``str()``
    instead.  ``original_input`` is unused in the default; custom
    chains take both for richer routing.
    """
    del original_input  # default ignores the original; custom chains use it
    if isinstance(prev_result.value, str):
        return prev_result.value
    return json.dumps(prev_result.value, default=str)


def _invoke_with_parent(
    child: Composable[Any, Any],
    input_value: Any,
    parent_episode_id: str | None,
) -> Run[Any]:
    """Invoke a child Composable, threading ``parent_episode_id`` correctly.

    Composers don't accept a ``tags=`` kwarg (narrow ``invoke`` surface,
    resolved 2026-05-16) so we bind ``parent_episode_id`` via the
    :class:`~contextvars.ContextVar` for nested composers — their own
    driver will read it via :func:`get_parent_episode_id` and propagate
    onward.  Agents (and any other Composable that accepts
    ``tags=``) get an explicit TZ tag so their first inference logs
    the correlation directly.

    The ContextVar binding survives the ``with`` block because
    :meth:`asyncio.create_task` captures the current ``contextvars``
    Context at task-creation time — so the child Composer's drive
    task still sees the binding even after :func:`_invoke_with_parent`
    returns.
    """
    if isinstance(child, Composer):
        if parent_episode_id is not None:
            with bind_parent_episode_id(parent_episode_id):
                return child.invoke(input_value)
        return child.invoke(input_value)
    tags: dict[str, str] | None = (
        {PARENT_EPISODE_ID_TAG: parent_episode_id}
        if parent_episode_id
        else None
    )
    # mypy: arbitrary Composable.invoke accepts **kwargs: Any — the
    # narrow type erasure is intentional, every concrete callee
    # (Agent.invoke / Composer.invoke) accepts the ``tags=`` kwarg or
    # rejects it cleanly at the Composer branch above.
    return child.invoke(input_value, tags=tags)


def _build_composite_result(
    *,
    value: Any,
    last_result: AgentResult[Any] | None,
    episode_id: str | None,
    inference_ids: list[str],
    turns: int,
) -> AgentResult[Any]:
    """Assemble the composite :class:`AgentResult` returned by a driver.

    Inherits ``diagnostics`` from the last successful sub-result so the
    v0.12 read-side properties (``response``, ``parsed``, etc.) reflect
    the final step.  ``inference_ids`` is the union of every sub-Run's
    ids in invocation order; ``turns`` sums every sub-Run's turn count.
    """
    diagnostics = last_result.diagnostics if last_result is not None else None
    return AgentResult(
        value=value,
        episode_id=episode_id,
        inference_ids=inference_ids,
        turns=turns,
        diagnostics=diagnostics,
    )


# ----------------------------------------------------------------------
# Drivers — multi-child topologies
# ----------------------------------------------------------------------


async def _drive_sequential(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    children: tuple[Composable[Any, Any], ...],
    chain: Callable[[AgentResult[Any], Any], Any],
) -> AgentResult[Any]:
    del composer  # unused in sequential; reserved for ComposerError fallback
    if not children:
        raise ValueError("sequential() requires at least one child")
    ambient_parent = get_parent_episode_id()
    root_episode_id: str | None = None
    accumulated_ids: list[str] = []
    turns: int = 0
    current: Any = input_value
    last_result: AgentResult[Any] | None = None

    with bind_parent_episode_id(ambient_parent):
        for i, child in enumerate(children):
            parent_for_this = root_episode_id or ambient_parent
            sub_run = _invoke_with_parent(child, current, parent_for_this)
            composite._sub_runs.append(sub_run)
            try:
                sub_result = await sub_run.result()
            except BaseException as exc:
                raise ComposerError(child, exc) from exc
            accumulated_ids.extend(sub_run.inference_ids)
            turns += sub_result.turns
            if root_episode_id is None and sub_run.episode_id:
                root_episode_id = sub_run.episode_id
                update_parent_episode_id(root_episode_id)
            last_result = sub_result
            # Compute chained input only if there is a next step.
            if i < len(children) - 1:
                current = chain(sub_result, input_value)

    final_value: Any = last_result.value if last_result is not None else current
    return _build_composite_result(
        value=final_value,
        last_result=last_result,
        episode_id=root_episode_id,
        inference_ids=accumulated_ids,
        turns=turns,
    )


async def _drive_parallel(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    children: tuple[Composable[Any, Any], ...],
    reducer: Callable[[Any], Any] | None,
    error_policy: Literal["fail_fast", "continue"],
) -> AgentResult[Any]:
    if not children:
        raise ValueError("parallel() requires at least one child")
    ambient_parent = get_parent_episode_id()

    with bind_parent_episode_id(ambient_parent):
        sub_runs = [
            _invoke_with_parent(child, input_value, ambient_parent)
            for child in children
        ]
        composite._sub_runs.extend(sub_runs)

    successful: list[AgentResult[Any]]
    for_reducer: Any
    if error_policy == "fail_fast":
        # True fail-fast: ``asyncio.wait(..., return_when=FIRST_EXCEPTION)``
        # returns as soon as any sub-Run raises (or all succeed), so we
        # don't wait for the slow ones.  When a failure surfaces we
        # cancel the still-pending sibling result tasks AND the
        # underlying sub-Runs (via ``composite._cancel_sub_runs`` under
        # the bounded ``cancel_timeout``).  Drain the lot at the end
        # to suppress "Task exception was never retrieved" warnings.
        tasks: list[asyncio.Task[AgentResult[Any]]] = [
            asyncio.create_task(r.result()) for r in sub_runs
        ]
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION,
        )
        failed_task: asyncio.Task[AgentResult[Any]] | None = next(
            (t for t in done if t.exception() is not None), None,
        )
        if failed_task is not None:
            for t in pending:
                t.cancel()
            await composite._cancel_sub_runs()
            await asyncio.gather(*tasks, return_exceptions=True)
            failed_idx = tasks.index(failed_task)
            failed_child = children[failed_idx]
            exc = failed_task.exception()
            assert exc is not None
            raise ComposerError(failed_child, exc) from exc
        # ``FIRST_EXCEPTION`` collapses to ``ALL_COMPLETED`` when nothing
        # raised — every task is in ``done`` with a successful result.
        successful = [t.result() for t in tasks]
        for_reducer = successful
    else:  # continue
        gathered: list[AgentResult[Any] | BaseException] = await asyncio.gather(
            *(r.result() for r in sub_runs),
            return_exceptions=True,
        )
        successful = [r for r in gathered if not isinstance(r, BaseException)]
        for_reducer = gathered

    if reducer is None:
        value: Any = [r.value for r in successful]
    else:
        try:
            value = reducer(for_reducer)
        except BaseException as exc:
            raise ComposerError(composer, exc) from exc

    accumulated_ids: list[str] = []
    episode_id: str | None = None
    for sr in sub_runs:
        accumulated_ids.extend(sr.inference_ids)
        if episode_id is None and sr.episode_id:
            episode_id = sr.episode_id
    turns = sum(r.turns for r in successful)

    return _build_composite_result(
        value=value,
        last_result=successful[-1] if successful else None,
        episode_id=episode_id,
        inference_ids=accumulated_ids,
        turns=turns,
    )


async def _drive_race(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    children: tuple[Composable[Any, Any], ...],
    cancel_others: bool,
    cancel_timeout: float,
) -> AgentResult[Any]:
    del composer  # unused: race never wraps via composer self
    if not children:
        raise ValueError("race() requires at least one child")
    ambient_parent = get_parent_episode_id()

    with bind_parent_episode_id(ambient_parent):
        sub_runs = [
            _invoke_with_parent(child, input_value, ambient_parent)
            for child in children
        ]
        composite._sub_runs.extend(sub_runs)
        # Wrap each ``await result()`` in a Task so we can wait on the
        # first to settle.  ``return_when=FIRST_COMPLETED`` treats
        # success and failure equally — the winner is the first to
        # produce or raise.
        tasks: list[asyncio.Task[AgentResult[Any]]] = [
            asyncio.create_task(r.result()) for r in sub_runs
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

    # Detach the pending result() awaitables so they don't leak.
    for t in pending:
        t.cancel()

    if cancel_others and pending:
        losing_runs = [
            r for r, t in zip(sub_runs, tasks, strict=True)
            if t in pending
        ]
        if losing_runs:
            try:
                async with asyncio.timeout(cancel_timeout):
                    await asyncio.gather(
                        *(r.cancel() for r in losing_runs),
                        return_exceptions=True,
                    )
            except TimeoutError:
                logger.warning(
                    "race() loser cancel exceeded %.1fs; %d sub-Runs may leak",
                    cancel_timeout, len(losing_runs),
                )

    winner_task = next(iter(done))
    winner_idx = tasks.index(winner_task)
    winner_run = sub_runs[winner_idx]
    winner_child = children[winner_idx]

    try:
        winner_result: AgentResult[Any] = winner_task.result()
    except BaseException as exc:
        raise ComposerError(winner_child, exc) from exc

    return _build_composite_result(
        value=winner_result.value,
        last_result=winner_result,
        episode_id=winner_result.episode_id,
        inference_ids=list(winner_run.inference_ids),
        turns=winner_result.turns,
    )


async def _drive_consensus(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    children: tuple[Composable[Any, Any], ...],
    merge: Callable[[list[Any]], Any],
) -> AgentResult[Any]:
    if not children:
        raise ValueError("consensus() requires at least one child")
    ambient_parent = get_parent_episode_id()

    with bind_parent_episode_id(ambient_parent):
        sub_runs = [
            _invoke_with_parent(child, input_value, ambient_parent)
            for child in children
        ]
        composite._sub_runs.extend(sub_runs)

    # Consensus has no ``continue`` policy — any sub-Run failure aborts
    # the vote.  ``FIRST_EXCEPTION`` lets us cancel siblings the moment
    # one proposer raises rather than waiting for slow voters to settle.
    tasks: list[asyncio.Task[AgentResult[Any]]] = [
        asyncio.create_task(r.result()) for r in sub_runs
    ]
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_EXCEPTION,
    )
    failed_task: asyncio.Task[AgentResult[Any]] | None = next(
        (t for t in done if t.exception() is not None), None,
    )
    if failed_task is not None:
        for t in pending:
            t.cancel()
        await composite._cancel_sub_runs()
        await asyncio.gather(*tasks, return_exceptions=True)
        failed_idx = tasks.index(failed_task)
        failed_child = children[failed_idx]
        exc = failed_task.exception()
        assert exc is not None
        raise ComposerError(failed_child, exc) from exc

    results: list[AgentResult[Any]] = [t.result() for t in tasks]
    values: list[Any] = [r.value for r in results]

    try:
        merged = merge(values)
    except BaseException as exc:
        raise ComposerError(composer, exc) from exc

    accumulated_ids: list[str] = []
    episode_id: str | None = None
    for sr in sub_runs:
        accumulated_ids.extend(sr.inference_ids)
        if episode_id is None and sr.episode_id:
            episode_id = sr.episode_id
    turns = sum(r.turns for r in results)

    return _build_composite_result(
        value=merged,
        last_result=results[-1] if results else None,
        episode_id=episode_id,
        inference_ids=accumulated_ids,
        turns=turns,
    )


async def _drive_map_reduce(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    child: Composable[Any, Any],
    reducer: Callable[[list[AgentResult[Any]]], Any],
    concurrency: int | None,
) -> AgentResult[Any]:
    if not isinstance(input_value, Iterable) or isinstance(input_value, (str, bytes)):
        raise TypeError(
            f"map_reduce() requires an iterable input "
            f"(list, tuple, generator), got {type(input_value).__name__}"
        )
    items: list[Any] = list(input_value)
    ambient_parent = get_parent_episode_id()
    semaphore = (
        asyncio.Semaphore(concurrency)
        if concurrency is not None and concurrency > 0
        else None
    )

    async def _run_one(item: Any) -> AgentResult[Any]:
        if semaphore is not None:
            async with semaphore:
                return await _invoke_and_await(item)
        return await _invoke_and_await(item)

    async def _invoke_and_await(item: Any) -> AgentResult[Any]:
        sub_run = _invoke_with_parent(child, item, ambient_parent)
        composite._sub_runs.append(sub_run)
        return await sub_run.result()

    if not items:
        try:
            value = reducer([])
        except BaseException as exc:
            raise ComposerError(composer, exc) from exc
        return _build_composite_result(
            value=value,
            last_result=None,
            episode_id=None,
            inference_ids=[],
            turns=0,
        )

    with bind_parent_episode_id(ambient_parent):
        try:
            results: list[AgentResult[Any]] = await asyncio.gather(
                *(_run_one(item) for item in items)
            )
        except BaseException as exc:
            await composite._cancel_sub_runs()
            raise ComposerError(child, exc) from exc

    try:
        value = reducer(results)
    except BaseException as exc:
        raise ComposerError(composer, exc) from exc

    accumulated_ids: list[str] = []
    episode_id: str | None = None
    # ``composite._sub_runs`` was appended in invocation order — but
    # gather completes out of order; use ``_sub_runs`` for id
    # aggregation so the order reflects scheduling rather than
    # completion.
    for sr in composite._sub_runs:
        accumulated_ids.extend(sr.inference_ids)
        if episode_id is None and sr.episode_id:
            episode_id = sr.episode_id
    turns = sum(r.turns for r in results)

    return _build_composite_result(
        value=value,
        last_result=results[-1] if results else None,
        episode_id=episode_id,
        inference_ids=accumulated_ids,
        turns=turns,
    )


async def _drive_conditional(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    router: Callable[[Any], str],
    routes: dict[str, Composable[Any, Any]],
    default: Composable[Any, Any] | None,
) -> AgentResult[Any]:
    try:
        route_key = router(input_value)
    except BaseException as exc:
        raise ComposerError(composer, exc) from exc

    target: Composable[Any, Any] | None = routes.get(route_key)
    if target is None:
        target = default
    if target is None:
        raise ComposerError(
            composer,
            ValueError(
                f"conditional: no route for {route_key!r} "
                f"(known routes: {sorted(routes.keys())!r}) and no default"
            ),
        )

    ambient_parent = get_parent_episode_id()
    sub_run = _invoke_with_parent(target, input_value, ambient_parent)
    composite._sub_runs.append(sub_run)
    try:
        result: AgentResult[Any] = await sub_run.result()
    except BaseException as exc:
        raise ComposerError(target, exc) from exc

    return _build_composite_result(
        value=result.value,
        last_result=result,
        episode_id=result.episode_id,
        inference_ids=list(sub_run.inference_ids),
        turns=result.turns,
    )


async def _drive_saga(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    steps: tuple[tuple[Composable[Any, Any], Composable[Any, Any] | None], ...],
) -> AgentResult[Any]:
    del composer  # unused; saga wraps via the failing action
    if not steps:
        raise ValueError("saga() requires at least one step")
    ambient_parent = get_parent_episode_id()
    root_episode_id: str | None = None
    accumulated_ids: list[str] = []
    turns: int = 0
    current: Any = input_value
    completed: list[
        tuple[Composable[Any, Any] | None, Any]
    ] = []  # (compensate, action_input)
    last_result: AgentResult[Any] | None = None

    with bind_parent_episode_id(ambient_parent):
        for i, (action, compensate) in enumerate(steps):
            parent_for_this = root_episode_id or ambient_parent
            action_input = current
            sub_run = _invoke_with_parent(action, action_input, parent_for_this)
            composite._sub_runs.append(sub_run)
            try:
                sub_result = await sub_run.result()
            except BaseException as exc:
                await _run_compensations(
                    composite,
                    completed,
                    root_episode_id or ambient_parent,
                )
                raise ComposerError(action, exc) from exc
            completed.append((compensate, action_input))
            accumulated_ids.extend(sub_run.inference_ids)
            turns += sub_result.turns
            if root_episode_id is None and sub_run.episode_id:
                root_episode_id = sub_run.episode_id
                update_parent_episode_id(root_episode_id)
            last_result = sub_result
            # Chain only when another step follows.
            if i < len(steps) - 1:
                current = _default_sequential_chain(sub_result, input_value)

    final_value: Any = last_result.value if last_result is not None else input_value
    return _build_composite_result(
        value=final_value,
        last_result=last_result,
        episode_id=root_episode_id,
        inference_ids=accumulated_ids,
        turns=turns,
    )


async def _run_compensations(
    composite: CompositeRun,
    completed: list[tuple[Composable[Any, Any] | None, Any]],
    parent_episode_id: str | None,
) -> None:
    """Run compensations in reverse for completed saga actions.

    Each compensation is best-effort: a raising compensate is logged
    but the cascade continues so earlier steps still get their chance
    to roll back.  ``None`` compensations are skipped silently.

    Compensations are themselves invoked through
    :func:`_invoke_with_parent` so they inherit the same correlation
    as the actions — useful for ClickHouse audit when reconstructing
    a saga that aborted.
    """
    for compensate, action_input in reversed(completed):
        if compensate is None:
            continue
        try:
            sub_run = _invoke_with_parent(
                compensate, action_input, parent_episode_id,
            )
            composite._sub_runs.append(sub_run)
            await sub_run.result()
        except BaseException as exc:
            logger.warning(
                "saga compensation %r failed: %r — continuing rollback",
                compensate, exc,
            )


async def _drive_retry_chain(
    composite: CompositeRun,
    composer: Composer,
    input_value: Any,
    children: tuple[Composable[Any, Any], ...],
    on_error: Callable[[BaseException], bool] | None,
) -> AgentResult[Any]:
    del composer  # retry_chain re-raises the original exception un-wrapped
    if not children:
        raise ValueError("retry_chain() requires at least one child")
    ambient_parent = get_parent_episode_id()
    accumulated_ids: list[str] = []
    last_exception: BaseException | None = None

    with bind_parent_episode_id(ambient_parent):
        for child in children:
            sub_run = _invoke_with_parent(child, input_value, ambient_parent)
            composite._sub_runs.append(sub_run)
            try:
                result: AgentResult[Any] = await sub_run.result()
            except BaseException as exc:
                last_exception = exc
                accumulated_ids.extend(sub_run.inference_ids)
                if on_error is not None and not on_error(exc):
                    raise
                continue
            accumulated_ids.extend(sub_run.inference_ids)
            return _build_composite_result(
                value=result.value,
                last_result=result,
                episode_id=result.episode_id,
                inference_ids=accumulated_ids,
                turns=result.turns,
            )

    assert last_exception is not None
    raise last_exception


async def _drive_circuit_breaker(
    composite: CompositeRun,
    composer: Any,
    input_value: Any,
    child: Composable[Any, Any],
) -> AgentResult[Any]:
    """Drive the circuit_breaker state machine.

    State lives on the ``composer`` instance (``_state``,
    ``_consecutive_failures``, ``_opened_at``) — per the Q10 spec
    recommendation, state is per-Composer-instance, not per-call.

    Transitions:

    - ``closed`` → call child.  Success resets the failure counter;
      failure increments it.  Counter reaching ``threshold`` opens
      the breaker.
    - ``open`` → if ``time.monotonic() - _opened_at >= reset_timeout``
      we transition to ``half_open`` and try once; otherwise we raise
      ``ComposerError(composer, RuntimeError("circuit open"))``
      *without* invoking the child.
    - ``half_open`` → call child.  Success resets to ``closed``;
      failure re-opens with a fresh ``_opened_at``.
    """
    now = time.monotonic()
    if composer._state == "open":
        if (
            composer._opened_at is None
            or (now - composer._opened_at) < composer.reset_timeout
        ):
            raise ComposerError(
                composer,
                RuntimeError(
                    f"circuit_breaker is open "
                    f"(threshold={composer.threshold}, "
                    f"reset_timeout={composer.reset_timeout}s)"
                ),
            )
        composer._state = "half_open"

    ambient_parent = get_parent_episode_id()
    sub_run = _invoke_with_parent(child, input_value, ambient_parent)
    composite._sub_runs.append(sub_run)
    try:
        result: AgentResult[Any] = await sub_run.result()
    except BaseException as exc:
        if composer._state == "half_open":
            composer._state = "open"
            composer._opened_at = time.monotonic()
        else:
            composer._consecutive_failures += 1
            if composer._consecutive_failures >= composer.threshold:
                composer._state = "open"
                composer._opened_at = time.monotonic()
        raise ComposerError(child, exc) from exc

    composer._consecutive_failures = 0
    if composer._state == "half_open":
        composer._state = "closed"
        composer._opened_at = None

    return _build_composite_result(
        value=result.value,
        last_result=result,
        episode_id=result.episode_id,
        inference_ids=list(sub_run.inference_ids),
        turns=result.turns,
    )
