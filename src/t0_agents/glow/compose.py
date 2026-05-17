"""``compose(spec, *, merge=None)`` — native-collection composition.

Maps Python's built-in collection literals to v1.0 :mod:`compose.builtins`
topologies so users can express agent graphs without naming a composer
factory:

- ``list`` / ``tuple`` →
  :func:`~t0_agents.compose.builtins.sequential` (pipeline; each node's
  result feeds the next).
- ``set`` / ``frozenset`` →
  :func:`~t0_agents.compose.builtins.parallel` with
  ``error_policy="continue"`` (every node runs; ``merge``-reduces).
  Anchored on the M0 dev-log §4 carry-forward — the `race()` deprecation
  warning literally points at this exact factory call.
- ``dict[str, Composable]`` → an LLM-routed dispatch composer
  (``_LLMRoutedComposer``, landed in commit 3).

The dispatcher returns a :class:`~t0_agents.compose.topology.Composer`,
which by the v1.0 :class:`~t0_agents.compose.topology.Composable`
protocol can be nested in another ``compose()`` call freely
(``compose([compose({a, b}, merge=concat), summarizer])`` is just
nesting; the type system already type-checks this without ceremony).

Glow's ``@app.agent`` decorator returns ``_AgentWrapper`` callables, not
raw :class:`~t0_agents.core.agent.Agent` instances.  :func:`compose`
accepts wrappers and unwraps them through ``app._lookup_or_build`` (a
lazy build, identical to the M3 path) before handing the underlying
:class:`Agent` to the v1.0 composer factory.  Pre-built Agents and
nested Composers pass through unchanged — that's the
``Composable[Any, Any]`` protocol doing its job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from t0_agents.compose.builtins import parallel, sequential
from t0_agents.compose.topology import Composable, Composer, ComposerError
from t0_agents.core.agent import Agent
from t0_agents.core.types import AgentResult

if TYPE_CHECKING:
    from t0_agents.glow.decorators import _AgentWrapper

__all__ = ["compose"]


def compose(
    spec: Any,
    *,
    merge: Callable[[list[Any]], Any] | None = None,
) -> Composer:
    """Build a Composer from a native Python collection literal.

    Dispatch table:

    +----------------------+-------------------------------------------+
    | ``spec`` type        | Returned composer                         |
    +======================+===========================================+
    | ``list`` / ``tuple`` | ``sequential(*unwrapped)``                |
    +----------------------+-------------------------------------------+
    | ``set``/``frozenset``| ``parallel(*unwrapped, reducer=...,``     |
    |                      | ``error_policy="continue")``              |
    +----------------------+-------------------------------------------+
    | ``dict``             | LLM-routed dispatch (commit 3)            |
    +----------------------+-------------------------------------------+

    ``merge`` is only consulted for the ``set``/``frozenset`` case; when
    omitted, :func:`_default_merge` concatenates string results and
    raises ``TypeError`` for mixed-type ensembles (forcing the caller to
    pass ``merge=`` explicitly).  Single-string results survive
    unchanged.

    Raises ``TypeError`` for any spec type outside the dispatch table or
    when an element is not :class:`Composable`.  ``compose`` never
    silently coerces — every glow wrapper or raw Composable must be
    explicit at the call site.
    """
    if isinstance(spec, list | tuple):
        unwrapped = tuple(_unwrap_node(node) for node in spec)
        return sequential(*unwrapped)
    if isinstance(spec, set | frozenset):
        unwrapped = tuple(_unwrap_node(node) for node in spec)
        reducer: Callable[[list[Any]], Any] = (
            merge if merge is not None else _default_merge
        )
        return parallel(*unwrapped, reducer=reducer, error_policy="continue")
    if isinstance(spec, dict):
        # Commit 3 lands ``_llm_routed``; until then the dict branch
        # raises a clear NotImplementedError so users know it's the
        # right surface but not yet built.
        raise NotImplementedError(
            "compose(dict[str, Composable]) is landed in the next commit "
            "(LLM-routed dispatch composer)."
        )
    raise TypeError(
        f"compose() got unsupported spec type {type(spec).__name__!r}; "
        f"expected list/tuple (sequential), set/frozenset (fan-out + merge), "
        f"or dict[str, Composable] (LLM router)."
    )


def _default_merge(results: list[Any]) -> Any:
    """Default reducer for ``compose(set/frozenset)`` ensembles.

    ``parallel(error_policy="continue")`` produces a list whose elements
    are either ``AgentResult[Any]`` (successes) or ``BaseException``
    (failures kept inline so the reducer can decide what to do).  The
    default policy:

    - If every successful element's ``value`` is a ``str``, return them
      joined by ``\\n---\\n``.  Plays nicely with the most common case
      (LLM-text ensembles) without bespoke configuration.
    - If at least one successful element's ``value`` is non-``str``, raise
      ``TypeError`` and demand an explicit ``merge=``.  Mixing strings
      and dicts (or Pydantic models, etc.) in one ensemble is the kind of
      silent bug the explicit-is-better-than-implicit policy is meant to
      catch.
    - If every branch failed, raise ``ComposerError`` with the parent
      composer placeholder — the caller wired a fan-out that never
      produced a value.
    """
    successes: list[AgentResult[Any]] = [
        r for r in results if not isinstance(r, BaseException)
    ]
    if not successes:
        # ``ComposerError`` expects a ``Composable`` ``node`` argument
        # for surfacing the failing topology.  The marker is a stable
        # sentinel for "default reducer ran out of successes" — cast
        # so mypy strict accepts the placeholder.
        raise ComposerError(
            cast(Composable[Any, Any], _DefaultMergeMarker()),
            RuntimeError("compose(set/frozenset): every branch failed"),
        )
    if all(isinstance(r.value, str) for r in successes):
        return "\n---\n".join(r.value for r in successes)
    raise TypeError(
        "compose(set/frozenset) with mixed or non-string results needs an "
        "explicit `merge=` reducer.  The default reducer only handles "
        "all-string ensembles; pass a callable to fuse heterogeneous "
        "outputs."
    )


class _DefaultMergeMarker:
    """Sentinel passed to :class:`ComposerError` when the default
    reducer fails because every branch errored.  Carries a stable
    ``repr`` so logs are readable; not part of the public surface."""

    def __repr__(self) -> str:
        return "<compose-default-merge>"


def _unwrap_node(node: Any) -> Composable[Any, Any]:
    """Resolve a node to its underlying :class:`Composable`.

    Glow ``@app.agent`` returns an ``_AgentWrapper`` (the typed callable
    layer), not the underlying :class:`Agent`.  ``compose()`` accepts
    wrappers and unwraps them via ``wrapper._app._lookup_or_build(spec)``
    so the v1.0 composer factories see the real :class:`Agent` they
    expect.  Pre-built ``Agent``/``Composer`` instances pass through.

    Anything else — a string, an integer, a function — raises
    ``TypeError``.  The error message names the offending node so the
    user can find the bad literal quickly.
    """
    # Late import to break the cycle: ``decorators`` imports ``app``
    # imports nothing from ``compose``, but ``compose`` needs to
    # ``isinstance``-check the wrapper class.
    from t0_agents.glow.decorators import _AgentWrapper

    if isinstance(node, _AgentWrapper):
        return cast(Composable[Any, Any], _build_wrapped_agent(node))
    if isinstance(node, Agent | Composer):
        # ``Agent.invoke`` is typed with overloads (str user_message); the
        # ``Composable`` Protocol's ``invoke(input, /, **kwargs) -> Run[T]``
        # matches structurally but mypy strict will not widen the
        # overload-typed method into the Protocol signature.  Both
        # ``Agent`` and ``Composer`` satisfy ``Composable`` at runtime
        # (the protocol is ``@runtime_checkable``), so the cast is
        # widening, not lying.
        return cast(Composable[Any, Any], node)
    raise TypeError(
        f"compose() node {node!r} is not a Composable (expected an Agent, "
        f"a Composer, or a glow @app.agent wrapper)."
    )


def _build_wrapped_agent(wrapper: _AgentWrapper[Any, Any]) -> Agent:
    """Trigger a lazy build of the wrapper's underlying ``Agent``.

    Lifted into its own helper so ``_unwrap_node`` reads top-to-bottom.
    Mirrors what ``app._invoke`` does on the regular ``await wrapper(msg)``
    path — both end up calling ``app._lookup_or_build(spec)``.
    """
    return wrapper._app._lookup_or_build(wrapper._spec_ref)
