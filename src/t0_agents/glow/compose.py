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

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastmcp import Client, FastMCP

from t0_agents.compose.builtins import parallel, sequential
from t0_agents.compose.topology import (
    DEFAULT_CANCEL_TIMEOUT,
    Composable,
    Composer,
    ComposerError,
    CompositeRun,
)
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs.tool_call import ToolCallOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult, ToolProvider

if TYPE_CHECKING:
    from collections.abc import Mapping

    from t0_agents.glow.decorators import _AgentWrapper

__all__ = ["compose"]

_ROUTER_SYSTEM_PROMPT = (
    "You are a router.  Given the user query, pick exactly one tool whose "
    "name matches the intended branch.  Do not produce additional text."
)


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
        return _llm_routed(spec)
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


# ---------------------------------------------------------------------------
# LLM-routed dict composer
# ---------------------------------------------------------------------------


@dataclass
class _LLMRoutedComposer(Composer):
    """Custom :class:`Composer` for ``compose(dict[str, Composable])``.

    The v1.0 composer factories (``sequential`` + ``conditional``) can't
    be wired to this case directly: ``_drive_sequential``'s default chain
    JSON-stringifies the classifier's ``AgentResult`` before
    ``conditional`` sees it, AND ``conditional`` passes its chained value
    to both the router *and* the picked route — but the picked route
    needs the ORIGINAL user query, not the route key.  This custom
    Composer runs the synthetic classifier internally and dispatches the
    original input to ``routes[picked_key]`` with no chain coupling.

    The classifier is a vanilla v1.0 :class:`Agent` configured with
    :class:`ToolCallOutput(tool_choice="required")` and a glow-private
    :class:`FastMCP` server holding one no-op tool per route key.  The
    LLM picks one tool; the loop terminates after the first inference
    (``ToolCallOutput.expects_tool_loop=False``); we read the picked
    name and dispatch.  The tool bodies never execute.

    The classifier's :class:`Client` is entered per-invocation through an
    ``async with`` block inside the driver — entries are cheap (the
    FastMCP server is in-memory) and this keeps the Composer stateless
    across invocations (concurrency-safe; no shared transport state).
    """

    routes: dict[str, Composable[Any, Any]]
    classifier: Agent
    classifier_client: Client[Any]
    name: str = "llm_routed"
    cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT

    @property
    def children(self) -> tuple[Composable[Any, Any], ...]:
        # Introspection helper — used by tests and by any caller that
        # wants to walk the graph (``_find_leaf_agent`` etc.).  Order is
        # classifier first, then routes in dict-insertion order.
        return (
            cast(Composable[Any, Any], self.classifier),
            *self.routes.values(),
        )

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
            # Enter the classifier's Client per-invocation — fastmcp's
            # in-memory transport is cheap and a fresh entry keeps the
            # composer stateless.  The Agent stored at construction
            # already references this Client as its ``tool_provider``,
            # so entering the context is sufficient — no rewiring needed.
            async with self.classifier_client:
                cls_run = self.classifier.invoke(input, system=_ROUTER_SYSTEM_PROMPT)
                composite._sub_runs.append(cls_run)
                cls_result = await cls_run.result()

            tool_calls = cls_result.value
            if not tool_calls:
                raise ComposerError(
                    self,
                    ValueError("LLM router classifier returned no tool calls"),
                )
            route_key = tool_calls[0].name
            if route_key not in self.routes:
                raise ComposerError(
                    self,
                    ValueError(
                        f"LLM router picked unknown route {route_key!r}; "
                        f"expected one of {list(self.routes)}"
                    ),
                )
            target = self.routes[route_key]
            target_run = target.invoke(input)
            composite._sub_runs.append(target_run)
            return await target_run.result()

        return CompositeRun(
            self.name,
            input,
            _driver,
            cancel_timeout=timeout,
            eager=eager,
            context=context,
        )


def _llm_routed(routes: dict[str, Any]) -> _LLMRoutedComposer:
    """Build a :class:`_LLMRoutedComposer` from a route dict.

    Each value is unwrapped via :func:`_unwrap_node` (glow wrappers
    triggered to build their underlying Agent; raw Agent/Composer pass
    through).  The classifier Agent inherits ``t0`` and ``function_name``
    from the first leaf :class:`Agent` reachable through the routes —
    the user's ``@app.agent`` graph defines those, so the classifier
    automatically uses the same gateway and chat function as the rest of
    the App's agents.
    """
    if not routes:
        raise ConfigurationError(
            "compose({}): routes cannot be empty — provide at least one "
            "key→Composable mapping."
        )
    unwrapped: dict[str, Composable[Any, Any]] = {
        key: _unwrap_node(value) for key, value in routes.items()
    }
    classifier_client, classifier = _make_router_agent(
        list(unwrapped), unwrapped,
    )
    return _LLMRoutedComposer(
        routes=unwrapped,
        classifier=classifier,
        classifier_client=classifier_client,
    )


def _make_router_agent(
    keys: list[str],
    unwrapped_routes: dict[str, Composable[Any, Any]],
) -> tuple[Client[Any], Agent]:
    """Construct the synthetic classifier ``Agent`` for the LLM router.

    Builds a glow-private :class:`FastMCP` server holding one no-op tool
    per route key (the tool body never executes; we only need the LLM
    to *pick* one).  Wraps the server in a :class:`Client`, configures
    an :class:`Agent` with ``ToolCallOutput(tool_choice="required")`` so
    the inference is forced to emit a tool call, and inherits
    ``function_name`` from the first leaf Agent reachable through the
    routes.  Returns the ``(client, agent)`` pair; the client is entered
    per-invocation by :class:`_LLMRoutedComposer`.
    """
    leaf = _find_leaf_agent(unwrapped_routes.values())
    if leaf is None:
        raise ConfigurationError(
            "compose(dict[str, Composable]): no Agent is reachable through "
            "the route values — at least one route must contain a glow "
            "@app.agent wrapper or a v1.0 Agent (the classifier inherits "
            "the gateway and function_name from a leaf Agent)."
        )

    server: FastMCP[Any] = FastMCP(f"glow-router-{secrets.token_hex(4)}")
    for key in keys:
        server.add_tool(_make_route_tool(key))

    client: Client[Any] = Client(server)

    classifier = Agent(
        AgentConfig(
            name=f"_glow_router_{secrets.token_hex(4)}",
            function_name=leaf.config.function_name,
            output=ToolCallOutput(tool_choice="required"),
        ),
        t0=leaf._t0,
        tool_provider=cast(ToolProvider, client),
    )
    return client, classifier


def _make_route_tool(key: str) -> Callable[..., Any]:
    """Build a no-op coroutine usable as a FastMCP tool.

    The tool's ``__name__`` becomes the tool name (and therefore the
    route key the LLM will surface in ``AgentResult.value[0].name``);
    its docstring becomes the MCP tool ``description`` field, which the
    LLM uses to disambiguate routes.  The body never executes —
    ``ToolCallOutput`` terminates the agent loop after the first
    inference returns a tool call.
    """

    async def _route_noop() -> None:
        return None

    _route_noop.__name__ = key
    _route_noop.__doc__ = f"Route the user query to the {key!r} branch."
    return _route_noop


def _find_leaf_agent(
    composables: Any,
) -> Agent | None:
    """Depth-first walk through ``composables`` for the first leaf Agent.

    Used by :func:`_make_router_agent` to inherit ``t0`` /
    ``function_name`` from somewhere in the route graph.  ``Composer``
    children are inspected via the ``children`` attribute every
    v1.0 topology exposes (either as a field or a property).
    """
    for c in composables:
        if isinstance(c, Agent):
            return c
        if isinstance(c, Composer):
            sub = _find_leaf_agent(getattr(c, "children", ()))
            if sub is not None:
                return sub
    return None
