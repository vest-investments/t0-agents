"""``@app.agent`` decorator factory and the ``_AgentWrapper`` callable.

Two pieces live here:

* :class:`_AgentWrapper` — generic callable returned by every
  ``@app.agent`` registration.  Stores the owning :class:`App` and
  ``_AgentSpec`` references, exposes ``._agent`` and ``._spec`` for
  escape-hatch use, and on call wires the user's message into the
  :class:`App` build closure to produce a :class:`FluentRun`.
* :func:`_make_decorator` — builds the decorator that the
  ``App.agent(name)`` factory returns.  At decoration time it walks
  ``inspect.signature`` + ``typing.get_type_hints`` to compute the
  :class:`OutputMode`, the system prompt (``inspect.cleandoc(__doc__)``),
  and the optional ``value_transform`` for ``list[BaseModel]`` returns.

The typing contract is the FastAPI pattern: ``App.agent`` returns
``Callable[[Callable[P, Awaitable[R]]], _AgentWrapper[P, R]]``, so a
``mypy``-strict caller sees ``tutor("hi")`` typed as ``FluentRun[R]``
and ``await tutor("hi")`` typed as ``R``.  The runtime body never
calls the decorated function — its signature is the public interface,
its docstring is the system prompt.

M2 scope: ``__call__`` accepts exactly one positional argument (the
user message); kwargs or extra positionals raise ``TypeError``.  M3
lifts this when ``Depends(...)`` introspection lands.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import (
    Annotated,
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from t0_agents.core.agent import Agent
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow._spec import _AgentSpec
from t0_agents.glow._typing import _is_list_of_pydantic
from t0_agents.glow.app import App
from t0_agents.glow.di import _Depends
from t0_agents.glow.handle import FluentRun
from t0_agents.glow.mcp_typing import _extract_mcp_specs, _is_mcp_parameter
from t0_agents.glow.output_inference import output_mode_from_return_type

__all__ = ["_AgentWrapper", "_make_tool_decorator"]

# ``ToolFn`` is the public-facing tool function type — async callable
# returning the tool's result.  FastMCP introspects ``__signature__``
# and ``__annotations__``, so we constrain to ``Callable[..., Any]``.
ToolFn = Callable[..., Awaitable[Any]] | Callable[..., Any]

P = ParamSpec("P")
R = TypeVar("R")


class _AgentWrapper(Generic[P, R]):
    """Callable wrapper returned by ``@app.agent``.

    Preserves the original function's parameters via :class:`ParamSpec`
    so ``mypy``-strict callers see the wrapped signature unchanged.
    On invocation, hands off to ``App._invoke`` which returns a
    :class:`FluentRun` typed at the original return annotation.

    The ``._agent`` and ``._spec`` accessors are the escape hatches
    documented in the plan ("``tutor._agent`` is exposed for users
    who want to drop down to the base API").  ``._agent`` is ``None``
    until the first invocation (or ``await app.build()``).
    """

    def __init__(self, app: App, spec: _AgentSpec) -> None:
        self._app = app
        self._spec_ref = spec

    @property
    def _agent(self) -> Agent | None:
        """The underlying :class:`Agent`, or ``None`` if not yet built."""
        return self._spec_ref.agent

    @property
    def _spec(self) -> _AgentSpec:
        """The registration-time :class:`_AgentSpec`."""
        return self._spec_ref

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> FluentRun[R]:
        """Invoke the agent with a user message.

        M3 scope: exactly one positional argument (the message);
        kwargs and additional positionals raise ``TypeError``.
        Sub-agent ``Depends(agent_wrapper)`` parameters declared on
        the decorated function are consumed at decoration time and
        compiled into ``AgentConfig.sub_agents``; the runtime call
        signature has no slots for them, so the user never passes
        them through ``__call__``.
        """
        if kwargs:
            raise TypeError(
                f"@app.agent({self._spec_ref.name!r}) wrapper accepts only "
                f"a single positional argument (the user message).  "
                f"`Depends(agent)` parameters declared on the decorated "
                f"function are compiled into the agent's sub_agents at "
                f"decoration time and must not be passed at call time.  "
                f"Got kwargs: {sorted(kwargs)!r}"
            )
        if len(args) != 1:
            raise TypeError(
                f"@app.agent({self._spec_ref.name!r}) wrapper expects exactly "
                f"one positional argument (the user message); got {len(args)}."
            )
        message = args[0]
        if not isinstance(message, str):
            raise TypeError(
                f"@app.agent({self._spec_ref.name!r}) wrapper expects a str "
                f"message; got {type(message).__name__}."
            )
        return cast(FluentRun[R], self._app._invoke(self._spec_ref, message))


def _make_decorator(
    app: App,
    name: str,
    *,
    function_name: str,
) -> Callable[[Callable[P, Awaitable[R]]], _AgentWrapper[P, R]]:
    """Build the decorator returned by ``App.agent(name)``.

    The decorator captures registration-time information (``OutputMode``,
    system prompt, return type, ``value_transform``) so it never has to
    walk the signature on invocation.  Forward-ref failures in the
    return annotation surface here as :class:`ConfigurationError`
    (raised by :func:`output_mode_from_return_type`) before the wrapper
    can be called.
    """
    if name in app._agent_registry:
        raise ConfigurationError(
            f"Duplicate agent name {name!r} — the App registry already "
            f"contains an agent registered under this name.  Pick a unique "
            f"name or remove the prior registration."
        )

    def decorator(fn: Callable[P, Awaitable[R]]) -> _AgentWrapper[P, R]:
        # Decoration-time wiring.  Any failure raises eagerly so the
        # error surfaces at import / module-load time rather than on
        # first invocation (FastAPI's "fail fast on registration"
        # ergonomic).
        output_mode = output_mode_from_return_type(fn)
        system_prompt: str | None = (
            inspect.cleandoc(fn.__doc__) if fn.__doc__ else None
        )
        try:
            hints = get_type_hints(fn, include_extras=True)
        except (NameError, TypeError) as exc:
            # output_mode_from_return_type would already have raised on
            # return-annotation forward-ref failure.  We only end up
            # here when parameter annotations carry an unresolvable
            # forward ref — surface the same ConfigurationError shape.
            raise ConfigurationError(
                f"@app.agent({name!r}): could not resolve type hints on "
                f"{fn.__module__}.{fn.__qualname__}: {exc}"
            ) from exc
        return_type: Any = hints.get("return", str)

        sub_agent_specs = _collect_sub_agent_specs(fn, app, name, hints)

        # M4: harvest any MCPServer specs reachable through the
        # decorated function's parameter annotations and append them
        # to the App's spec list.  The actual ``FastMCP.mount`` happens
        # in ``app.build()`` BEFORE the owned client is entered; the
        # decorator only records the spec.  Forward-ref failures inside
        # ``_extract_mcp_specs`` are silently no-ops there — but
        # ``get_type_hints`` above already raised for the same case if
        # the failure is fatal, so we never reach here with an
        # unresolved annotation that ``_extract_mcp_specs`` cares about.
        for mcp_spec in _extract_mcp_specs(fn, hints):
            app._mcp_specs.append(mcp_spec)

        # ``value_transform`` is the M1 carry-forward §4(d) hook: when
        # the return annotation is ``list[BaseModel]``,
        # ``output_mode_from_return_type`` chose a ``StructuredOutput``
        # over the synthetic single-field wrapper model (``items:
        # list[T]``).  The wrapper layer must hand the user the inner
        # list, not the wrapper, so the user-visible value matches
        # ``list[T]`` exactly.
        value_transform: Callable[[Any], Any] | None = None
        if _is_list_of_pydantic(return_type):
            value_transform = _unwrap_wrapper_items

        spec = _AgentSpec(
            name=name,
            function_name=function_name,
            fn=cast(Callable[..., Awaitable[Any]], fn),
            system_prompt=system_prompt,
            output_mode=output_mode,
            return_type=return_type,
            value_transform=value_transform,
            sub_agent_specs=sub_agent_specs,
        )
        app._agent_registry[name] = spec
        return _AgentWrapper[P, R](app, spec)

    return decorator


def _collect_sub_agent_specs(
    fn: Callable[..., Any],
    app: App,
    agent_name: str,
    hints: dict[str, Any],
) -> list[_AgentSpec]:
    """Walk ``fn``'s signature and return the sub-agent specs declared via ``Depends``.

    The decoration-time contract for ``@app.agent`` parameters is
    deliberately narrow at M3:

    * The first positional parameter is the user message; type
      annotation must be ``str`` (or ``Annotated[str, ...]`` — the
      Annotated wrapper is peeled before comparison so M4's
      guardrail-on-message pattern lands cleanly without rewriting
      this check).
    * Every subsequent parameter MUST have a :func:`Depends` default
      whose ``provider`` is an :class:`_AgentWrapper` *registered on
      the same App* (the sub-agent special case).  The parameter is
      then "compiled out" of the runtime call signature; its provider's
      ``_spec_ref`` is appended to the returned list.

    Any other shape raises :class:`ConfigurationError`:

    * Value-style ``Depends(get_db)`` on an ``@app.agent`` param:
      point the user at ``@app.tool`` (the correct surface for tool
      dependencies at M3).
    * ``Depends(other_wrapper)`` where ``other_wrapper`` is not in
      ``app._agent_registry``: order-of-registration mistake; ask
      the user to register the sub-agent BEFORE the orchestrator.
    * Missing default, missing annotation, non-``Depends`` default,
      non-positional kind: structural mismatch with the spec.

    ``hints`` is the already-resolved
    :func:`typing.get_type_hints(fn, include_extras=True)` map from
    ``_make_decorator``; passing it down avoids a second resolve and
    means ``Annotated`` wrappers + string forward-refs are already
    materialised when we inspect the message parameter's type.

    Returns the sub-agent specs in the order their parameters appear
    on the user function.  Empty list when the function has only a
    message parameter (the M2 case).
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        raise ConfigurationError(
            f"@app.agent({agent_name!r}): {fn.__module__}.{fn.__qualname__} "
            f"must declare at least one positional parameter (the user "
            f"message)."
        )

    message_param = params[0]
    if message_param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ConfigurationError(
            f"@app.agent({agent_name!r}): the first parameter "
            f"({message_param.name!r}) must be positional (the user "
            f"message)."
        )
    if message_param.name in hints:
        # ``hints`` was resolved with ``include_extras=True``, so an
        # ``Annotated[str, ...]`` parameter survives as a real
        # ``_AnnotatedAlias`` here.  Peel one layer of ``Annotated`` —
        # the underlying type must be ``str``.  This makes M4's
        # ``Annotated[T, Guardrail(...)]`` pattern compatible at the
        # message position without revisiting this check.
        message_hint: Any = hints[message_param.name]
        if get_origin(message_hint) is Annotated:
            message_hint = get_args(message_hint)[0]
        if message_hint is not str:
            raise ConfigurationError(
                f"@app.agent({agent_name!r}): the first parameter "
                f"({message_param.name!r}) must be annotated as ``str`` "
                f"(the user message); got {message_hint!r}."
            )

    sub_agent_specs: list[_AgentSpec] = []
    for param in params[1:]:
        # M4: MCP-typed parameters are pure decoration metadata (the
        # spec is harvested separately by ``_extract_mcp_specs``); they
        # have no runtime binding and no default, so we skip them here
        # rather than demanding a ``Depends`` default.  The wrapper's
        # ``__call__`` continues to accept only the user message — the
        # MCP parameter slot is never populated at invocation.
        if _is_mcp_parameter(hints.get(param.name)):
            continue
        if param.default is inspect.Parameter.empty:
            raise ConfigurationError(
                f"@app.agent({agent_name!r}): parameter {param.name!r} has "
                f"no default — every parameter beyond the user message must "
                f"declare ``= Depends(<glow agent wrapper>)`` (the sub-agent "
                f"special case).  Value-style dependencies belong on "
                f"``@app.tool`` at M3."
            )
        default = param.default
        if not isinstance(default, _Depends):
            raise ConfigurationError(
                f"@app.agent({agent_name!r}): parameter {param.name!r} has a "
                f"non-Depends default {default!r}.  Every parameter beyond "
                f"the user message must declare ``= Depends(<glow agent "
                f"wrapper>)``."
            )

        provider = default.provider
        if not isinstance(provider, _AgentWrapper):
            raise ConfigurationError(
                f"@app.agent({agent_name!r}): parameter {param.name!r} uses "
                f"Depends({provider!r}), a value-style provider.  "
                f"@app.agent only accepts Depends(<glow agent wrapper>) for "
                f"non-message parameters at M3 (the sub-agent special case); "
                f"use @app.tool for value-style dependencies."
            )

        provider_spec = provider._spec_ref
        if provider_spec not in app._agent_registry.values():
            raise ConfigurationError(
                f"@app.agent({agent_name!r}): parameter {param.name!r} uses "
                f"Depends(<wrapper for {provider_spec.name!r}>), but that "
                f"wrapper is not registered on this App.  Register the "
                f"sub-agent with ``@app.agent({provider_spec.name!r})`` "
                f"BEFORE the orchestrator that depends on it (glow does not "
                f"support forward references between agents)."
            )
        sub_agent_specs.append(provider_spec)

    return sub_agent_specs


def _unwrap_wrapper_items(wrapper_value: Any) -> Any:
    """Extract ``items`` from a synthetic-list wrapper :class:`BaseModel`.

    Lives at module scope (rather than as a lambda) so the value
    transform survives serialisation if any future tooling needs to
    introspect the spec.
    """
    return wrapper_value.items


# ---------------------------------------------------------------------------
# ``@app.tool`` decorator
# ---------------------------------------------------------------------------


def _make_tool_decorator(app: App) -> Callable[[ToolFn], ToolFn]:
    """Return the decorator returned by :meth:`App.tool`.

    The decorator registers a FastMCP tool on ``app._mcp_server``.
    User functions may declare ``Depends(provider)`` defaults on
    parameters that should be injected by glow at call time; those
    parameters are **stripped** from the signature FastMCP introspects
    (so the tool's JSON Schema does not advertise them to the LLM),
    and a per-tool async wrapper resolves them via
    :class:`~t0_agents.glow.di.DependencyResolver` before invoking the
    user function.

    ``Depends(agent_wrapper)`` is rejected at decoration time — that
    sentinel is reserved for ``@app.agent``'s sub-agent special case
    and has no defined runtime behaviour inside a tool (a tool running
    inside the MCP server task that re-entered the agent loop would
    surface complex re-entrance semantics with marginal benefit).
    """

    def decorator(fn: ToolFn) -> ToolFn:
        sig = inspect.signature(fn)
        depends_params: dict[str, _Depends] = {}
        public_params: list[inspect.Parameter] = []
        for name, param in sig.parameters.items():
            default = param.default
            if isinstance(default, _Depends):
                if isinstance(default.provider, _AgentWrapper):
                    raise ConfigurationError(
                        f"@app.tool({fn.__name__!r}): parameter {name!r} uses "
                        f"Depends(<glow agent wrapper>), but Depends(agent) is "
                        f"reserved for @app.agent's sub-agent special case.  "
                        f"Use @app.agent for orchestrator-with-sub-agents "
                        f"patterns, or pick a value-style provider for the "
                        f"tool dependency."
                    )
                depends_params[name] = default
            else:
                public_params.append(param)

        stripped_sig = sig.replace(parameters=public_params)
        # FastMCP's ``Tool.from_function`` reads ``__annotations__`` via
        # ``get_type_hints`` (`fastmcp/tools/function_parsing.py:198`).
        # We resolve the user function's hints HERE (using ``fn.__globals__``)
        # so the wrapper carries actual type objects, not strings.  Without
        # this step, ``from __future__ import annotations`` in user code
        # would leave ``fn.__annotations__`` as strings, and FastMCP's
        # ``get_type_hints(wrapper)`` would try to resolve them against
        # ``wrapper.__globals__`` (this module's globals) where user-
        # defined types are NOT in scope.  Resolution is best-effort:
        # if it fails (rare — usually a deeply-forward-ref'd type), we
        # fall back to the raw ``__annotations__`` so registration
        # still proceeds.
        try:
            resolved_hints = get_type_hints(fn, include_extras=True)
        except (NameError, TypeError):
            resolved_hints = getattr(fn, "__annotations__", {})
        stripped_annotations: dict[str, Any] = {
            ann_name: ann_type
            for ann_name, ann_type in resolved_hints.items()
            if ann_name == "return" or ann_name not in depends_params
        }

        async def wrapper(**call_kwargs: Any) -> Any:
            resolved = await app._resolver.resolve(fn, app)
            merged = {**call_kwargs, **resolved}
            result = fn(**merged)
            if inspect.isawaitable(result):
                return await result
            return result

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__doc__ = fn.__doc__
        wrapper.__module__ = fn.__module__
        wrapper.__signature__ = stripped_sig  # type: ignore[attr-defined]
        wrapper.__annotations__ = stripped_annotations

        # ``FastMCP.add_tool`` accepts ``Tool | Callable[..., Any]``; the
        # callable branch goes through ``Tool.from_function``, which
        # introspects ``wrapper.__signature__`` to build the JSON schema.
        # Strict mypy can't see the Tool branch (no FastMCP stubs +
        # no @overload), so we pick the callable side explicitly.
        app._mcp_server.add_tool(cast(Callable[..., Any], wrapper))
        return fn

    return decorator
