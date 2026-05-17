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
from typing import Any, Generic, ParamSpec, TypeVar, cast, get_type_hints

from t0_agents.core.agent import Agent
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow._spec import _AgentSpec
from t0_agents.glow._typing import _is_list_of_pydantic
from t0_agents.glow.app import App
from t0_agents.glow.handle import FluentRun
from t0_agents.glow.output_inference import output_mode_from_return_type

__all__ = ["_AgentWrapper"]

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

        M2 scope: exactly one positional argument (the message);
        kwargs and additional positionals raise ``TypeError``.  M3
        replaces this with full signature introspection so
        ``Depends(...)`` defaults and typed sub-agent injection work.
        """
        if kwargs:
            raise TypeError(
                f"@app.agent({self._spec_ref.name!r}) wrapper accepts only a "
                f"single positional argument (the user message) in M2; "
                f"keyword arguments will be supported in M3 when Depends() "
                f"injection lands.  Got: {sorted(kwargs)!r}"
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
        )
        app._agent_registry[name] = spec
        return _AgentWrapper[P, R](app, spec)

    return decorator


def _unwrap_wrapper_items(wrapper_value: Any) -> Any:
    """Extract ``items`` from a synthetic-list wrapper :class:`BaseModel`.

    Lives at module scope (rather than as a lambda) so the value
    transform survives serialisation if any future tooling needs to
    introspect the spec.
    """
    return wrapper_value.items
