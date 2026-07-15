"""Resolve ``module:attr`` specs to Agent factories.

The CLI treats every subcommand's primary argument as a
``module:attr`` reference to an **async-context-manager factory** for an
:class:`~t0_agents.core.agent.Agent` — idiomatically a zero-argument
callable decorated with :func:`contextlib.asynccontextmanager`.

The factory contract gives the CLI a well-defined lifecycle hook.  The
CLI opens the context around the REPL / the uvicorn lifespan / the
``info`` lookup, and the user's code is responsible for entering the
:class:`tensorzero.AsyncTensorZeroGateway` and the
:class:`fastmcp.Client` that the :class:`Agent` delegates to.  Neither
of those resources is usable without its ``async with`` entered —
:meth:`Agent.invoke` calls through to ``Client.list_tools`` which fails
without an open MCP session.

Example user factory::

    # examples/10_cli/agent.py
    from contextlib import asynccontextmanager
    from typing import Awaitable, cast

    from fastmcp import Client
    from tensorzero import AsyncTensorZeroGateway

    from t0_agents import Agent, AgentConfig
    from .server import mcp

    @asynccontextmanager
    async def assistant():
        gateway = cast(
            Awaitable[AsyncTensorZeroGateway],
            AsyncTensorZeroGateway.build_http(
                gateway_url="http://localhost:3000",
            ),
        )
        async with await gateway as t0, Client(mcp) as client:
            yield Agent(
                AgentConfig(name="assistant", function_name="agent"),
                t0=t0,
                tool_provider=client,
            )
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, TypeAlias, cast

import typer

from t0_agents.core.agent import Agent


def _ensure_cwd_on_path() -> None:
    """Add the current working directory to ``sys.path`` if missing.

    Installed console scripts (``t0-agents``) do not get ``cwd`` on
    ``sys.path`` the way ``python script.py`` does — Python adds the
    script's directory instead.  For the ``module:attr`` convention to
    resolve user-authored factories without forcing ``PYTHONPATH=.``,
    the loader prepends ``cwd`` defensively.  Idempotent: if the cwd
    (``''`` or the absolute path) is already present we don't touch
    ``sys.path``.
    """
    cwd = os.getcwd()
    if "" in sys.path or cwd in sys.path:
        return
    sys.path.insert(0, cwd)

AgentFactory: TypeAlias = Callable[[], AbstractAsyncContextManager[Agent]]
"""Type alias for the object a ``module:attr`` spec must resolve to.

A zero-argument callable that returns an async context manager yielding
an :class:`Agent`.  The canonical form is
``@asynccontextmanager async def my_agent(): ...``, but any object
that satisfies the protocol works (including plain classes implementing
``__call__``).
"""


def load_agent_factory(spec: str, *, agent_name: str | None = None) -> AgentFactory:
    """Resolve ``module:attr`` to an :data:`AgentFactory`.

    Args:
        spec: A ``module.path:attribute`` string.  The attribute must be
            a zero-argument callable whose ``()`` invocation returns an
            async context manager yielding an :class:`Agent`, **or a
            glow** :class:`~t0_agents.glow.app.App` instance — the CLI
            adapts it into a factory that enters ``async with app:``
            and yields the selected agent's underlying ``Agent``.
        agent_name: For glow ``App`` specs with several registered
            agents, the one to run.  Defaults to the sole registered
            agent; passing it with a non-``App`` spec is an error.

    Returns:
        The resolved factory, ready to call as ``async with
        factory() as agent:``.

    Raises:
        typer.BadParameter: If the spec is malformed, the module is not
            importable, the attribute is missing, the attribute is a
            bare :class:`Agent` instance (error message points at the
            factory idiom), the attribute is not callable, or the glow
            agent selection is ambiguous/unknown.
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"Spec must be of the form 'module:attr', got {spec!r}. "
            "Example: 'examples.10_cli.agent:assistant'.",
        )

    module_name, attr_name = spec.rsplit(":", 1)
    if not module_name or not attr_name:
        raise typer.BadParameter(
            f"Spec must have a non-empty module and attribute, got {spec!r}.",
        )

    _ensure_cwd_on_path()

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            f"Could not import module {module_name!r}: {exc}.  "
            "Make sure the module is importable from your current "
            "working directory or on PYTHONPATH.",
        ) from exc
    except ImportError as exc:
        raise typer.BadParameter(
            f"Failed to import module {module_name!r}: {exc}.",
        ) from exc

    try:
        obj = getattr(module, attr_name)
    except AttributeError as exc:
        raise typer.BadParameter(
            f"Module {module_name!r} has no attribute {attr_name!r}.",
        ) from exc

    if isinstance(obj, Agent):
        raise typer.BadParameter(
            f"{spec!r} resolves to a bare Agent instance, but the CLI "
            "expects an async-context-manager factory so it can manage "
            "the gateway and MCP client lifecycle.  Wrap your Agent "
            "construction in an @asynccontextmanager async def, for "
            "example:\n\n"
            "    from contextlib import asynccontextmanager\n\n"
            "    @asynccontextmanager\n"
            f"    async def {attr_name}():\n"
            "        async with await gateway as t0, Client(mcp) as client:\n"
            "            yield Agent(config, t0=t0, tool_provider=client)",
        )

    # Glow App instances are adapted, not rejected — checked before the
    # callable test (App defines no __call__).  Imported lazily: glow
    # rides the [glow] extra (pydantic), the loader must import clean
    # without it until an App spec is actually used.
    from t0_agents.glow.app import App as _GlowApp

    if isinstance(obj, _GlowApp):
        return _factory_from_glow_app(obj, spec, agent_name)

    if agent_name is not None:
        raise typer.BadParameter(
            f"--agent only applies when the spec resolves to a glow App; "
            f"{spec!r} resolves to {type(obj).__name__}."
        )

    if not callable(obj):
        raise typer.BadParameter(
            f"{spec!r} resolves to {type(obj).__name__}, but the CLI "
            "expects a callable that returns an async context manager "
            "yielding an Agent.",
        )

    return cast(AgentFactory, cast(Any, obj))


def _factory_from_glow_app(
    glow_app: Any, spec: str, agent_name: str | None
) -> AgentFactory:
    """Adapt a glow ``App`` into the CLI's Agent-factory contract.

    Agent selection is validated eagerly (clear errors before any
    boot).  The returned factory enters ``async with glow_app:`` —
    gateway resolution, MCP mounts, client connection, ``tools=``
    filters — yields the selected agent's built core :class:`Agent`,
    and tears the App down on exit.
    """
    registered = sorted(glow_app._agent_registry)
    if not registered:
        raise typer.BadParameter(
            f"{spec!r} is a glow App with no registered agents."
        )
    if agent_name is None:
        if len(registered) > 1:
            raise typer.BadParameter(
                f"{spec!r} registers {len(registered)} agents "
                f"({', '.join(registered)}); pick one with --agent."
            )
        agent_name = registered[0]
    elif agent_name not in glow_app._agent_registry:
        raise typer.BadParameter(
            f"--agent {agent_name!r} is not registered on {spec!r}.  "
            f"Available: {', '.join(registered)}."
        )
    selected = agent_name

    @asynccontextmanager
    async def factory() -> AsyncIterator[Agent]:
        async with glow_app:
            yield glow_app._lookup_or_build(glow_app._agent_registry[selected])

    return factory
