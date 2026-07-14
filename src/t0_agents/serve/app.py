"""Top-level FastAPI factory — one-line wrap of an ``Agent`` in HTTP.

The public surface is :func:`create_app` and :func:`mount_agent`.  Both
take an existing :class:`Agent` plus *exactly one* of three modes:

* ``session_provider`` — caller-supplied :class:`SessionProvider`
  (advanced path: auth, multi-tenant routing, telemetry, custom WS
  elicitation wiring).  See ``docs/streaming.md``.
* ``client_factory`` — zero-arg callable that builds a fresh
  ``fastmcp.Client`` for every session.  The default path for
  non-streaming agents.  Optionally paired with
  ``elicitation_client_factory`` for streaming WebSocket elicitation.
* ``share_client=True`` — reuse the agent's tool provider across every
  session.  Safe only when no tool uses elicitation; the caller owns the
  ``Client``'s open/close lifecycle.

Supplying the wrong combination raises :class:`ConfigurationError` at
startup — no silent ambiguity.

``deployment_label``, when set, is merged into ``AgentConfig.tags`` as
``t0_agents.deployment=<label>`` before any session is built, so every
inference and every forwarded feedback record carries the same
"which surface?" observability tag.  ``namespace`` on ``AgentConfig``
and ``deployment_label`` are orthogonal: the former routes variant
sampling, the latter is pure metadata.

``streaming=True`` additionally mounts the SSE (``GET /chat/stream``) and
WebSocket (``WS /chat/ws``) routes from
:mod:`t0_agents.serve.streaming_routes`.  Default is ``False``;
batch-only deployments pay nothing for the streaming imports when they
don't need them.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fastapi import FastAPI

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.serve.lifespan import build_lifespan
from t0_agents.serve.routes import build_router
from t0_agents.serve.session import (
    ClientFactory,
    ElicitationClientFactory,
    SessionFactory,
    SessionProvider,
)

if TYPE_CHECKING:
    from t0_agents.glow.app import App as GlowApp


def create_app(
    agent: Agent,
    *,
    session_provider: SessionProvider | None = None,
    client_factory: ClientFactory | None = None,
    elicitation_client_factory: ElicitationClientFactory | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
    streaming: bool = False,
) -> FastAPI:
    """Build a ``FastAPI`` application that serves ``agent`` over HTTP.

    Raises :class:`ConfigurationError` when the mode is ambiguous (zero or
    more than one of ``session_provider`` / ``client_factory`` /
    ``share_client``) or when ``elicitation_client_factory`` is passed
    without ``client_factory``.  Does not start the server — the caller
    runs ``uvicorn`` (or an equivalent ASGI runner) against the returned
    app.
    """
    session_factory = _build_session_factory(
        agent=agent,
        session_provider=session_provider,
        client_factory=client_factory,
        elicitation_client_factory=elicitation_client_factory,
        share_client=share_client,
        deployment_label=deployment_label,
    )

    app = FastAPI(
        lifespan=build_lifespan(session_factory=session_factory, agent=agent),
        title=f"t0-agents · {session_factory.base_config.name}",
    )
    app.state.t0 = agent._t0
    app.state.session_factory = session_factory
    app.include_router(build_router(session_factory=session_factory, t0=agent._t0))
    if streaming:
        # Local import so create_app() stays cheap when streaming is off.
        from t0_agents.serve.streaming_routes import build_streaming_router

        app.include_router(build_streaming_router(session_factory=session_factory))
    return app


async def create_app_from_glow(
    glow_app: GlowApp,
    *,
    agent: str | None = None,
    share_client: bool = True,
    deployment_label: str | None = None,
    streaming: bool = False,
) -> FastAPI:
    """Bridge a glow :class:`~t0_agents.glow.app.App` into :func:`create_app`.

    Boots the App (``await glow_app.build()`` — gateway resolution, MCP
    mounts, client connection, ``tools=`` filters; idempotent), selects
    an agent (the sole registered one, or ``agent=`` by name — anything
    else raises :class:`ConfigurationError` listing the options), and
    delegates to :func:`create_app` with the built core
    :class:`Agent`.

    The **caller owns the glow lifecycle**: keep the App inside
    ``async with glow_app:`` (or call ``glow_app.shutdown()``) for the
    server's lifetime — mirroring ``share_client``'s "caller owns the
    Client" contract.  ``share_client=True`` (the default) serves every
    session over the App's connected FastMCP client; a toolless glow
    App has no client, so :func:`create_app` raises its standard
    ConfigurationError — compose :func:`create_app` directly with a
    ``client_factory`` for that shape.
    """
    await glow_app.build()
    registered = sorted(glow_app._agent_registry)
    if not registered:
        raise ConfigurationError(
            "create_app_from_glow: the App has no registered agents."
        )
    if agent is None:
        if len(registered) > 1:
            raise ConfigurationError(
                f"create_app_from_glow: the App registers {len(registered)} "
                f"agents ({', '.join(registered)}); pick one with agent=."
            )
        agent = registered[0]
    elif agent not in glow_app._agent_registry:
        raise ConfigurationError(
            f"create_app_from_glow: agent {agent!r} is not registered.  "
            f"Available: {', '.join(registered)}."
        )
    core_agent = glow_app._lookup_or_build(glow_app._agent_registry[agent])
    return create_app(
        core_agent,
        share_client=share_client,
        deployment_label=deployment_label,
        streaming=streaming,
    )


def mount_agent(
    parent: FastAPI,
    agent: Agent,
    prefix: str,
    *,
    session_provider: SessionProvider | None = None,
    client_factory: ClientFactory | None = None,
    elicitation_client_factory: ElicitationClientFactory | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
    streaming: bool = False,
) -> None:
    """Mount a per-agent sub-app under ``prefix`` on an existing FastAPI app.

    Each mounted sub-app runs its own lifespan and its own
    :class:`SessionFactory`, so a process can expose several agents
    (e.g. ``/mobile``, ``/analytics``) against the same gateway with
    different ``deployment_label`` values.  ``streaming`` is per-sub-app.
    """
    sub_app = create_app(
        agent,
        session_provider=session_provider,
        client_factory=client_factory,
        elicitation_client_factory=elicitation_client_factory,
        share_client=share_client,
        deployment_label=deployment_label,
        streaming=streaming,
    )
    parent.mount(prefix, sub_app)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_session_factory(
    *,
    agent: Agent,
    session_provider: SessionProvider | None,
    client_factory: ClientFactory | None,
    elicitation_client_factory: ElicitationClientFactory | None,
    share_client: bool,
    deployment_label: str | None,
) -> SessionFactory:
    modes = sum(
        [
            session_provider is not None,
            client_factory is not None,
            share_client,
        ]
    )
    if modes != 1:
        if client_factory is not None and share_client:
            raise ConfigurationError(
                "create_app: 'client_factory=...' and 'share_client=True' are "
                "mutually exclusive. Pick the per-session factory OR reuse the "
                "agent's existing tool_provider.",
            )
        raise ConfigurationError(
            "create_app requires exactly one of 'session_provider=...', "
            "'client_factory=...' (per-session mode), or 'share_client=True' "
            "(reuse the agent's tool_provider). "
            "These options are mutually exclusive. See docs/serving.md.",
        )
    if elicitation_client_factory is not None and client_factory is None:
        raise ConfigurationError(
            "create_app: 'elicitation_client_factory' is additive to "
            "'client_factory'. For advanced WS wiring use a "
            "'session_provider' instead.",
        )

    base_config = _apply_deployment_label(agent.config, deployment_label)

    if session_provider is not None:
        return SessionFactory(
            base_config=base_config,
            t0=agent._t0,
            session_provider=session_provider,
            agent_class=type(agent),
        )
    if share_client:
        if agent._tool_provider is None:
            raise ConfigurationError(
                "create_app: share_client=True requires the agent to have a "
                "tool_provider, but this Agent was constructed without one "
                "(typically because output=JsonOutput(...) or another "
                "single-shot output mode does not need MCP tools). "
                "Either pass an explicit tool_provider when "
                "constructing the Agent, or serve with "
                "'client_factory=...' / 'session_provider=...' instead."
            )
        return SessionFactory(
            base_config=base_config,
            t0=agent._t0,
            shared_tool_provider=agent._tool_provider,
            agent_class=type(agent),
        )
    assert client_factory is not None  # guarded by XOR above
    return SessionFactory(
        base_config=base_config,
        t0=agent._t0,
        client_factory=client_factory,
        elicitation_client_factory=elicitation_client_factory,
        agent_class=type(agent),
    )


def _apply_deployment_label(
    base_config: AgentConfig,
    deployment_label: str | None,
) -> AgentConfig:
    if deployment_label is None:
        return base_config
    merged = {**(base_config.tags or {}), "t0_agents.deployment": deployment_label}
    return dataclasses.replace(base_config, tags=merged)
