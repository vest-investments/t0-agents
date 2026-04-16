"""Top-level FastAPI factory — one-line wrap of an ``Agent`` in HTTP.

The public surface is :func:`create_app` and :func:`mount_agent`.  Both
take an existing :class:`Agent` plus *exactly one* of:

* ``client_factory`` — a zero-arg callable that builds a fresh
  ``fastmcp.Client`` for every session.  Default, per-session mode.
* ``share_client=True`` — reuse ``agent.tool_provider`` across every
  session.  Safe only when no tool uses elicitation; the caller owns
  the ``Client``'s open/close lifecycle.

Supplying both or neither raises :class:`ConfigurationError` at startup
— no silent ambiguity.

``deployment_label``, when set, is merged into ``AgentConfig.tags`` as
``t0_agents.deployment=<label>`` before any session is built, so every
inference record carries it automatically (resolved T0 Q3).
"""

from __future__ import annotations

import dataclasses

from fastapi import FastAPI

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.serve.lifespan import build_lifespan
from t0_agents.serve.routes import build_router
from t0_agents.serve.session import ClientFactory, SessionFactory


def create_app(
    agent: Agent,
    *,
    client_factory: ClientFactory | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
) -> FastAPI:
    """Build a ``FastAPI`` application that serves ``agent`` over HTTP.

    Raises :class:`ConfigurationError` when neither or both of
    ``client_factory`` / ``share_client`` is supplied.  Does not start
    the server — the caller runs ``uvicorn`` (or an equivalent ASGI
    runner) against the returned app.
    """
    session_factory = _build_session_factory(
        agent=agent,
        client_factory=client_factory,
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
    return app


def mount_agent(
    parent: FastAPI,
    agent: Agent,
    prefix: str,
    *,
    client_factory: ClientFactory | None = None,
    share_client: bool = False,
    deployment_label: str | None = None,
) -> None:
    """Mount a per-agent sub-app under ``prefix`` on an existing FastAPI app.

    Each mounted sub-app runs its own lifespan and its own
    :class:`SessionFactory`, so a process can expose several agents
    (e.g. ``/mobile``, ``/analytics``) against the same gateway with
    different ``deployment_label`` values.
    """
    sub_app = create_app(
        agent,
        client_factory=client_factory,
        share_client=share_client,
        deployment_label=deployment_label,
    )
    parent.mount(prefix, sub_app)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_session_factory(
    *,
    agent: Agent,
    client_factory: ClientFactory | None,
    share_client: bool,
    deployment_label: str | None,
) -> SessionFactory:
    if client_factory is None and not share_client:
        raise ConfigurationError(
            "create_app requires either 'client_factory=...' (per-session mode) "
            "or 'share_client=True' (reuse the agent's tool_provider). "
            "See docs/serving.md for the trade-offs.",
        )
    if client_factory is not None and share_client:
        raise ConfigurationError(
            "create_app: 'client_factory=...' and 'share_client=True' are "
            "mutually exclusive. Pick the per-session factory OR reuse the "
            "agent's existing tool_provider.",
        )

    base_config = _apply_deployment_label(agent.config, deployment_label)

    if share_client:
        return SessionFactory(
            base_config=base_config,
            t0=agent._t0,
            shared_tool_provider=agent._tool_provider,
            agent_class=type(agent),
        )
    assert client_factory is not None  # guarded above
    return SessionFactory(
        base_config=base_config,
        t0=agent._t0,
        client_factory=client_factory,
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
