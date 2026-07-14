"""HTTP serving layer for ``t0-agents``.

Requires the ``serve`` extra::

    pip install 't0-agents[serve]'

Public surface:

* :func:`create_app` — wrap an :class:`~t0_agents.core.Agent` in a
  FastAPI app with ``/chat``, ``/feedback``, ``/health``, ``/info``
  (plus ``/chat/stream`` SSE and ``/chat/ws`` WebSocket when
  ``streaming=True``).
* :func:`mount_agent` — mount a per-agent sub-app under a URL prefix
  on an existing FastAPI application.
* :class:`SessionFactory` — advanced: per-session ``(Client, Agent)``
  builder that ``create_app`` instantiates internally.
* :class:`SessionProvider` Protocol + :class:`CallableSessionProvider`
  adapter — per-session tool-Client construction hook.  Implement
  :class:`SessionProvider` directly for auth / multi-tenant routing /
  bespoke elicitation wiring; use the ``client_factory`` /
  ``elicitation_client_factory`` kwargs on :func:`create_app` for the
  simple path.  See ``docs/streaming.md``.

Streaming elicitation (``streaming=True``) flows through
:class:`~t0_agents.core.ElicitationForwarder`; the default non-streaming
HTTP surface declines elicitation automatically when no remote resolver
is wired.
"""

from __future__ import annotations

from t0_agents.core.exceptions import T0AgentsError

try:
    from fastapi import FastAPI as _FastAPI  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on install profile
    raise T0AgentsError(
        "t0_agents.serve requires the 'serve' extra. "
        "Install with: pip install 't0-agents[serve]'",
    ) from exc

from t0_agents.serve.app import create_app, create_app_from_glow, mount_agent
from t0_agents.serve.session import (
    CallableSessionProvider,
    ClientFactory,
    ElicitationClientFactory,
    SessionFactory,
    SessionProvider,
)

__all__ = [
    "CallableSessionProvider",
    "ClientFactory",
    "ElicitationClientFactory",
    "SessionFactory",
    "SessionProvider",
    "create_app",
    "create_app_from_glow",
    "mount_agent",
]
