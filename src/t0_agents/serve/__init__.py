"""HTTP serving layer for ``t0-agents``.

Requires the ``serve`` extra::

    pip install 't0-agents[serve]'

Public surface:

* :func:`create_app` — wrap an :class:`~t0_agents.core.Agent` in a
  FastAPI app with ``/chat``, ``/feedback``, ``/health``, ``/info``.
* :func:`mount_agent` — mount a per-agent sub-app under a URL prefix
  on an existing FastAPI application.
* :class:`SessionFactory` — advanced: per-session ``(Client, Agent)``
  builder that ``create_app`` instantiates internally.  Exposed so
  integration tests and custom wiring can reuse the same contract.

Elicitation is not piped to live callers in Phase D (the serving layer
defaults to auto-decline via
:class:`~t0_agents.core.ElicitationForwarder`); Phase E ships the SSE /
WebSocket pipe.
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

from t0_agents.serve.app import create_app, mount_agent
from t0_agents.serve.session import ClientFactory, SessionFactory

__all__ = [
    "ClientFactory",
    "SessionFactory",
    "create_app",
    "mount_agent",
]
