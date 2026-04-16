"""Per-session factory for ``(Client, Agent)`` pairs.

Contract §3.5 + FastMCP Q3 mandate that a fresh ``Client`` serve each
logical caller whenever elicitation may fire.  Phase D defaults to this
strict interpretation (one ``Client`` per :meth:`SessionFactory.session`
call) and offers ``share_client=True`` as an explicit opt-out for
deployments that are certain no tool uses elicitation.

The :meth:`SessionFactory.session` async context manager carries the
invariant across transports: its lifetime equals *the caller's
``async with`` block*.  Phase D calls it inside a FastAPI route
(lifetime = one HTTP request); Phase E will call it inside an SSE
streaming endpoint (lifetime = one stream) without changing the shape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider

ClientFactory = Callable[[], Client[Any]]


class SessionFactory:
    """Builds ``(Client, Agent)`` pairs whose lifetime is a caller's session.

    Phase D supports two modes, selected at construction time:

    * **Per-session** — pass ``client_factory``.  Every call to
      :meth:`session` invokes the factory and enters the returned
      ``Client`` for the duration of the caller's ``async with`` block;
      the ``Client`` is closed deterministically on exit (including
      cancellation).  Matches the FastMCP Q3 mandate — each caller
      gets its own elicitation handler slot.
    * **Shared** — pass ``shared_tool_provider``.  The same tool
      provider is handed to every session unchanged; this
      :class:`SessionFactory` does *not* manage its lifecycle.  Callers
      are responsible for entering and closing the underlying client
      themselves (typically by entering the ``Client`` outside
      ``create_app`` or relying on ``Client`` being a FastMCP in-process
      instance that is always connected).

    Exactly one of the two modes must be selected.  The constructor
    raises :class:`ConfigurationError` otherwise.
    """

    def __init__(
        self,
        *,
        base_config: AgentConfig,
        t0: AsyncTensorZeroGateway,
        client_factory: ClientFactory | None = None,
        shared_tool_provider: ToolProvider | None = None,
        agent_class: type[Agent] = Agent,
    ) -> None:
        if (client_factory is None) == (shared_tool_provider is None):
            raise ConfigurationError(
                "SessionFactory requires exactly one of 'client_factory' "
                "(per-session mode) or 'shared_tool_provider' (shared mode)",
            )
        self._base_config = base_config
        self._t0 = t0
        self._client_factory = client_factory
        self._shared_tool_provider = shared_tool_provider
        self._agent_class = agent_class

    @property
    def base_config(self) -> AgentConfig:
        """The configuration applied before per-request tag merging."""
        return self._base_config

    async def startup(self) -> None:
        """Lifecycle hook for :func:`t0_agents.serve.build_lifespan`.

        Phase D is a no-op in both modes.  The shared-mode caller owns
        their client's lifecycle; the per-session mode allocates inside
        :meth:`session`.  Future phases may place an embedded-gateway
        warm-up here without changing callers.
        """
        return None

    async def shutdown(self) -> None:
        """Lifecycle hook for :func:`t0_agents.serve.build_lifespan`.

        Phase D is a no-op — see :meth:`startup`.
        """
        return None

    def probe_tool_provider(self) -> str:
        """Best-effort status string for ``GET /health``.

        Returns ``"per-session"`` when no persistent tool provider
        exists (no way to check without opening a throw-away
        ``Client``).  Returns ``"ok"`` / ``"down"`` when sharing a tool
        provider that exposes a synchronous ``is_connected()`` method
        (``fastmcp.Client`` does, per
        ``.venv/.../fastmcp/client/client.py:412-414``).
        """
        if self._shared_tool_provider is None:
            return "per-session"
        probe = getattr(self._shared_tool_provider, "is_connected", None)
        if probe is None:
            return "per-session"
        try:
            connected = bool(probe())
        except Exception:
            return "down"
        return "ok" if connected else "down"

    @asynccontextmanager
    async def session(
        self,
        *,
        extra_tags: dict[str, str] | None = None,
    ) -> AsyncIterator[Agent]:
        """Yield an :class:`Agent` bound to a session-scoped tool provider.

        ``extra_tags`` are shallow-merged on top of ``base_config.tags``
        before the session-scoped :class:`AgentConfig` is built; per-
        request mutations never touch the base configuration.

        The context manager exits cleanly on
        :class:`asyncio.CancelledError` — when an HTTP caller
        disconnects mid-request FastAPI cancels the task, Python runs
        ``__aexit__``, and any freshly-opened ``Client`` is torn down
        deterministically.
        """
        merged_tags = self._merge_tags(extra_tags)
        session_config = dataclasses.replace(self._base_config, tags=merged_tags)

        if self._shared_tool_provider is not None:
            yield self._agent_class(
                session_config,
                t0=self._t0,
                tool_provider=self._shared_tool_provider,
            )
            return

        assert self._client_factory is not None  # guaranteed by __init__
        client = self._client_factory()
        async with client:
            yield self._agent_class(
                session_config, t0=self._t0, tool_provider=client,
            )

    def _merge_tags(
        self,
        extra_tags: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Shallow-merge ``extra_tags`` on top of ``base_config.tags``.

        Returns ``None`` when both inputs are empty so downstream
        observability receives the same payload it would for a plain
        :meth:`Agent.run` call with no tags configured.
        """
        base = self._base_config.tags or {}
        extra = extra_tags or {}
        if not base and not extra:
            return None
        return {**base, **extra}
