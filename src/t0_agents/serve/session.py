"""Per-session factory for ``(Client, Agent)`` pairs.

A fresh ``Client`` must serve each logical caller whenever elicitation
may fire: ``fastmcp.Client`` binds exactly one ``elicitation_handler``
at construction, so a shared Client cannot demultiplex concurrent
elicitation requests back to the originating callers.  This factory
enforces that invariant by default and exposes a public
:class:`SessionProvider` Protocol for advanced wiring (auth,
multi-tenant routing, telemetry) — see ``docs/streaming.md`` for the
full recipe.

Modes (mutually exclusive; exactly one must be selected):

1. **``session_provider``** — caller-supplied object implementing
   :class:`SessionProvider`.  Advanced hook for auth, multi-tenant
   routing, telemetry, custom elicitation wiring.
2. **``client_factory``** — zero-arg callable returning a fresh
   ``Client``.  Internally wrapped in a :class:`CallableSessionProvider`.
   Optionally paired with ``elicitation_client_factory`` for the
   streaming WebSocket path, which threads a per-connection handler
   into each Client.
3. **``shared_tool_provider``** — reuse a single ``Client`` across every
   session.  Does **not** support streaming elicitation;
   :meth:`session` raises when ``elicitation_handler`` is supplied.

The :meth:`SessionFactory.session` async context manager carries the
invariant across transports: its lifetime equals *the caller's
``async with`` block*.  The non-streaming HTTP routes use it inside
FastAPI routes (one HTTP request); the streaming SSE / WS endpoints use
it inside one stream / one WebSocket connection without changing the
shape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from fastmcp import Client
from fastmcp.client.elicitation import ElicitationHandler
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.types import ToolProvider

ClientFactory = Callable[[], Client[Any]]
ElicitationClientFactory = Callable[[ElicitationHandler], Client[Any]]


@runtime_checkable
class SessionProvider(Protocol):
    """Per-session tool-Client construction hook.

    The framework calls :meth:`build_client` at the start of every session.
    The returned async context manager owns the ``Client``'s lifecycle —
    ``__aenter__`` connects it, ``__aexit__`` closes it deterministically
    (including on :class:`asyncio.CancelledError`).

    Default path — pass ``client_factory=...`` (optionally with
    ``elicitation_client_factory=...``) to :func:`create_app`.  The
    framework wraps them in a :class:`CallableSessionProvider`.

    Advanced path — implement this Protocol directly when you need to hook
    auth, multi-tenant routing, telemetry, or custom elicitation wiring
    into session construction.  See ``docs/streaming.md``.

    ``elicitation_handler is None`` signals the non-streaming
    ``POST /chat`` path — the provider is free to return a Client that
    has no elicitation handler registered.  ``elicitation_handler`` being
    set signals the streaming WS path — the provider must register it on
    the returned Client or raise :class:`ConfigurationError`.
    """

    def build_client(
        self,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AbstractAsyncContextManager[Client[Any]]: ...


class CallableSessionProvider:
    """Default :class:`SessionProvider` adapter built from one or two lambdas.

    ``client_factory`` is the zero-arg factory used when no elicitation
    is required.  ``elicitation_client_factory`` — when supplied —
    receives the per-connection handler and is called whenever
    :meth:`build_client` is invoked with ``elicitation_handler`` set
    (the streaming WebSocket path).

    When the caller asks for an elicitation-aware Client and only
    ``client_factory`` is configured, the adapter raises
    :class:`ConfigurationError` pointing to the missing slot.
    """

    __slots__ = ("_client_factory", "_elicitation_client_factory")

    def __init__(
        self,
        client_factory: ClientFactory,
        elicitation_client_factory: ElicitationClientFactory | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._elicitation_client_factory = elicitation_client_factory

    def build_client(
        self,
        *,
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AbstractAsyncContextManager[Client[Any]]:
        if elicitation_handler is None:
            return self._client_factory()
        if self._elicitation_client_factory is None:
            raise ConfigurationError(
                "Streaming elicitation requires 'elicitation_client_factory' "
                "on create_app(). See docs/streaming.md 'Simple path'.",
            )
        return self._elicitation_client_factory(elicitation_handler)


class SessionFactory:
    """Builds ``(Client, Agent)`` pairs whose lifetime is a caller's session.

    Exactly one of ``session_provider`` / ``client_factory`` /
    ``shared_tool_provider`` must be passed; :class:`ConfigurationError`
    is raised otherwise.  ``elicitation_client_factory`` is additive —
    only valid together with ``client_factory``.
    """

    def __init__(
        self,
        *,
        base_config: AgentConfig,
        t0: AsyncTensorZeroGateway,
        session_provider: SessionProvider | None = None,
        client_factory: ClientFactory | None = None,
        elicitation_client_factory: ElicitationClientFactory | None = None,
        shared_tool_provider: ToolProvider | None = None,
        agent_class: type[Agent] = Agent,
    ) -> None:
        modes = sum(
            [
                session_provider is not None,
                client_factory is not None,
                shared_tool_provider is not None,
            ]
        )
        if modes != 1:
            raise ConfigurationError(
                "SessionFactory requires exactly one of 'session_provider', "
                "'client_factory', or 'shared_tool_provider' (three-way XOR)",
            )
        if elicitation_client_factory is not None and client_factory is None:
            raise ConfigurationError(
                "'elicitation_client_factory' is additive to 'client_factory'; "
                "use 'session_provider' for advanced WS wiring instead",
            )

        self._base_config = base_config
        self._t0 = t0
        self._shared_tool_provider = shared_tool_provider
        self._agent_class = agent_class

        self._provider: SessionProvider | None
        if session_provider is not None:
            if not isinstance(session_provider, SessionProvider):
                raise ConfigurationError(
                    f"session_provider must implement the SessionProvider "
                    f"protocol (build_client), got "
                    f"{type(session_provider).__name__}",
                )
            self._provider = session_provider
        elif client_factory is not None:
            self._provider = CallableSessionProvider(
                client_factory=client_factory,
                elicitation_client_factory=elicitation_client_factory,
            )
        else:
            self._provider = None  # shared-mode path

    @property
    def base_config(self) -> AgentConfig:
        """The configuration applied before per-request tag merging."""
        return self._base_config

    async def startup(self) -> None:
        """Lifecycle hook for :func:`t0_agents.serve.build_lifespan`.

        Returns immediately in every mode today.  The shared-mode caller
        owns their client's lifecycle; per-session / provider modes
        allocate inside :meth:`session`.  Subclasses or future extensions
        may place an embedded-gateway warm-up here without changing
        callers.
        """
        return None

    async def shutdown(self) -> None:
        """Lifecycle hook for :func:`t0_agents.serve.build_lifespan`."""
        return None

    def probe_tool_provider(self) -> str:
        """Best-effort status string for ``GET /health``.

        * ``"per-session"`` when no persistent tool provider exists.
        * ``"ok"`` / ``"down"`` when a shared provider exposes
          ``is_connected()`` (``fastmcp.Client`` does, per
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
        elicitation_handler: ElicitationHandler | None = None,
    ) -> AsyncIterator[Agent]:
        """Yield an :class:`Agent` bound to a session-scoped tool provider.

        ``extra_tags`` are shallow-merged on top of ``base_config.tags``.
        ``elicitation_handler`` — when set — threads through to the
        :class:`SessionProvider`'s :meth:`build_client`; the WS streaming
        route passes its per-connection
        :class:`~t0_agents.core.elicitation_forwarder.ElicitationForwarder.handler`
        here.  Shared-tool-provider mode raises :class:`ConfigurationError`
        when ``elicitation_handler`` is set — streaming elicitation
        requires a per-session Client.

        The context manager exits cleanly on
        :class:`asyncio.CancelledError` — when an HTTP / WS caller
        disconnects mid-request FastAPI cancels the task, Python runs
        ``__aexit__``, and any freshly-opened ``Client`` is torn down
        deterministically.
        """
        merged_tags = self._merge_tags(extra_tags)
        session_config = dataclasses.replace(self._base_config, tags=merged_tags)

        if self._shared_tool_provider is not None:
            if elicitation_handler is not None:
                raise ConfigurationError(
                    "Streaming elicitation is incompatible with "
                    "'shared_tool_provider' mode — each caller needs its own "
                    "elicitation handler.  Use 'client_factory' "
                    "+ 'elicitation_client_factory' or a 'session_provider'.",
                )
            yield self._agent_class(
                session_config,
                t0=self._t0,
                tool_provider=self._shared_tool_provider,
            )
            return

        assert self._provider is not None  # guaranteed by __init__ XOR
        async with self._provider.build_client(
            elicitation_handler=elicitation_handler
        ) as client:
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
