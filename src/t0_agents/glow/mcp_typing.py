""":class:`MCPServer` — type-annotation-driven typed MCP integration.

Two patterns let users declare MCP servers their glow agents should see:

1. **Class-based** (canonical for shared servers).  Subclass
   :class:`MCPServer` with the spec passed as keyword arguments to the
   class definition; the spec is recorded on the class and surfaced via
   ``cls._mcp_spec``::

       class GithubMCP(MCPServer, url="https://api.githubcopilot.com/mcp/"): ...

       @app.agent("devops")
       async def devops(query: str, gh: GithubMCP) -> str:
           \"\"\"DevOps assistant with Github access.\"\"\"

2. **Annotated** (per-tool override).  Instantiate :class:`MCPServer`
   inside an :class:`typing.Annotated` and attach to a parameter::

       @app.agent("calc")
       async def calc(
           msg: str,
           math: Annotated[Any, MCPServer(url="http://localhost:8001/mcp")],
       ) -> str: ...

Either pattern surfaces a :class:`MCPServerSpec` at decoration time;
the App accumulates them and mounts each onto its long-lived in-memory
:class:`fastmcp.FastMCP` server during :meth:`App.build`.

Both transports are supported:

- ``server=<FastMCP instance>`` mounts the in-memory server directly
  (``examples/14_multi_server`` pattern).
- ``url=<str URL>`` is wrapped via :func:`fastmcp.server.create_proxy`
  into a proxy ``FastMCP`` and then mounted.  HTTP, SSE, and any
  ``ClientTransport`` accepted by ``create_proxy`` work transparently.

The composed ``FastMCP("glow-app")`` server feeds ``app._owned_client``,
which is the ``tool_provider`` every glow agent is constructed with.
The result: a single ``list_tools()`` view across every ``@app.tool``
registration plus every ``MCPServer`` spec discovered through
``servers=[...]`` or ``Annotated``/subclass annotations.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, get_args, get_origin, get_type_hints

from fastmcp import FastMCP

from t0_agents.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    import httpx

__all__ = ["MCPServer", "MCPServerSpec"]


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """Immutable descriptor for a typed MCP server registration.

    Exactly one of :attr:`url` or :attr:`server` must be set.  ``url``
    drives the HTTP path (wrapped via ``create_proxy`` at mount time);
    ``server`` is the in-memory escape hatch for users with embedded
    ``FastMCP`` instances.  ``namespace`` becomes the ``namespace=`` kwarg
    on :meth:`FastMCP.mount` — mounted tool names are prefixed with it,
    keeping multi-server setups collision-free.

    ``headers`` and ``auth`` (an :class:`httpx.Auth`) attach per-request
    identity to a ``url`` mount; combining either with an in-memory
    ``server`` is a validation error.  When either is set the mount rides
    an explicit ``transport`` (``"http"`` or ``"sse"``) rather than the
    URL-inferred one, so ``transport`` is load-bearing on the authed path
    (it is otherwise inferred by ``create_proxy``).

    ``prefix`` (default ``True``) applies ``namespace`` as the tool-name
    prefix.  Set ``prefix=False`` to mount unprefixed — ``FastMCP.mount``
    with ``namespace=None`` — so a single-server mount presents tool names
    verbatim; ``namespace`` is then a logical label only.  Keep the default
    for multi-server setups so their tool names stay collision-free.
    """

    url: str | None = None
    server: FastMCP[Any] | None = None
    transport: str = "http"
    namespace: str | None = None
    headers: dict[str, str] | None = None
    auth: httpx.Auth | None = None
    prefix: bool = True


class MCPServer:
    """Dual-personality base for typed MCP server declarations.

    Use as a **base class** with keyword arguments on the class
    statement to record a shared spec::

        class GithubMCP(MCPServer, url="https://api.githubcopilot.com/mcp/"): ...

    Use as an **Annotated metadata instance** for per-parameter overrides::

        Annotated[Any, MCPServer(url="http://localhost:8001/mcp")]

    Both patterns produce an :class:`MCPServerSpec` consumed by
    :class:`~t0_agents.glow.app.App` at decoration time.  Validation
    (exactly one of ``url``/``server``) fires eagerly in both paths so
    misconfiguration surfaces at import / module-load time rather than
    during a live invocation.
    """

    #: Class-level spec set by :meth:`__init_subclass__` for the
    #: class-based pattern.  Instance-level fields live alongside (set
    #: by :meth:`__init__` for the Annotated pattern) — Python does not
    #: object to having both because attribute lookups walk the instance
    #: first.  Marked ``ClassVar`` so dataclass-style introspection
    #: skips it on subclasses.
    _mcp_spec: ClassVar[MCPServerSpec]

    def __init_subclass__(
        cls,
        *,
        url: str | None = None,
        server: FastMCP[Any] | None = None,
        transport: str = "http",
        namespace: str | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        prefix: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        _validate_spec(cls.__name__, url, server, headers, auth)
        cls._mcp_spec = MCPServerSpec(
            url=url,
            server=server,
            transport=transport,
            namespace=namespace,
            headers=headers,
            auth=auth,
            prefix=prefix,
        )

    def __init__(
        self,
        *,
        url: str | None = None,
        server: FastMCP[Any] | None = None,
        transport: str = "http",
        namespace: str | None = None,
        headers: dict[str, str] | None = None,
        auth: httpx.Auth | None = None,
        prefix: bool = True,
    ) -> None:
        _validate_spec("MCPServer", url, server, headers, auth)
        self.url = url
        self.server = server
        self.transport = transport
        self.namespace = namespace
        self.headers = headers
        self.auth = auth
        self.prefix = prefix

    def _to_spec(self) -> MCPServerSpec:
        """Project the Annotated-instance fields to an :class:`MCPServerSpec`.

        Used by :func:`_extract_mcp_specs` when an ``MCPServer`` instance
        appears in an ``Annotated`` metadata tuple, and by ``App(servers=)``
        when an instance is passed directly.  Subclasses with a class-level
        ``_mcp_spec`` use that directly; this helper is for the instance path.
        """
        return MCPServerSpec(
            url=self.url,
            server=self.server,
            transport=self.transport,
            namespace=self.namespace,
            headers=self.headers,
            auth=self.auth,
            prefix=self.prefix,
        )


def _validate_spec(
    name: str,
    url: str | None,
    server: FastMCP[Any] | None,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
) -> None:
    """Validate an ``MCPServer`` spec on both ``__init_subclass__`` and
    ``__init__``: exactly one of ``url``/``server``, and ``headers``/``auth``
    only on the remote ``url`` path (they have no meaning for an in-memory
    ``server`` mount, which is not an HTTP transport)."""
    if url is None and server is None:
        raise ConfigurationError(
            f"{name}: at least one of `url` or `server` must be set on "
            f"an MCPServer (url for HTTP/SSE-style remotes wrapped via "
            f"fastmcp.create_proxy; server for in-memory FastMCP instances)."
        )
    if url is not None and server is not None:
        raise ConfigurationError(
            f"{name}: pass exactly one of `url` or `server` to an "
            f"MCPServer — both were provided.  Use `url` for remote "
            f"transports and `server` for in-memory FastMCP instances."
        )
    if server is not None and (headers is not None or auth is not None):
        raise ConfigurationError(
            f"{name}: headers=/auth= apply only to a remote `url` mount, "
            f"not an in-memory `server` mount."
        )


def _extract_mcp_specs(
    fn: Callable[..., Any],
    hints: dict[str, Any] | None = None,
) -> list[MCPServerSpec]:
    """Walk ``fn``'s parameters and pull out every :class:`MCPServerSpec`.

    Two shapes are recognised:

    1. A parameter annotated directly with a subclass of
       :class:`MCPServer` — the class-level ``_mcp_spec`` is harvested.
    2. A parameter annotated with ``Annotated[X, MCPServer(url=...)]``
       (or any ``Annotated`` whose metadata contains an :class:`MCPServer`
       instance) — every instance's :meth:`_to_spec` is harvested in
       metadata-order.

    Returns the discovered specs in parameter-declaration order, then
    metadata order within each parameter.  Empty list when no
    ``MCPServer`` markers appear.

    ``hints`` may be passed pre-resolved by the caller to avoid a second
    ``get_type_hints`` pass; when ``None``, it is resolved here.  Failure
    to resolve type hints (e.g. unresolvable forward refs) is **not**
    fatal — we simply skip the extraction.  The caller (``_make_decorator``)
    has its own forward-ref enforcement path; this helper is purely
    additive.
    """
    if hints is None:
        try:
            hints = get_type_hints(fn, include_extras=True)
        except (NameError, TypeError):
            return []

    sig = inspect.signature(fn)
    specs: list[MCPServerSpec] = []
    for param_name, _param in sig.parameters.items():
        annotation = hints.get(param_name)
        if annotation is None:
            continue
        specs.extend(_specs_from_annotation(annotation))
    return specs


def _specs_from_annotation(annotation: Any) -> list[MCPServerSpec]:
    """Return every :class:`MCPServerSpec` reachable from one annotation.

    Handles three shapes:

    - ``annotation`` is a subclass of :class:`MCPServer` →
      ``[annotation._mcp_spec]``.
    - ``annotation`` is ``Annotated[..., MCPServer(...), ...]`` →
      ``[meta._to_spec() for meta in metadata if isinstance(meta, MCPServer)]``.
    - Anything else → ``[]``.

    Other ``Annotated`` metadata entries (Guardrails, plain markers) are
    silently ignored — they belong to other extractors.
    """
    if isinstance(annotation, type) and issubclass(annotation, MCPServer):
        return [annotation._mcp_spec]
    if get_origin(annotation) is Annotated:
        metadata = get_args(annotation)[1:]
        return [m._to_spec() for m in metadata if isinstance(m, MCPServer)]
    return []


def _is_mcp_parameter(annotation: Any) -> bool:
    """``True`` when ``annotation`` carries any :class:`MCPServer` metadata.

    Used by :mod:`t0_agents.glow.decorators` to skip MCP-typed parameters
    when walking ``@app.agent`` signatures — those parameters are pure
    decoration metadata and have no runtime binding (the wrapper's
    ``__call__`` only accepts the user message; the parameter slot is
    never populated).
    """
    if annotation is None:
        return False
    if isinstance(annotation, type) and issubclass(annotation, MCPServer):
        return True
    if get_origin(annotation) is Annotated:
        metadata = get_args(annotation)[1:]
        return any(isinstance(m, MCPServer) for m in metadata)
    return False
