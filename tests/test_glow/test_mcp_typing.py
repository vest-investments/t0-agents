"""Unit + integration tests for :mod:`t0_agents.glow.mcp_typing` and the
M4 typed-MCP integration in :class:`~t0_agents.glow.App`.

Unit tests exercise the ``MCPServer`` dual-personality, the
``MCPServerSpec`` validation, and the parameter-walk extraction without
touching the live TZ gateway.  The integration test mounts two in-memory
``FastMCP`` servers via ``App(servers=[...])`` and asserts the agent
sees the combined toolset through ``app._owned_client.list_tools()``.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import pytest
from fastmcp import FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, MCPServer
from t0_agents.glow.mcp_typing import (
    MCPServerSpec,
    _extract_mcp_specs,
    _is_mcp_parameter,
)


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


# ---------------------------------------------------------------------------
# Module-level MCPServer subclasses for the forward-ref-resolves tests.
# Test-local subclasses can't be used in function annotations because
# ``get_type_hints`` walks the function's globals (the test module's
# globals); enclosing-test locals are not in scope.  Mirrors the pattern
# established in ``tests/test_glow/test_output_inference.py``.
# ---------------------------------------------------------------------------


class _GithubMCP(MCPServer, url="https://api.githubcopilot.com/mcp/"): ...


class _ParamMCP(MCPServer, url="https://param/mcp"): ...


class _FirstMCP(MCPServer, url="https://first/mcp"): ...


class _SecondMCP(MCPServer, url="https://second/mcp"): ...


# ---------------------------------------------------------------------------
# MCPServer class form (``__init_subclass__``)
# ---------------------------------------------------------------------------


def test_subclass_with_url_records_spec_on_class() -> None:
    class GithubMCP(MCPServer, url="https://api.githubcopilot.com/mcp/"): ...

    assert GithubMCP._mcp_spec.url == "https://api.githubcopilot.com/mcp/"
    assert GithubMCP._mcp_spec.server is None
    assert GithubMCP._mcp_spec.transport == "http"
    assert GithubMCP._mcp_spec.namespace is None


def test_subclass_with_in_memory_server_records_spec() -> None:
    server: FastMCP[Any] = FastMCP("test-server")

    class MathTools(MCPServer, server=server, namespace="math"): ...

    assert MathTools._mcp_spec.server is server
    assert MathTools._mcp_spec.url is None
    assert MathTools._mcp_spec.namespace == "math"


def test_subclass_without_url_or_server_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="at least one of"):

        class BadMCP(MCPServer): ...


def test_subclass_with_both_url_and_server_raises_configuration_error() -> None:
    server: FastMCP[Any] = FastMCP("test-server")
    with pytest.raises(ConfigurationError, match="exactly one of"):

        class DualMCP(
            MCPServer,
            url="https://example.com/mcp",
            server=server,
        ): ...


# ---------------------------------------------------------------------------
# MCPServer instance form (Annotated metadata)
# ---------------------------------------------------------------------------


def test_instance_with_url_to_spec_returns_spec() -> None:
    meta = MCPServer(url="https://example.com/mcp", namespace="ext")
    spec = meta._to_spec()
    assert isinstance(spec, MCPServerSpec)
    assert spec.url == "https://example.com/mcp"
    assert spec.namespace == "ext"


def test_instance_without_url_or_server_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="at least one of"):
        MCPServer()


def test_instance_with_both_url_and_server_raises_configuration_error() -> None:
    server: FastMCP[Any] = FastMCP("test-server")
    with pytest.raises(ConfigurationError, match="exactly one of"):
        MCPServer(url="https://example.com/mcp", server=server)


# ---------------------------------------------------------------------------
# _extract_mcp_specs / _is_mcp_parameter
# ---------------------------------------------------------------------------


def test_extract_mcp_specs_finds_class_form() -> None:
    async def devops(query: str, gh: _GithubMCP) -> str:  # type: ignore[empty-body]
        """Devops."""

    specs = _extract_mcp_specs(devops)
    assert len(specs) == 1
    assert specs[0].url == "https://api.githubcopilot.com/mcp/"


def test_extract_mcp_specs_finds_annotated_form() -> None:
    async def calc(  # type: ignore[empty-body]
        msg: str,
        math: Annotated[Any, MCPServer(url="http://localhost/mcp")],
    ) -> str:
        """Calculator."""

    specs = _extract_mcp_specs(calc)
    assert len(specs) == 1
    assert specs[0].url == "http://localhost/mcp"


def test_extract_mcp_specs_ignores_unrelated_annotations() -> None:
    async def plain(msg: str, x: int = 0) -> str:  # type: ignore[empty-body]
        """Nothing MCP-related."""

    assert _extract_mcp_specs(plain) == []


def test_extract_mcp_specs_collects_multiple_in_declaration_order() -> None:
    async def multi(  # type: ignore[empty-body]
        msg: str,
        first: _FirstMCP,
        second: _SecondMCP,
    ) -> str:
        """Two MCPs."""

    specs = _extract_mcp_specs(multi)
    urls = [s.url for s in specs]
    assert urls == ["https://first/mcp", "https://second/mcp"]


def test_is_mcp_parameter_class_form() -> None:
    class FooMCP(MCPServer, url="https://foo/mcp"): ...

    assert _is_mcp_parameter(FooMCP)


def test_is_mcp_parameter_annotated_form() -> None:
    annotation = Annotated[Any, MCPServer(url="https://foo/mcp")]
    assert _is_mcp_parameter(annotation)


def test_is_mcp_parameter_false_for_plain_types() -> None:
    assert not _is_mcp_parameter(str)
    assert not _is_mcp_parameter(int)
    assert not _is_mcp_parameter(None)


# ---------------------------------------------------------------------------
# App(servers=...) wiring
# ---------------------------------------------------------------------------


def test_app_accepts_servers_kwarg_and_seeds_mcp_specs() -> None:
    class FooMCP(MCPServer, url="https://foo/mcp", namespace="foo"): ...
    class BarMCP(MCPServer, url="https://bar/mcp", namespace="bar"): ...

    app = App(t0=_fake_t0(), servers=[FooMCP, BarMCP])
    namespaces = [spec.namespace for spec in app._mcp_specs]
    assert namespaces == ["foo", "bar"]


def test_app_rejects_non_mcp_class_in_servers() -> None:
    class NotAnMCP:
        """Not an MCPServer subclass."""

    with pytest.raises(ConfigurationError, match="subclass of MCPServer"):
        App(t0=_fake_t0(), servers=[cast(Any, NotAnMCP)])


def test_decorator_appends_mcp_specs_from_parameters() -> None:
    app = App(t0=_fake_t0())

    @app.agent("inspector")
    async def inspector(query: str, mcp: _ParamMCP) -> str:  # type: ignore[empty-body]
        """Inspector that uses MCP."""

    urls = [spec.url for spec in app._mcp_specs]
    assert "https://param/mcp" in urls


# ---------------------------------------------------------------------------
# End-to-end mount via App.lifespan()
# ---------------------------------------------------------------------------
#
# This case exercises the full M4 mount machinery against in-memory
# ``FastMCP`` servers — no TZ gateway needed, so it stays in the offline
# suite.  We assert that both mounted servers' tools surface on
# ``app._owned_client.list_tools()`` with the expected namespacing.


def _build_math_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("math-tools")

    @server.tool
    async def add(a: int, b: int) -> int:
        """Sum two integers."""
        return a + b

    return server


def _build_text_server() -> FastMCP[Any]:
    server: FastMCP[Any] = FastMCP("text-tools")

    @server.tool
    async def length(value: str) -> int:
        """Return the length of a string."""
        return len(value)

    return server


# Module-level: must outlive the lifespan() context manager.
_MATH_SERVER = _build_math_server()
_TEXT_SERVER = _build_text_server()


class _MathToolsMCP(MCPServer, server=_MATH_SERVER, namespace="math"): ...
class _TextToolsMCP(MCPServer, server=_TEXT_SERVER, namespace="text"): ...


async def test_app_with_two_mcp_servers_lists_both_toolsets_after_lifespan() -> None:
    """``App(servers=[MathTools, TextTools])`` mounts both in-memory
    FastMCP servers before entering ``_owned_client``; ``list_tools``
    returns the namespaced union of every tool from every mounted
    server.

    Offline-safe because no inference runs — only the FastMCP mount +
    Client introspection are exercised.
    """
    app = App(t0=_fake_t0(), servers=[_MathToolsMCP, _TextToolsMCP])

    async with app.lifespan():
        tools = await app._owned_client.list_tools()

    tool_names = {t.name for t in tools}
    # ``FastMCP.mount(namespace="math")`` prefixes ``add`` → ``math_add``.
    assert "math_add" in tool_names
    assert "text_length" in tool_names
