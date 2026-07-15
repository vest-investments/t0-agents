"""Per-mount namespace-prefix opt-out (``MCPServer(prefix=False)``).

By default a mount's tool names are prefixed with its ``namespace``
(``namespace_tool``), keeping multi-server setups collision-free.
``prefix=False`` mounts unprefixed — the single-server / legacy-name case
— while multi-server prefixing is unchanged (the regression pin below).

An unprefixed mount has no prefix to filter on, so passing it into
``tools=`` is a loud error directing to name/glob selection.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp import FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App, MCPServer
from t0_agents.glow.app import _mcp_server_namespace


def _fake_t0() -> AsyncTensorZeroGateway:
    return cast(AsyncTensorZeroGateway, object())


_SERVER_A: FastMCP[Any] = FastMCP("srv-a")


@_SERVER_A.tool
async def alpha() -> str:
    """Tool alpha."""
    return "a"


_SERVER_B: FastMCP[Any] = FastMCP("srv-b")


@_SERVER_B.tool
async def beta() -> str:
    """Tool beta."""
    return "b"


async def _tool_names(app: App) -> set[str]:
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        return {t.name for t in await client.list_tools()}


async def test_single_server_prefix_false_is_unprefixed() -> None:
    app = App(t0=_fake_t0(), servers=[MCPServer(server=_SERVER_A, namespace="a", prefix=False)])
    names = await _tool_names(app)
    assert "alpha" in names
    assert "a_alpha" not in names


async def test_single_server_default_is_prefixed() -> None:
    app = App(t0=_fake_t0(), servers=[MCPServer(server=_SERVER_A, namespace="a")])
    names = await _tool_names(app)
    assert "a_alpha" in names
    assert "alpha" not in names


async def test_multi_server_prefixing_unchanged() -> None:
    """Regression pin: default (``prefix=True``) multi-server mounts keep
    prefixing identically."""
    app = App(
        t0=_fake_t0(),
        servers=[
            MCPServer(server=_SERVER_A, namespace="a"),
            MCPServer(server=_SERVER_B, namespace="b"),
        ],
    )
    names = await _tool_names(app)
    assert {"a_alpha", "b_beta"} <= names
    assert "alpha" not in names
    assert "beta" not in names


async def test_prefix_optout_is_per_mount() -> None:
    """The opt-out is per mount: a prefixed and an unprefixed server coexist."""
    app = App(
        t0=_fake_t0(),
        servers=[
            MCPServer(server=_SERVER_A, namespace="a"),
            MCPServer(server=_SERVER_B, namespace="b", prefix=False),
        ],
    )
    names = await _tool_names(app)
    assert "a_alpha" in names  # prefixed mount unchanged
    assert "beta" in names  # opted-out mount unprefixed
    assert "b_beta" not in names


def test_mcp_server_namespace_rejects_prefix_false_entry() -> None:
    entry = MCPServer(server=_SERVER_A, namespace="a", prefix=False)
    with pytest.raises(ConfigurationError, match="prefix=False"):
        _mcp_server_namespace("agent", entry)


async def test_prefix_false_server_in_tools_raises_at_build() -> None:
    """Passing an unprefixed server into ``tools=`` fails at build with a
    message pointing to name/glob selection."""
    unprefixed = MCPServer(server=_SERVER_A, namespace="a", prefix=False)
    app = App(t0=_fake_t0(), servers=[unprefixed])

    @app.agent("picky", tools=[unprefixed])
    async def picky(query: str) -> str:  # type: ignore[empty-body]
        """Agent whose tools= points at an unprefixed server."""

    with pytest.raises(ConfigurationError, match="prefix=False"):
        async with app.lifespan():
            pass
