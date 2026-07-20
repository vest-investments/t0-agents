"""Elicitation-handler seam on the glow ``App``-owned Client.

Glow owns a single composite ``fastmcp.Client`` (``app._owned_client``).
``App(elicitation_handler=...)`` binds a human-in-the-loop resolver onto
that client so a mounted tool's ``ctx.elicit(...)`` reaches a real handler
instead of raising "Elicitation not supported".

Every test drives through the **real** FastMCP boundary — an in-process
server whose tool calls ``ctx.elicit``, mounted on the App, invoked via
``app._owned_client`` — never a hand-built client or a mock.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, MCPServer
from t0_agents.mcp.elicitation import auto_approve_handler, auto_decline_handler
from t0_agents.testing import ScriptedElicitation


def _fake_t0() -> AsyncTensorZeroGateway:
    # No inference runs in these tests — only the FastMCP mount + a tool
    # round-trip through the owned Client — so a bare object stands in.
    return cast(AsyncTensorZeroGateway, object())


_HITL_SERVER: FastMCP[Any] = FastMCP("hitl-tools")


@_HITL_SERVER.tool
async def approve(ctx: Context) -> str:
    """Binary approval (``response_type=None``)."""
    result = await ctx.elicit("Approve?", response_type=None)
    return "approved" if result.action == "accept" else "rejected"


@_HITL_SERVER.tool
async def place(ctx: Context) -> str:
    """Bool-scalar confirm — the trade-confirm shape (hidden ``value``)."""
    result = await ctx.elicit("Place the order?", response_type=bool)
    if result.action == "accept" and result.data is True:
        return "placed"
    return "skipped"


class _HitlMCP(MCPServer, server=_HITL_SERVER, namespace="hitl"): ...


async def test_app_binds_reference_handler_auto_approve() -> None:
    """``elicitation_handler=`` reaches the owned Client — approval path."""
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=auto_approve_handler)
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        result = await client.call_tool("hitl_approve", {})
    assert result.data == "approved"


async def test_app_binds_reference_handler_auto_decline() -> None:
    """``elicitation_handler=`` reaches the owned Client — decline path."""
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=auto_decline_handler)
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        result = await client.call_tool("hitl_approve", {})
    assert result.data == "rejected"


async def test_app_without_handler_keeps_unsupported_default() -> None:
    """No ``elicitation_handler`` → the owned Client offers no elicitation
    capability, so ``ctx.elicit`` surfaces as a ``ToolError`` — the
    pre-M14 default, unchanged."""
    app = App(t0=_fake_t0(), servers=[_HitlMCP])
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        with pytest.raises(ToolError, match="Elicitation not supported"):
            await client.call_tool("hitl_approve", {})


async def test_app_forwarder_round_trip_accept() -> None:
    """A full HITL round-trip through the owned Client: the tool blocks in
    ``ctx.elicit``, the scripted handler emits a wire-typed request envelope
    and accepts it, and the tool observes the value."""
    scripted = ScriptedElicitation([{"action": "accept", "content": {"value": True}}])
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=scripted.handler)

    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        result = await client.call_tool("hitl_place", {})

    assert result.data == "placed"
    assert scripted.received[0]["type"] == "elicitation_request"
    assert scripted.received[0]["response_type"] == "bool"
    assert scripted.received[0]["schema"] is not None


async def test_app_forwarder_round_trip_decline() -> None:
    """Same round-trip, the handler declines → the tool takes its else branch."""
    scripted = ScriptedElicitation(["decline"])
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=scripted.handler)

    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        result = await client.call_tool("hitl_place", {})

    assert result.data == "skipped"
