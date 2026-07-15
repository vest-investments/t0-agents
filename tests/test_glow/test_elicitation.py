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

import asyncio
from contextlib import suppress
from typing import Any, cast

import pytest
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.elicitation_forwarder import ElicitationForwarder
from t0_agents.glow import App, MCPServer
from t0_agents.mcp.elicitation import auto_approve_handler, auto_decline_handler


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
    ``ctx.elicit``, the forwarder emits a wire-typed request envelope, the
    'remote' side resolves accept, and the tool observes the value."""
    requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def on_request(envelope: dict[str, Any]) -> None:
        await requests.put(envelope)

    forwarder = ElicitationForwarder(on_request=on_request, default_timeout=5.0)
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=forwarder.handler)

    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        call = asyncio.create_task(client.call_tool("hitl_place", {}))
        try:
            envelope = await asyncio.wait_for(requests.get(), timeout=2.0)
            assert envelope["type"] == "elicitation_request"
            assert envelope["response_type"] == "bool"
            assert envelope["schema"] is not None
            await forwarder.resolve(
                envelope["elicitation_id"], {"action": "accept", "content": {"value": True}}
            )
            result = await asyncio.wait_for(call, timeout=2.0)
        finally:
            # Don't leak the tool task if an assertion above fails.
            if not call.done():
                call.cancel()
                with suppress(asyncio.CancelledError):
                    await call

    assert result.data == "placed"
    assert forwarder.pending == {}


async def test_app_forwarder_round_trip_decline() -> None:
    """Same round-trip, remote declines → the tool takes its else branch."""
    requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def on_request(envelope: dict[str, Any]) -> None:
        await requests.put(envelope)

    forwarder = ElicitationForwarder(on_request=on_request, default_timeout=5.0)
    app = App(t0=_fake_t0(), servers=[_HitlMCP], elicitation_handler=forwarder.handler)

    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        call = asyncio.create_task(client.call_tool("hitl_place", {}))
        try:
            envelope = await asyncio.wait_for(requests.get(), timeout=2.0)
            await forwarder.resolve(envelope["elicitation_id"], {"action": "decline"})
            result = await asyncio.wait_for(call, timeout=2.0)
        finally:
            # Don't leak the tool task if an assertion above fails.
            if not call.done():
                call.cancel()
                with suppress(asyncio.CancelledError):
                    await call

    assert result.data == "skipped"
    assert forwarder.pending == {}
