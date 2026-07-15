"""Unit tests for :class:`ElicitationForwarder`."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import pytest
from fastmcp.client.elicitation import ElicitResult

from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    RequestEnvelope,
)


async def test_empty_forwarder_auto_declines() -> None:
    """Empty constructor auto-declines every elicitation."""
    forwarder = ElicitationForwarder()
    result = await forwarder.handler("proceed?", None, None, None)
    assert result.action == "decline"


async def test_handler_pushes_envelope_and_awaits_resolve() -> None:
    received: list[RequestEnvelope] = []

    async def on_request(env: RequestEnvelope) -> None:
        received.append(env)

    forwarder = ElicitationForwarder(on_request=on_request)

    async def resolve_soon() -> None:
        # Wait for the request envelope, then resolve.
        while not received:
            await asyncio.sleep(0)
        elicit_id = received[0]["elicitation_id"]
        await forwarder.resolve(
            elicit_id, {"action": "accept", "content": {"value": 42}}
        )

    resolver = asyncio.create_task(resolve_soon())
    try:
        result = await forwarder.handler("pick a number", None, _FakeParams(), None)
    finally:
        resolver.cancel()
        with suppress(asyncio.CancelledError):
            await resolver

    assert result.action == "accept"
    assert result.content == {"value": 42}
    assert received[0]["type"] == "elicitation_request"
    assert received[0]["message"] == "pick a number"
    assert received[0]["schema"] == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    # response_type arg was None (binary approval) → stays None.
    assert received[0]["response_type"] is None
    # Pending dict is drained after resolve.
    assert forwarder.pending == {}


async def test_handler_normalizes_response_type_from_schema() -> None:
    """A scalar ``response_type`` is derived from the requested schema, not
    from FastMCP's unstable generated wrapper-class name."""
    received: list[RequestEnvelope] = []

    async def on_request(env: RequestEnvelope) -> None:
        received.append(env)

    forwarder = ElicitationForwarder(on_request=on_request)

    class ScalarElicitationType:  # mirrors FastMCP's generated wrapper name
        pass

    async def resolve_soon() -> None:
        while not received:
            await asyncio.sleep(0)
        await forwarder.resolve(
            received[0]["elicitation_id"],
            {"action": "accept", "content": {"value": 7}},
        )

    resolver = asyncio.create_task(resolve_soon())
    try:
        await forwarder.handler(
            "pick a number", ScalarElicitationType, _FakeParams(), None
        )
    finally:
        resolver.cancel()
        with suppress(asyncio.CancelledError):
            await resolver

    # _FakeParams' schema wraps an integer → "int", not "ScalarElicitationType".
    assert received[0]["response_type"] == "int"


async def test_handler_timeout_declines() -> None:
    async def on_request(env: RequestEnvelope) -> None:
        pass  # never resolved

    forwarder = ElicitationForwarder(on_request=on_request, default_timeout=0.05)
    result = await forwarder.handler("timeout test", None, None, None)
    assert result.action == "decline"
    assert forwarder.pending == {}


async def test_handler_cancellation_declines() -> None:
    async def on_request(env: RequestEnvelope) -> None:
        pass

    forwarder = ElicitationForwarder(on_request=on_request, default_timeout=60.0)

    async def run_handler() -> ElicitResult[Any]:
        return await forwarder.handler("will cancel", None, None, None)

    task = asyncio.create_task(run_handler())
    # Give the handler a tick to register its pending Future.
    for _ in range(10):
        await asyncio.sleep(0)
        if forwarder.pending:
            break
    assert forwarder.pending, "handler did not register a pending Future"

    task.cancel()
    # Handler catches CancelledError internally and returns decline.
    result = await task
    assert result.action == "decline"


async def test_resolve_unknown_id_is_noop() -> None:
    forwarder = ElicitationForwarder()
    # Should not raise.
    await forwarder.resolve("nonexistent", {"action": "accept"})


async def test_cancel_all_drains_pending() -> None:
    async def on_request(env: RequestEnvelope) -> None:
        pass

    forwarder = ElicitationForwarder(on_request=on_request, default_timeout=60.0)

    async def run_handler(label: str) -> ElicitResult[Any]:
        return await forwarder.handler(label, None, None, None)

    t1 = asyncio.create_task(run_handler("one"))
    t2 = asyncio.create_task(run_handler("two"))

    # Wait until both handlers registered their pending Futures.
    for _ in range(20):
        await asyncio.sleep(0)
        if len(forwarder.pending) == 2:
            break
    assert len(forwarder.pending) == 2

    await forwarder.cancel_all(action="decline")

    r1, r2 = await asyncio.gather(t1, t2)
    assert r1.action == "decline"
    assert r2.action == "decline"


async def test_on_request_exception_declines() -> None:
    async def on_request(env: RequestEnvelope) -> None:
        raise RuntimeError("transport closed")

    forwarder = ElicitationForwarder(on_request=on_request)
    result = await forwarder.handler("bad", None, None, None)
    assert result.action == "decline"
    assert forwarder.pending == {}


def test_invalid_default_timeout() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ElicitationForwarder(default_timeout=0)


class _FakeParams:
    """Minimal stand-in for FastMCP's ``ElicitRequestParams``."""

    def __init__(self) -> None:
        # FastMCP's real param object uses this camelCase attr name — we
        # mirror it verbatim so the forwarder's ``getattr(params,
        # "requestedSchema", None)`` probe resolves.
        self.requestedSchema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        }
