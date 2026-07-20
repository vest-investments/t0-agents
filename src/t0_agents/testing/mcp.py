"""In-memory helpers for testing glow agents that mount MCP servers.

Driving a mounted MCP tool through the real fastmcp client boundary — the
integration test the guidelines call for, since an in-process
``registered_tools[name](**kwargs)`` call bypasses FastMCP's schema and
serialization layer and misses framework-level bugs — costs a few coupled
facts today: a stand-in gateway (no inference happens in a tool-surface test),
the App lifespan, and the private App-owned runtime.  :func:`glow_tools` hides
them behind one async context manager that yields the mounted tools as a
:class:`~t0_agents.core.types.ToolProvider`.  :class:`ScriptedElicitation`
adds a scripted human-in-the-loop handler for tools that call ``ctx.elicit``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.elicitation_forwarder import ElicitationForwarder, RequestEnvelope
from t0_agents.glow import App, MCPServer

if TYPE_CHECKING:
    from fastmcp.client.elicitation import ElicitationHandler

    from t0_agents.core.types import ToolProvider

__all__ = ["ScriptedElicitation", "glow_tools"]


@asynccontextmanager
async def glow_tools(
    *servers: MCPServer | type[MCPServer],
    elicitation_handler: ElicitationHandler | None = None,
) -> AsyncIterator[ToolProvider]:
    """Mount ``servers`` on an in-memory glow App and yield their tools.

    The yielded :class:`~t0_agents.core.types.ToolProvider` is the App-owned
    MCP runtime: ``await tools.list_tools()`` returns the merged, namespace-
    prefixed inventory and ``await tools.call_tool(name, arguments)`` drives a
    real in-memory ``fastmcp.Client`` → server → schema → result round-trip.
    In-memory ``server=`` mounts connect with no sockets or transport; nothing
    is sent to TensorZero, so the gateway is a stand-in.

    Pass ``elicitation_handler=`` (for example
    :attr:`ScriptedElicitation.handler`) to test tools that call
    ``ctx.elicit(...)``.

    :param servers: ``MCPServer`` specs (instances or subclasses) to mount.
    :raises ValueError: if no server mounts any tool.
    """
    app = App(
        t0=cast(AsyncTensorZeroGateway, object()),
        servers=list(servers),
        elicitation_handler=elicitation_handler,
    )
    async with app.lifespan():
        provider = app._owned_client
        if provider is None:
            raise ValueError("glow_tools requires at least one MCP server")
        yield provider


class ScriptedElicitation:
    """Scripted human-in-the-loop handler for in-memory elicit round-trips.

    Wraps an
    :class:`~t0_agents.core.elicitation_forwarder.ElicitationForwarder` whose
    ``on_request`` resolves each incoming elicitation from a fixed script,
    inline — the forwarder settles the future before awaiting it, so the tool's
    ``ctx.elicit(...)`` returns without the queue / ``create_task`` /
    ``resolve`` dance an out-of-process resolver needs.  Pass :attr:`handler`
    to ``App(elicitation_handler=...)`` or ``glow_tools(elicitation_handler=...)``.

    Each response is an action string (``"accept" | "decline" | "cancel"``) or
    a payload dict (``{"action": ..., "content": {...}}``) — the shape
    :meth:`~t0_agents.core.elicitation_forwarder.ElicitationForwarder.resolve`
    accepts.  A scalar accept carries the value under ``content``
    (``{"action": "accept", "content": {"value": True}}``).  Responses are
    consumed in order; an elicitation beyond the script is declined.  Every
    request is recorded in :attr:`received` for assertions on the ``message`` /
    ``schema`` / ``response_type`` the server sent.
    """

    def __init__(self, responses: Sequence[str | dict[str, Any]]) -> None:
        self.received: list[RequestEnvelope] = []
        self._responses: deque[str | dict[str, Any]] = deque(responses)
        self._forwarder = ElicitationForwarder(on_request=self._on_request)

    @property
    def handler(self) -> ElicitationHandler:
        """The FastMCP elicitation handler to bind on the App or client."""
        return self._forwarder.handler

    async def _on_request(self, envelope: RequestEnvelope) -> None:
        self.received.append(envelope)
        response = self._responses.popleft() if self._responses else "decline"
        payload: dict[str, Any] = (
            {"action": response} if isinstance(response, str) else response
        )
        await self._forwarder.resolve(envelope["elicitation_id"], payload)
