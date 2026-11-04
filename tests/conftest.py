from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from sse_starlette.sse import AppStatus

from tests.server import mcp


@pytest.fixture(autouse=True)
def _reset_sse_shutdown_latch() -> None:
    """Clear sse-starlette's process-wide shutdown flag before every test.

    Its watcher sets ``AppStatus.should_exit`` once it sees a uvicorn server
    exit, and nothing resets it; from then on every SSE response closes at
    once.  A test whose ephemeral server takes over 0.5 s to stop would
    otherwise drop the next test's ``initialize`` response and hang it.
    """
    AppStatus.should_exit = False


@pytest.fixture
async def mcp_client() -> Client[Any]:
    """FastMCP client connected to the shared test server."""
    return Client(mcp)


# --- Elicitation handler fixtures -------------------------------------------


@pytest.fixture
def auto_approve_handler() -> Any:
    """Handler that accepts all elicitation requests with empty content."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="accept", content={})

    return handler


@pytest.fixture
def auto_decline_handler() -> Any:
    """Handler that declines all elicitation requests."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    return handler


@pytest.fixture
async def elicitation_client(auto_approve_handler: Any) -> Client[Any]:
    """Client with auto-approve elicitation handler."""
    return Client(mcp, elicitation_handler=auto_approve_handler)
