from __future__ import annotations

import pytest
from fastmcp import Client

from tests.server import mcp


@pytest.fixture
async def mcp_client() -> Client:
    """FastMCP client connected to the shared test server."""
    return Client(mcp)
