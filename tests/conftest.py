from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client

from tests.server import mcp


@pytest.fixture
async def mcp_client() -> Client[Any]:
    """FastMCP client connected to the shared test server."""
    return Client(mcp)
