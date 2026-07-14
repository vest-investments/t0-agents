"""``create_app_from_glow`` — the glow App → serve bridge (M7).

The glow App runs entirely in-process here (FakeT0 gateway, in-memory
FastMCP client from ``@app.tool``) — composition only, no network.
"""

from __future__ import annotations

from typing import cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.glow import App
from t0_agents.serve import create_app_from_glow
from tests._fake_t0 import FakeT0


def _glow_app_with_tool() -> App:
    app = App(t0=cast(AsyncTensorZeroGateway, FakeT0(responses=[])))

    @app.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    @app.agent("tutor")
    async def tutor(query: str) -> str:  # type: ignore[empty-body]
        """Tutor."""

    return app


async def test_bridge_builds_and_wraps_the_sole_agent() -> None:
    glow_app = _glow_app_with_tool()
    async with glow_app:
        api = await create_app_from_glow(glow_app)
        assert api.title == "t0-agents · tutor"
        # share_client mode rides the App-owned, connected client.
        assert api.state.session_factory.base_config.name == "tutor"


async def test_bridge_requires_agent_name_when_ambiguous() -> None:
    glow_app = _glow_app_with_tool()

    @glow_app.agent("second")
    async def second(query: str) -> str:  # type: ignore[empty-body]
        """Second."""

    async with glow_app:
        with pytest.raises(ConfigurationError, match="agent="):
            await create_app_from_glow(glow_app)
        api = await create_app_from_glow(glow_app, agent="second")
        assert api.title == "t0-agents · second"


async def test_bridge_rejects_unknown_agent_and_empty_app() -> None:
    empty = App(t0=cast(AsyncTensorZeroGateway, FakeT0(responses=[])))
    with pytest.raises(ConfigurationError, match="no registered agents"):
        await create_app_from_glow(empty)

    glow_app = _glow_app_with_tool()
    async with glow_app:
        with pytest.raises(ConfigurationError, match="not registered"):
            await create_app_from_glow(glow_app, agent="ghost")
