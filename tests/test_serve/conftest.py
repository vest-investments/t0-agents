"""Shared fixtures for ``tests/test_serve/``.

Hand-written stubs rather than ``unittest.mock`` so the suite stays
mypy-strict-clean (matches ``tests/_fake_t0.py`` and ``tests/_fake_mcp.py``
conventions).  ``FakeFastMCPClient`` is a superset of the MCP fake that
additionally satisfies the ``async with`` contract and exposes the
``is_connected()`` probe used by ``SessionFactory.probe_tool_provider``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from tests._fake_t0 import FakeChatResponse, FakeT0


class FakeFastMCPClient:
    """``fastmcp.Client`` stand-in — ToolProvider + async CM + is_connected."""

    def __init__(
        self,
        *,
        tools: list[MCPTool] | None = None,
        call_results: dict[str, CallToolResult | Exception] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else []
        self._call_results = call_results or {}
        self._connected = False
        self.enter_count = 0
        self.exit_count = 0
        self.id = uuid4()
        self.call_tool_invocations: list[
            tuple[str | None, dict[str, Any] | None, float | None]
        ] = []

    async def __aenter__(self) -> FakeFastMCPClient:
        self._connected = True
        self.enter_count += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._connected = False
        self.exit_count += 1

    def is_connected(self) -> bool:
        return self._connected

    async def list_tools(self) -> list[MCPTool]:
        return self._tools

    async def call_tool(
        self,
        name: str | None,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> CallToolResult:
        self.call_tool_invocations.append((name, arguments, timeout))
        key = name or ""
        if key not in self._call_results:
            raise AssertionError(f"FakeFastMCPClient: no canned result for {name!r}")
        outcome = self._call_results[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _text_response(text: str = "hi") -> FakeChatResponse:
    """Script a one-turn text response (no tool calls)."""
    from tensorzero.types import Text  # local import; keeps import surface tight

    return FakeChatResponse(content=[Text(type="text", text=text)])


@pytest.fixture
def base_config() -> AgentConfig:
    return AgentConfig(name="serve-test", function_name="agent")


@pytest.fixture
def fake_t0() -> FakeT0:
    return FakeT0(responses=[_text_response("hi")])


@pytest.fixture
def fake_tool_provider() -> FakeFastMCPClient:
    return FakeFastMCPClient()


@pytest.fixture
def base_agent(
    base_config: AgentConfig,
    fake_t0: FakeT0,
    fake_tool_provider: FakeFastMCPClient,
) -> Agent:
    return Agent(
        base_config,
        t0=fake_t0,  # type: ignore[arg-type]
        tool_provider=fake_tool_provider,
    )


@pytest.fixture
def client_factory_log() -> dict[str, list[FakeFastMCPClient]]:
    return {"created": []}


@pytest.fixture
def fake_client_factory(
    client_factory_log: dict[str, list[FakeFastMCPClient]],
) -> Iterator[Callable[[], FakeFastMCPClient]]:
    """Factory that mints a fresh FakeFastMCPClient per call."""

    def _factory() -> FakeFastMCPClient:
        client = FakeFastMCPClient()
        client_factory_log["created"].append(client)
        return client

    yield _factory
