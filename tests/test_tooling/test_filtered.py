"""Unit tests for :class:`t0_agents.tooling.FilteredToolProvider`.

Gateway-free: a :class:`~t0_agents.testing.MockToolProvider` stands in for the
wrapped provider so both the listing filter and the ``call_tool`` boundary are
exercised without a live FastMCP transport.
"""

from __future__ import annotations

import pytest

from t0_agents.core.exceptions import T0AgentsError
from t0_agents.testing import MockToolProvider
from t0_agents.tooling import FilteredToolProvider


def _provider() -> MockToolProvider:
    return MockToolProvider(
        {
            "get_weather": lambda _a: "sunny",
            "get_quote": lambda _a: "100",
            "place_order": lambda _a: "filled",
        }
    )


async def test_list_tools_filters_to_allow() -> None:
    filtered = FilteredToolProvider(_provider(), {"get_weather", "get_quote"})
    names = {tool.name for tool in await filtered.list_tools()}
    assert names == {"get_weather", "get_quote"}


async def test_empty_allow_is_toolless() -> None:
    filtered = FilteredToolProvider(_provider(), set())
    assert await filtered.list_tools() == []


async def test_call_tool_allowed_delegates_to_inner() -> None:
    inner = _provider()
    filtered = FilteredToolProvider(inner, {"get_weather"})
    await filtered.call_tool("get_weather", {})
    assert inner.calls == [{"name": "get_weather", "arguments": {}}]


async def test_call_tool_rejects_filtered_out_name() -> None:
    inner = _provider()
    filtered = FilteredToolProvider(inner, {"get_weather"})
    with pytest.raises(T0AgentsError, match="place_order"):
        await filtered.call_tool("place_order", {})
    assert inner.calls == []


async def test_allow_accepts_any_iterable() -> None:
    # A generator (single-use iterable) must be materialised, not exhausted.
    filtered = FilteredToolProvider(
        _provider(), (n for n in ["get_weather"])
    )
    first = {tool.name for tool in await filtered.list_tools()}
    second = {tool.name for tool in await filtered.list_tools()}
    assert first == second == {"get_weather"}


def test_server_for_delegates_to_attributing_inner() -> None:
    class _Attributing(MockToolProvider):
        def server_for(self, name: str) -> str | None:
            return name.split("_", 1)[0] if "_" in name else None

    filtered = FilteredToolProvider(
        _Attributing({"svc_x": lambda _a: "ok"}), {"svc_x"}
    )
    assert filtered.server_for("svc_x") == "svc"


def test_server_for_none_when_inner_cannot_attribute() -> None:
    filtered = FilteredToolProvider(_provider(), {"get_weather"})
    assert filtered.server_for("get_weather") is None
