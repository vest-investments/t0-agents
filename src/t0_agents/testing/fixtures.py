"""Pytest fixtures for agent tests.

Four fixtures, every one importable from :mod:`t0_agents.testing`:

* :func:`mock_t0` — a fresh :class:`MockGateway` wired with an empty
  script.  Override the script in-test by reassigning the gateway.
* :func:`mock_tool_provider` — a fresh :class:`MockToolProvider` with
  no tools; tests register tools inline as needed.
* :func:`embedded_t0` — an async :class:`AsyncTensorZeroGateway` built
  via :class:`EmbeddedGatewayHelper`.  Parametrizable via
  ``@pytest.mark.parametrize("embedded_t0", ["path/to/config.toml"],
  indirect=True)``.
* :func:`agent_test_factory` — zero-config factory that builds an
  :class:`Agent` against a fresh gateway + tool provider pair.  Reduces
  test boilerplate when the scenario doesn't care about pre-populating
  either mock.

Usage::

    from t0_agents.testing import mock_t0, mock_tool_provider  # fixtures
    from t0_agents.testing import scenarios, assert_tool_called

    async def test_lookup(mock_tool_provider):
        mock_tool_provider._tools["lookup"] = lambda args: "42"
        gateway = scenarios.single_tool_happy("lookup", final_text="done")
        agent = Agent(
            AgentConfig(name="t", function_name="agent"),
            t0=gateway,
            tool_provider=mock_tool_provider,
        )
        result = await agent.run("hi")
        assert_tool_called("lookup", tool_provider=mock_tool_provider)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider
from t0_agents.testing.embedded import EmbeddedGatewayHelper
from t0_agents.testing.mocks import MockGateway, MockToolProvider


@pytest.fixture
def mock_t0() -> MockGateway:
    """A fresh :class:`MockGateway` with an empty script.

    Tests typically reassign the fixture's value to
    :meth:`MockGateway.script` / :meth:`MockGateway.canned` output
    before passing it to :class:`Agent`; alternatively use the
    :mod:`t0_agents.testing.scenarios` factories directly.
    """
    return MockGateway.script([])


@pytest.fixture
def mock_tool_provider() -> MockToolProvider:
    """A fresh :class:`MockToolProvider` with no registered tools."""
    return MockToolProvider({})


@pytest.fixture
async def embedded_t0(request: pytest.FixtureRequest) -> AsyncIterator[AsyncTensorZeroGateway]:
    """An embedded gateway for integration-style tests without Docker.

    Marked as an async fixture; pytest-asyncio in auto mode picks it up
    without extra decoration.  Defaults to ``tests/configs/minimal.toml``
    — override per-test via::

        @pytest.mark.parametrize("embedded_t0", ["my/config.toml"], indirect=True)
        async def test_thing(embedded_t0): ...

    The fixture yields the real :class:`AsyncTensorZeroGateway`; its
    lifecycle (build + teardown) is owned by
    :class:`EmbeddedGatewayHelper`.
    """
    config_file = getattr(request, "param", "tests/configs/minimal.toml")
    async with EmbeddedGatewayHelper.from_config(config_file) as gw:
        yield gw


AgentFactory = Callable[..., Agent]


@pytest.fixture
def agent_test_factory(
    mock_t0: MockGateway,
    mock_tool_provider: MockToolProvider,
) -> AgentFactory:
    """Return a zero-config factory building an :class:`Agent`.

    Every positional / keyword argument is forwarded to
    :class:`AgentConfig`; ``t0`` and ``tool_provider`` default to the
    sibling fixtures but can be overridden by passing them explicitly.
    Defaults: ``name="test"``, ``function_name="test-function"``,
    ``max_turns=5``.
    """

    def factory(
        *,
        t0: Any = None,
        tool_provider: Any = None,
        **config_overrides: Any,
    ) -> Agent:
        cfg_kwargs: dict[str, Any] = {
            "name": "test",
            "function_name": "test-function",
            "max_turns": 5,
        }
        cfg_kwargs.update(config_overrides)
        config = AgentConfig(**cfg_kwargs)
        return Agent(
            config,
            # MockGateway duck-types AsyncTensorZeroGateway — the narrow
            # type is enforced structurally by the agent loop, not
            # nominally.  Cast so mypy strict accepts the fixture.
            t0=cast(
                AsyncTensorZeroGateway,
                t0 if t0 is not None else mock_t0,
            ),
            tool_provider=cast(
                ToolProvider,
                tool_provider
                if tool_provider is not None
                else mock_tool_provider,
            ),
        )

    return factory


__all__ = [
    "AgentFactory",
    "agent_test_factory",
    "embedded_t0",
    "mock_t0",
    "mock_tool_provider",
]
