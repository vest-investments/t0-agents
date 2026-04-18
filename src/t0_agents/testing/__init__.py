"""Test-authoring surface for t0-agents.

Two complementary tools for writing agent tests:

* :class:`MockGateway` + :class:`MockToolProvider` — deterministic, no
  infrastructure, ideal for unit tests pinning down *exactly* what the
  agent did.  Pair with :mod:`t0_agents.testing.scenarios` for canned
  loop shapes.
* :class:`EmbeddedGatewayHelper` — real TensorZero inference against any
  provider whose API key is in the environment, no Docker / ClickHouse /
  Postgres required.  Ideal for integration tests that need realistic
  model behaviour but shouldn't spin up containers.

Public surface::

    from t0_agents.testing import (
        MockGateway, MockChatResponse, MockToolProvider, RecordingGateway,
        MockStreamingGateway, fake_event_stream,
        EmbeddedGatewayHelper,
        assert_tool_called, assert_tool_not_called, assert_tool_args_equal,
        assert_max_turns, assert_response_contains,
        scenarios, streaming_mocks,
        mock_t0, mock_tool_provider, embedded_t0, agent_test_factory,
    )

**Test-only submodule.**  Every symbol here is meant for ``tests/`` and
``examples/*/test_*`` files.  Never import them from a production code
path — the mocks bypass real inference and tool dispatch by design.

See :doc:`/testing` in the public docs for a full walkthrough.
"""

from t0_agents.testing import scenarios, streaming_mocks
from t0_agents.testing.assertions import (
    assert_max_turns,
    assert_response_contains,
    assert_tool_args_equal,
    assert_tool_called,
    assert_tool_not_called,
)
from t0_agents.testing.embedded import EmbeddedGatewayHelper
from t0_agents.testing.fixtures import (
    AgentFactory,
    agent_test_factory,
    embedded_t0,
    mock_t0,
    mock_tool_provider,
)
from t0_agents.testing.mocks import (
    MockChatResponse,
    MockGateway,
    MockToolProvider,
    RecordingGateway,
    ToolHandler,
)
from t0_agents.testing.streaming_mocks import (
    MockStreamingGateway,
    fake_event_stream,
)

__all__ = [
    "AgentFactory",
    "EmbeddedGatewayHelper",
    "MockChatResponse",
    "MockGateway",
    "MockStreamingGateway",
    "MockToolProvider",
    "RecordingGateway",
    "ToolHandler",
    "agent_test_factory",
    "assert_max_turns",
    "assert_response_contains",
    "assert_tool_args_equal",
    "assert_tool_called",
    "assert_tool_not_called",
    "embedded_t0",
    "fake_event_stream",
    "mock_t0",
    "mock_tool_provider",
    "scenarios",
    "streaming_mocks",
]
