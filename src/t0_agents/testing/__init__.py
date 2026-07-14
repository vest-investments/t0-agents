"""Test-authoring surface for t0-agents.

Three complementary tools for writing agent tests:

* :class:`MockGateway` + :class:`MockToolProvider` — deterministic, no
  infrastructure, ideal for unit tests pinning down *exactly* what the
  agent did.  Pair with :mod:`t0_agents.testing.scenarios` for canned
  loop shapes.  Chat and JSON paths share the same gateway:
  :meth:`MockGateway.script` / :meth:`MockGateway.canned` script chat
  responses; :meth:`MockGateway.script_json` /
  :meth:`MockGateway.canned_json` script JSON responses for agents whose
  ``function_kind == "json"``.
* :class:`ScriptedAgent` — a whole-:class:`~t0_agents.core.agent.Agent`
  fake whose ``invoke()`` replays a fixed event script + result with no
  gateway at all.  Pair with ``App.agent_overrides`` to swap a glow agent
  by name.  Use it when the unit under test consumes ``invoke() -> Run``
  and you want to script the events, not the gateway responses.
* :class:`EmbeddedGatewayHelper` — real TensorZero inference against any
  provider whose API key is in the environment, no Docker / ClickHouse /
  Postgres required.  Ideal for integration tests that need realistic
  model behaviour but shouldn't spin up containers.

Public surface::

    from t0_agents.testing import (
        MockGateway, MockChatResponse, MockJsonResponse,
        MockToolProvider, RecordingGateway, ScriptedAgent,
        MockStreamingGateway, fake_event_stream,
        EmbeddedGatewayHelper,
        assert_tool_called, assert_tool_not_called, assert_tool_args_equal,
        assert_max_turns, assert_response_contains,
        assert_parsed_equals, assert_parsed_matches_schema,
        assert_invalid_json, assert_parse_strategy, assert_retry_attempts,
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
    assert_invalid_json,
    assert_max_turns,
    assert_parse_strategy,
    assert_parsed_equals,
    assert_parsed_matches_schema,
    assert_response_contains,
    assert_retry_attempts,
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
    MockJsonResponse,
    MockToolProvider,
    RecordingGateway,
    ToolHandler,
)
from t0_agents.testing.scripted import ScriptedAgent
from t0_agents.testing.streaming_mocks import (
    MockStreamingGateway,
    fake_event_stream,
)

__all__ = [
    "AgentFactory",
    "EmbeddedGatewayHelper",
    "MockChatResponse",
    "MockGateway",
    "MockJsonResponse",
    "MockStreamingGateway",
    "MockToolProvider",
    "RecordingGateway",
    "ScriptedAgent",
    "ToolHandler",
    "agent_test_factory",
    "assert_invalid_json",
    "assert_max_turns",
    "assert_parse_strategy",
    "assert_parsed_equals",
    "assert_parsed_matches_schema",
    "assert_response_contains",
    "assert_retry_attempts",
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
