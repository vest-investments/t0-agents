"""``agent.invoke(...).events()`` rejects non-streaming output modes.

v0.12's ``Agent.stream`` raised ``ConfigurationError`` on JSON-mode
agents because TensorZero does not emit ``output.parsed`` mid-stream.
v1.0 keys the same rejection off ``output.supports_streaming``:
:class:`JsonOutput` and :class:`StructuredOutput` ship with
``supports_streaming = False`` and the gate fires at first
``events()`` call.
"""

from __future__ import annotations

from typing import cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import JsonOutput, StructuredOutput
from t0_agents.core.types import ToolProvider
from t0_agents.testing import MockGateway
from tests._fake_mcp import FakeMCPClient


async def test_events_rejects_jsonoutput() -> None:
    t0 = MockGateway.script_json([{"raw": '{"a": 1}', "parsed": {"a": 1}}])
    agent = Agent(
        AgentConfig(
            name="x",
            function_name="extract",
            output=JsonOutput(json_mode="strict"),
        ),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )
    run = agent.invoke("go")
    with pytest.raises(ConfigurationError, match="streaming"):
        async for _ in run.events():
            pass


async def test_events_rejects_structuredoutput() -> None:
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    t0 = MockGateway.script_json([{"raw": '{"a": 1}', "parsed": {"a": 1}}])
    agent = Agent(
        AgentConfig(
            name="x",
            function_name="extract",
            output=StructuredOutput(M),
        ),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )
    run = agent.invoke("go")
    with pytest.raises(ConfigurationError, match="streaming"):
        async for _ in run.events():
            pass
