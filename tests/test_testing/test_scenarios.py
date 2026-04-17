"""End-to-end tests: scenarios drive a real ``Agent`` with mocks attached."""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import ToolProvider
from t0_agents.testing import (
    MockToolProvider,
    assert_max_turns,
    assert_tool_args_equal,
    assert_tool_called,
    scenarios,
)


def _agent(gateway: Any, tools: MockToolProvider, *, max_turns: int = 5) -> Agent:
    return Agent(
        AgentConfig(name="t", function_name="assistant", max_turns=max_turns),
        t0=cast(AsyncTensorZeroGateway, gateway),
        tool_provider=cast(ToolProvider, tools),
    )


class TestSingleToolHappy:
    async def test_drives_agent_to_final_text(self) -> None:
        gateway = scenarios.single_tool_happy("lookup", final_text="The answer is 42")
        tools = MockToolProvider({"lookup": lambda args: "42"})

        result = await _agent(gateway, tools).run("what is it?")

        assert_tool_called("lookup", tool_provider=tools)
        assert "42" in result.response
        assert result.turns == 2
        assert_max_turns(result, at_most=2)

    async def test_tool_receives_empty_arguments(self) -> None:
        gateway = scenarios.single_tool_happy("ping")
        tools = MockToolProvider({"ping": lambda args: "pong"})

        await _agent(gateway, tools).run("go")

        assert_tool_args_equal("ping", {}, tool_provider=tools)


class TestToolErrorRecovery:
    async def test_loop_recovers_from_raising_handler(self) -> None:
        gateway = scenarios.tool_error_recovery(
            "flaky", final_text="I couldn't run it — answering anyway."
        )

        def handler(_args: dict[str, Any]) -> str:
            raise RuntimeError("provider is offline")

        tools = MockToolProvider({"flaky": handler})

        result = await _agent(gateway, tools).run("try it")

        assert_tool_called("flaky", tool_provider=tools)
        assert "answering" in result.response.lower()


class TestMaxTurnsExhausted:
    async def test_raises_after_max_turns(self) -> None:
        gateway = scenarios.max_turns_exhausted("loop")
        tools = MockToolProvider({"loop": lambda args: "still running"})

        with pytest.raises(MaxTurnsExceededError):
            await _agent(gateway, tools, max_turns=3).run("forever")
