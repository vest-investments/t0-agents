"""Unit tests for the :func:`require_parsed` built-in guardrail.

``require_parsed`` is the opt-in counterpart to the permissive-parsing
layer: when a JSON run completes with ``parsed is None`` despite every
recovery attempt, the guardrail raises :class:`GuardrailBlockedError`
so the run aborts with a clear policy violation instead of returning
an :class:`AgentResult` whose ``parsed`` field is silently unusable.

Coverage:

- The happy path (valid JSON) passes through.
- An unparseable JSON run with permissive parsing disabled and the
  retry budget exhausted aborts.
- A chat agent with ``require_parsed`` configured is a strict no-op —
  ``parsed`` is structurally absent on the chat path.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.outputs import JsonOutput
from t0_agents.core.types import ToolProvider
from t0_agents.guardrails import require_parsed
from t0_agents.testing import MockGateway
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


def _json_agent(
    *,
    t0: MockGateway,
    output: JsonOutput | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {
        "name": "extractor",
        "function_name": "extract_customer_info",
        "output": output if output is not None else JsonOutput(json_mode="strict"),
    }
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


async def test_require_parsed_passes_when_parsed() -> None:
    """Valid JSON → guardrail does not abort the run."""
    t0 = MockGateway.script_json(
        [{"raw": '{"a": 1}', "parsed": {"a": 1}}]
    )
    agent = _json_agent(t0=t0, guardrails=[require_parsed()])

    result = await agent.invoke("go")

    assert result.parsed == {"a": 1}
    assert result.parse_strategy == "strict"
    assert any(
        entry.guardrail_name == "require_parsed"
        and entry.hook == "after_run"
        and entry.kind == "pass"
        for entry in result.guardrail_log
    )


async def test_require_parsed_aborts_when_none() -> None:
    """Unrecoverable JSON + no retries → guardrail raises GuardrailBlockedError."""
    t0 = MockGateway.script_json(
        [{"raw": "totally not json", "parsed": None}]
    )
    agent = _json_agent(
        t0=t0,
        output=JsonOutput(
            json_mode="strict", permissive=False, max_retries=0
        ),
        guardrails=[require_parsed()],
    )

    with pytest.raises(GuardrailBlockedError, match="parsed=None"):
        await agent.invoke("go")


async def test_require_parsed_noop_on_chat_run() -> None:
    """Chat agent (parsed is structurally None) → guardrail is a no-op."""
    t0 = FakeT0([FakeChatResponse(content=[Text(text="hi")])])
    client = FakeMCPClient(tools=[])
    agent = Agent(
        AgentConfig(
            name="chat",
            function_name="agent",
            guardrails=[require_parsed()],
        ),
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, client),
    )

    result = await agent.invoke("go")

    assert result.function_kind == "chat"
    assert result.parsed is None
    assert result.response == "hi"
