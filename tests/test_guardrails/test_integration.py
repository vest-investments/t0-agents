"""Integration tests for guardrails — require T0 Gateway running.

Marked ``@pytest.mark.integration`` so ``uv run pytest -m "not integration"``
continues to pass without Docker.  Exercise the two flows that most benefit
from a real inference pass:

- ``require_approval`` with an auto-decline handler proves the LLM sees the
  rejection text as a ``ToolResult`` and continues gracefully.
- ``max_tool_calls(per_run=0)`` proves that ``GuardrailBlockedError``
  propagates past the live loop's exception handling.  We don't use
  ``max_cost`` here because :class:`tensorzero.Usage.cost` depends on
  provider pricing being declared in ``tensorzero.toml``; Groq-hosted
  ``gpt_oss_20`` returns ``cost=None``, which the cost tracker treats as
  ``0.0`` — the budget never gets crossed and the abort never fires.  See
  ``docs/guardrails.md`` → "Cost reporting depends on TOML pricing" for
  the caveat, and ``tests/test_guardrails/test_max_cost.py`` for the full
  accumulation/abort unit tests that run offline with a seeded
  :class:`FakeUsage`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import ToolProvider
from t0_agents.guardrails import max_tool_calls, require_approval
from tests.server import mcp

pytestmark = pytest.mark.integration


def _gateway_factory() -> Awaitable[AsyncTensorZeroGateway]:
    return cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )


async def test_require_approval_auto_decline_surfaces_to_llm() -> None:
    """Auto-decline handler → the LLM sees a 'declined' tool result."""

    async def _decline(
        message: str, response_type: Any, params: Any, context: Any
    ) -> ElicitResult[Any]:
        return ElicitResult(action="decline")

    client: Client[Any] = Client(mcp)
    config = AgentConfig(
        name="approval-it",
        function_name="agent",
        dryrun=True,
        tool_timeout=None,
        max_turns=5,
        guardrails=[
            require_approval(tools=["add", "greet", "search"], handler=_decline),
        ],
    )

    async with await _gateway_factory() as t0, client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, client),
        )
        result = await agent.run("Add 2 and 3 using the add tool.")

    assert result.response
    # Some rejection entry must appear in the log for the auto-declined call.
    assert any(e.kind == "rejection" for e in result.guardrail_log)


async def test_max_tool_calls_zero_aborts_through_live_loop() -> None:
    """``max_tool_calls(per_run=0)`` raises ``GuardrailBlockedError``.

    Chosen over ``max_cost`` for the fatal-propagation path because the
    cap is deterministic regardless of provider cost reporting.  Proves
    the end-to-end propagation through ``_execute_single_tool_call``'s
    new ``T0AgentsFatalError`` branch and out of ``Agent.run()``.
    """
    client: Client[Any] = Client(mcp)
    config = AgentConfig(
        name="cap-it",
        function_name="agent",
        dryrun=True,
        max_turns=4,
        guardrails=[max_tool_calls(per_run=0)],
    )

    async with await _gateway_factory() as t0, client:
        agent = Agent(
            config,
            t0=t0,
            tool_provider=cast(ToolProvider, client),
        )
        with pytest.raises(GuardrailBlockedError, match="Tool-call cap"):
            await agent.run("Use the add tool to compute 2 + 3.")
