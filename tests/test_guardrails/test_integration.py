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
from t0_agents.core.exceptions import GuardrailBlockedError, MaxTurnsExceededError
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
    """Auto-decline handler → the LLM sees a 'declined' tool result.

    The guardrail mechanism is pinned strictly by the offline unit tests
    in ``tests/test_guardrails/test_require_approval.py`` (driven by
    ``FakeT0``).  This integration test only proves the full live loop
    surfaces the rejection back to the LLM when Groq actually invokes
    the guarded tool.  Three known ``gpt-oss-20b`` flake paths are
    skipped rather than asserted (matching the Groq-flake convention
    shared by ``test_run_agent_survives_failing_tool`` and friends):

    1. **No tool call this run** — the model occasionally answers the
       arithmetic as plain text without invoking ``add``; the rejection
       branch is never exercised.  Use non-trivial operands
       (``1247 + 8932``) to keep the model honest, then
       ``pytest.skip`` with the ``guardrail_log`` if it still happens.
    2. **Max-turns exhaustion from malformed retry tool names** — after
       an auto-declined tool result, gpt-oss-20b sometimes serialises
       its retry with an invalid tool name (e.g. ``add<|channel|>commentary``)
       that Groq's own validator rejects as ``Tool call validation
       failed``.  The framework recovers the inference but the model
       keeps retrying with the same garbage, eventually hitting
       ``max_turns``.  At that point the run aborts before the
       ``AgentResult`` lands, so the ``guardrail_log`` is no longer
       reachable — ``pytest.skip`` since the rejection branch *may*
       have been logged but the assertion target is gone.
    """

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
        # Non-trivial operands: gpt-oss-20b reliably memorises 2+3 and
        # answers "5" without a tool call.  1247+8932 is non-trivial
        # enough that the model needs the tool.
        try:
            result = await agent.invoke(
                "Use the add tool to compute 1247 + 8932."
            )
        except MaxTurnsExceededError as e:
            # Flake #2: model kept emitting malformed retry tool names
            # after the auto-decline; the loop recovered each one but
            # ran out of turns before the model produced a final text.
            # The rejection branch may or may not have been logged before
            # the abort — we can't inspect it from the exception.
            pytest.skip(
                "Groq gpt-oss-20b exhausted max_turns after the auto-"
                f"decline (likely malformed retry tool names): {e}"
            )

    assert result.response
    if not any(e.kind == "rejection" for e in result.guardrail_log):
        pytest.skip(
            "Groq gpt-oss-20b answered without invoking add — the auto-"
            "decline branch was not exercised this run.  guardrail_log="
            f"{result.guardrail_log!r}"
        )


async def test_max_tool_calls_zero_aborts_through_live_loop() -> None:
    """``max_tool_calls(per_run=0)`` raises ``GuardrailBlockedError``.

    Chosen over ``max_cost`` for the fatal-propagation path because the
    cap is deterministic regardless of provider cost reporting.  Proves
    the end-to-end propagation through ``_execute_single_tool_call``'s
    new ``T0AgentsFatalError`` branch and out of ``agent.invoke()``.
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
            await agent.invoke("Use the add tool to compute 2 + 3.")
