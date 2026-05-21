"""Order invariant: ``GuardrailsMiddleware`` wraps ``SubAgentToolProvider``.

Contract §3.2 invariant: orchestrator-level guardrails must observe
sub-agent delegations as tool calls.  Post-refactor the wiring is:

- :class:`Agent` auto-allocates a :class:`SubAgentToolProvider` at
  construction time and stores it on ``self._tool_provider``.
- The middleware chain (``Memory → SubAgents → Guardrails → ...``) runs
  with ``ctx.tool_provider`` initialised from ``self._tool_provider``.
- :class:`GuardrailsMiddleware` is the only middleware that *still*
  wraps the provider — :class:`SubAgentMiddleware` is now ContextVar-only
  and never touches ``ctx.tool_provider``.

Concretely: at terminal time ``ctx.tool_provider`` is a
:class:`GuardedToolProvider` wrapping the auto-allocated
:class:`SubAgentToolProvider`.  Without guardrails it is the bare
:class:`SubAgentToolProvider`.
"""

from __future__ import annotations

from typing import Any, cast

from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.middleware.base import Middleware, RunContext
from t0_agents.core.types import AgentResult, Guardrail, ToolProvider
from t0_agents.delegation.provider import SubAgentToolProvider
from t0_agents.guardrails.runtime import GuardedToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0


class _NoopGuardrail:
    name = "noop"

    async def before_run(self, ctx: Any) -> None: ...
    async def before_tool(self, ctx: Any, tc: Any) -> str | None:
        return None
    async def after_tool(self, ctx: Any, tc: Any, r: str) -> None: ...
    async def after_run(self, ctx: Any, result: Any) -> None: ...


class _StackInspector:
    """Innermost middleware: snapshot ``ctx.tool_provider`` and short-circuit.

    Placed right before the terminal so the snapshot reflects every
    wrapping decision made by earlier middlewares.  Short-circuits
    (does NOT call ``next_``) so the terminal — and therefore the
    fastmcp Client's ``list_tools`` — never runs.  This keeps the
    test offline without needing ``async with client:``.
    """

    def __init__(self) -> None:
        self.name = "stack_inspector"
        self.seen_tool_provider: Any = None
        self.seen_stack: list[Any] = []

    async def __call__(self, ctx: RunContext, next_: Any) -> Any:
        self.seen_tool_provider = ctx.tool_provider
        self.seen_stack = list(ctx.extras.get("_tool_provider_stack", []))
        # Short-circuit: bypass the terminal entirely.
        return AgentResult(
            response="inspected",
            episode_id="ep-test",
            inference_ids=[],
            turns=0,
        )


def _build_orchestrator(
    inspector: _StackInspector,
    *,
    with_guardrails: bool,
) -> Agent:
    """Build an orchestrator agent with sub-agent + optional guardrail.

    The orchestrator's ``tool_provider`` is a bare :class:`FakeMCPClient`
    — :class:`SubAgentToolProvider` is auto-allocated by ``Agent.__init__``
    and wraps it as ``inner``.  No real fastmcp server is needed.
    """
    sub = Agent(
        AgentConfig(name="sub", function_name="sub_fn"),
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )

    guards: list[Guardrail] = (
        [cast(Guardrail, _NoopGuardrail())] if with_guardrails else []
    )

    # Build the default chain, then splice the inspector innermost.
    defaults = AgentConfig(
        name="orch",
        function_name="agent",
        guardrails=guards,
        sub_agents=[sub],
    )
    assert defaults.middleware is not None
    chain: list[Middleware] = [*defaults.middleware, inspector]

    cfg = AgentConfig(
        name="orch",
        function_name="agent",
        guardrails=guards,
        sub_agents=[sub],
        middleware=chain,
    )
    return Agent(
        cfg,
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),  # never reached
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


class TestToolProviderWrapOrder:
    async def test_guardrails_wraps_sub_agent_provider_outermost(self) -> None:
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        # OUTER wrap (what the terminal sees at ctx.tool_provider) is
        # GuardedToolProvider — guardrails fire before any tool call,
        # including sub-agent delegations.
        assert isinstance(inspector.seen_tool_provider, GuardedToolProvider)

    async def test_guarded_wraps_sub_agent_provider(self) -> None:
        # Direct structural check via the GuardedToolProvider's
        # ``_inner`` attribute (private but stable contract).
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        guarded = inspector.seen_tool_provider
        assert isinstance(guarded, GuardedToolProvider)
        # Peek at the wrapped provider — should be the auto-allocated
        # SubAgentToolProvider, NOT the bare client.
        inner = getattr(guarded, "_inner", None)
        assert isinstance(inner, SubAgentToolProvider)

    async def test_stack_records_sub_agent_provider_inside_guardrails(
        self,
    ) -> None:
        # ``_tool_provider_stack`` is the audit trail of prior providers
        # in chain order.  Post-refactor only Guardrails wraps; the
        # SubAgentToolProvider is already on the agent itself and never
        # gets replaced mid-chain.  So the stack length is 1 (the
        # original SubAgentToolProvider, replaced by GuardedToolProvider).
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        assert len(inspector.seen_stack) == 1
        # Index 0: the SubAgentToolProvider that was on the agent
        # before GuardrailsMiddleware wrapped it.
        assert isinstance(inspector.seen_stack[0], SubAgentToolProvider)


class TestNoGuardrailsLeavesSubAgentProviderOutermost:
    """When guardrails are absent, the auto-allocated provider is exposed directly."""

    async def test_no_guardrails_no_guarded_wrap(self) -> None:
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=False)

        await agent.invoke("delegate").result()

        # No wrap applied: the agent's auto-allocated provider IS the
        # terminal's tool_provider.
        assert isinstance(
            inspector.seen_tool_provider,
            SubAgentToolProvider,
        )
