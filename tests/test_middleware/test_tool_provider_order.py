"""Order invariant: ``GuardrailsMiddleware`` wraps ``SubAgentMiddleware``.

Contract §3.2 invariant: ``SubAgentMiddleware`` wraps ``tool_provider``
INSIDE ``GuardrailsMiddleware`` so orchestrator-level guardrails
observe sub-agent delegations via their ``before_tool``/``after_tool``
hooks.

Concretely: at terminal time ``ctx.tool_provider`` must be a
:class:`GuardedToolProvider` wrapping a :class:`_OrchestrationToolProvider`
wrapping the bare client.  The default chain order ``Memory →
SubAgents → Guardrails → CostTracking → JsonRetry`` produces exactly
this because each middleware's ``replace_tool_provider`` runs in
chain order — SubAgents wraps first (innermost), Guardrails wraps
next (outermost).
"""

from __future__ import annotations

from typing import Any, cast

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.middleware.base import Middleware, RunContext
from t0_agents.core.types import AgentResult, Guardrail, ToolProvider
from t0_agents.delegation.orchestrator import _OrchestrationToolProvider
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

    Uses a real ``fastmcp.Client(FastMCP(...))`` because
    ``install_sub_agents`` requires the orchestrator's tool_provider to
    be a fastmcp Client instance (it registers each sub-agent as a tool
    on the FastMCP server).  The sub-agent itself uses
    :class:`FakeMCPClient` since it has no nested sub-agents.
    """
    sub = Agent(
        AgentConfig(name="sub", function_name="sub_fn"),
        t0=cast(AsyncTensorZeroGateway, FakeT0([])),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )

    server = FastMCP("orch")
    client: Client[Any] = Client(server)

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
        tool_provider=cast(ToolProvider, client),
    )


class TestToolProviderWrapOrder:
    async def test_guardrails_wraps_orchestration_outermost(self) -> None:
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        # OUTER wrap (what the terminal sees at ctx.tool_provider) is
        # GuardedToolProvider — guardrails fire before any tool call,
        # including sub-agent delegations.
        assert isinstance(inspector.seen_tool_provider, GuardedToolProvider)

    async def test_stack_records_orchestration_inside_guardrails(self) -> None:
        # ``_tool_provider_stack`` is the audit trail of prior
        # providers in chain order — index 0 is the original client
        # (replaced first by SubAgent's wrap), index 1 is the
        # ``_OrchestrationToolProvider`` (replaced next by
        # Guardrails's wrap).  ``GuardedToolProvider`` is the current
        # ``ctx.tool_provider`` so it does NOT appear in the stack.
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        assert len(inspector.seen_stack) == 2
        # Index 1: _OrchestrationToolProvider (SubAgent's wrap, INNER
        # of GuardedToolProvider but OUTER relative to the raw client).
        assert isinstance(inspector.seen_stack[1], _OrchestrationToolProvider)

    async def test_outer_guarded_wraps_inner_orchestration(self) -> None:
        # Direct structural check via the GuardedToolProvider's
        # ``_inner`` attribute (private but stable in v0.12).
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=True)

        await agent.invoke("delegate").result()

        guarded = inspector.seen_tool_provider
        assert isinstance(guarded, GuardedToolProvider)
        # Peek at the wrapped provider — should be the orchestration
        # wrap, NOT the bare client.
        inner = getattr(guarded, "_inner", None)
        assert isinstance(inner, _OrchestrationToolProvider)


class TestNoGuardrailsLeavesOrchestrationOutermost:
    """When guardrails are absent, sub-agents are the only wrap."""

    async def test_no_guardrails_no_guarded_wrap(self) -> None:
        inspector = _StackInspector()
        agent = _build_orchestrator(inspector, with_guardrails=False)

        await agent.invoke("delegate").result()

        # Only one wrap applied: _OrchestrationToolProvider.
        assert isinstance(
            inspector.seen_tool_provider,
            _OrchestrationToolProvider,
        )
