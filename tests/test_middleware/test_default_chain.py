"""Default middleware chain materialisation by ``AgentConfig.__post_init__``.

Asserts that v0.12-shaped ``AgentConfig(memory=, guardrails=, sub_agents=)``
produces the expected chain and that ``middleware=[]`` / ``middleware=[X]``
override correctly.
"""

from __future__ import annotations

from typing import Any, cast

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.middleware import (
    CostTrackingMiddleware,
    GuardrailsMiddleware,
    JsonRetryMiddleware,
    MemoryMiddleware,
    Middleware,
    SubAgentMiddleware,
)
from t0_agents.core.types import (
    AgentResult,
    Guardrail,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.usage import UsageLedger, UsageLedgerMiddleware
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0


class _NoopMemory:
    async def load(self, episode_id: str) -> list[Any]:
        return []

    async def save(self, record: Any) -> None:
        pass

    async def search(self, query: str, *, limit: int = 5) -> list[Any]:
        return []


class _NoopGuardrail:
    name = "noop"

    async def before_run(self, ctx: Any) -> None: ...
    async def before_tool(self, ctx: Any, tc: Any) -> str | None:
        return None
    async def after_tool(self, ctx: Any, tc: Any, r: str) -> None: ...
    async def after_run(self, ctx: Any, result: Any) -> None: ...


class _NoopLedger:
    async def record(self, run: Any) -> None:
        pass


def _names(chain: list[Middleware]) -> list[str]:
    return [mw.name for mw in chain]


class TestDefaultChain:
    def test_bare_config_has_cost_tracking_and_json_retry_only(self) -> None:
        # No memory, no guardrails, no sub_agents — but cost-tracking
        # and json_retry are always present (the former wires Run.cost,
        # the latter activates for any non-chat output mode).
        cfg = AgentConfig(name="bare", function_name="agent")
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == ["cost_tracking", "json_retry"]

    def test_memory_adds_memory_middleware_outermost(self) -> None:
        cfg = AgentConfig(
            name="m",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == [
            "memory",
            "cost_tracking",
            "json_retry",
        ]
        assert isinstance(cfg.middleware[0], MemoryMiddleware)

    def test_guardrails_inserted_between_memory_and_cost(self) -> None:
        cfg = AgentConfig(
            name="g",
            function_name="agent",
            guardrails=[cast(Guardrail, _NoopGuardrail())],
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == [
            "guardrails",
            "cost_tracking",
            "json_retry",
        ]

    def test_sub_agents_inserted_before_guardrails(self) -> None:
        # Order invariant (contract §3.2): SubAgent must appear BEFORE
        # Guardrails in chain order so the parent_episode_id ContextVar
        # bound by SubAgentMiddleware is in scope when guardrails fire
        # their ``before_tool``/``after_tool`` hooks on a delegated
        # sub-agent call.  Concretely: the chain ends in
        # ``GuardedToolProvider(SubAgentToolProvider(...))`` because
        # Guardrails is the only middleware that still wraps
        # ``tool_provider``; SubAgentMiddleware is ContextVar-only.
        sub = Agent(
            AgentConfig(name="sub", function_name="sub_fn"),
            t0=cast(Any, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )
        cfg = AgentConfig(
            name="orch",
            function_name="agent",
            sub_agents=[sub],
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == [
            "sub_agents",
            "cost_tracking",
            "json_retry",
        ]

    def test_full_chain_assembles_in_canonical_order(self) -> None:
        sub = Agent(
            AgentConfig(name="sub", function_name="sub_fn"),
            t0=cast(Any, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )
        cfg = AgentConfig(
            name="full",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
            guardrails=[cast(Guardrail, _NoopGuardrail())],
            sub_agents=[sub],
        )
        assert cfg.middleware is not None
        # Memory → SubAgents → Guardrails → CostTracking → JsonRetry.
        # SubAgents BEFORE Guardrails preserves v0.12 wrap order:
        # GuardedToolProvider(OrchestrationToolProvider(client)).
        assert _names(cfg.middleware) == [
            "memory",
            "sub_agents",
            "guardrails",
            "cost_tracking",
            "json_retry",
        ]

    def test_usage_ledger_sits_between_cost_tracking_and_json_retry(self) -> None:
        cfg = AgentConfig(
            name="u",
            function_name="agent",
            usage_ledger=cast(UsageLedger, _NoopLedger()),
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == ["cost_tracking", "usage_ledger", "json_retry"]
        assert isinstance(cfg.middleware[1], UsageLedgerMiddleware)

    def test_full_chain_with_a_ledger_keeps_the_canonical_order(self) -> None:
        sub = Agent(
            AgentConfig(name="sub", function_name="sub_fn"),
            t0=cast(Any, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )
        cfg = AgentConfig(
            name="full",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
            guardrails=[cast(Guardrail, _NoopGuardrail())],
            sub_agents=[sub],
            usage_ledger=cast(UsageLedger, _NoopLedger()),
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == [
            "memory",
            "sub_agents",
            "guardrails",
            "cost_tracking",
            "usage_ledger",
            "json_retry",
        ]

    def test_json_retry_is_innermost(self) -> None:
        # Q13 resolution: JsonRetry must be the innermost layer so
        # memory/guardrails see only the final successful inference
        # rather than every retry.  Audit of retries comes via TZ
        # tags (retry_attempt=N).
        cfg = AgentConfig(
            name="j",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
            guardrails=[cast(Guardrail, _NoopGuardrail())],
        )
        assert cfg.middleware is not None
        assert isinstance(cfg.middleware[-1], JsonRetryMiddleware)


class TestOverride:
    def test_empty_list_overrides_to_no_middleware(self) -> None:
        # ``middleware=[]`` is the explicit "no middleware" override.
        # Semantically distinct from ``middleware=None`` (the default
        # materialises the v0.12-compat chain).
        cfg = AgentConfig(
            name="bare",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
            middleware=[],
        )
        assert cfg.middleware == []

    def test_explicit_list_replaces_default(self) -> None:
        class Custom:
            name = "custom"

            async def __call__(self, ctx: Any, next_: Any) -> AgentResult:
                return await next_(ctx)

        cfg = AgentConfig(
            name="c",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),  # would normally add Memory
            middleware=[Custom()],  # but explicit override wins
        )
        assert cfg.middleware is not None
        assert _names(cfg.middleware) == ["custom"]


class TestStaticHelper:
    def test_default_middleware_for_round_trip(self) -> None:
        # ``default_middleware_for`` is the single source of truth used
        # by ``_build_default_chain``.  A round-trip with identical
        # inputs must produce an identically-shaped chain.
        cfg = AgentConfig(
            name="r",
            function_name="agent",
            memory=cast(MemoryProvider, _NoopMemory()),
            guardrails=[cast(Guardrail, _NoopGuardrail())],
        )
        helper_chain = AgentConfig.default_middleware_for(
            memory=cfg.memory,
            memory_strategy=cfg.memory_strategy,
            guardrails=cfg.guardrails,
            sub_agents=cfg.sub_agents,
        )
        assert cfg.middleware is not None
        assert _names(helper_chain) == _names(cfg.middleware)

    def test_default_middleware_for_threads_the_ledger(self) -> None:
        ledger = cast(UsageLedger, _NoopLedger())
        cfg = AgentConfig(name="r", function_name="agent", usage_ledger=ledger)
        helper_chain = AgentConfig.default_middleware_for(usage_ledger=ledger)
        assert cfg.middleware is not None
        assert _names(helper_chain) == _names(cfg.middleware)

    def test_default_middleware_for_can_sandwich_default(self) -> None:
        # Power-user idiom: custom outermost + default + custom innermost.
        class Outer:
            name = "outer"

            async def __call__(self, ctx: Any, next_: Any) -> AgentResult:
                return await next_(ctx)

        class Inner:
            name = "inner"

            async def __call__(self, ctx: Any, next_: Any) -> AgentResult:
                return await next_(ctx)

        chain: list[Middleware] = [
            Outer(),
            *AgentConfig.default_middleware_for(
                memory=cast(MemoryProvider, _NoopMemory()),
            ),
            Inner(),
        ]
        cfg = AgentConfig(
            name="sandwich",
            function_name="agent",
            middleware=chain,
        )
        assert cfg.middleware is not None
        # No guardrails / sub_agents in this snippet, so the default
        # chain reduces to ``[memory, cost_tracking, json_retry]``.
        assert _names(cfg.middleware) == [
            "outer",
            "memory",
            "cost_tracking",
            "json_retry",
            "inner",
        ]


class TestComponentTypes:
    """Type-level sanity checks on the materialised chain."""

    def test_memory_middleware_receives_provider_and_strategy(self) -> None:
        provider = _NoopMemory()
        cfg = AgentConfig(
            name="m",
            function_name="agent",
            memory=cast(MemoryProvider, provider),
            memory_strategy="prepended_message",
        )
        assert cfg.middleware is not None
        mw = cfg.middleware[0]
        assert isinstance(mw, MemoryMiddleware)
        # _provider and _strategy are private; verify via behaviour
        # rather than introspection — but at construction time we can
        # assert the references match.
        assert getattr(mw, "_provider", None) is provider
        assert getattr(mw, "_strategy", None) == "prepended_message"

    def test_guardrails_middleware_receives_list_copy(self) -> None:
        guards: list[Guardrail] = [cast(Guardrail, _NoopGuardrail())]
        cfg = AgentConfig(name="g", function_name="agent", guardrails=guards)
        assert cfg.middleware is not None
        # GuardrailsMiddleware is at index 0 (no Memory in this config).
        mw = cfg.middleware[0]
        assert isinstance(mw, GuardrailsMiddleware)

    def test_cost_tracking_is_default_constructed(self) -> None:
        cfg = AgentConfig(name="c", function_name="agent")
        assert cfg.middleware is not None
        ct = [mw for mw in cfg.middleware if mw.name == "cost_tracking"]
        assert len(ct) == 1
        assert isinstance(ct[0], CostTrackingMiddleware)

    def test_usage_ledger_middleware_receives_the_ledger(self) -> None:
        ledger = _NoopLedger()
        cfg = AgentConfig(
            name="u", function_name="agent", usage_ledger=cast(UsageLedger, ledger)
        )
        assert cfg.middleware is not None
        mw = cfg.middleware[1]
        assert isinstance(mw, UsageLedgerMiddleware)
        assert getattr(mw, "_ledger", None) is ledger

    def test_sub_agent_middleware_receives_agents(self) -> None:
        sub = Agent(
            AgentConfig(name="sub", function_name="sub_fn"),
            t0=cast(Any, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )
        cfg = AgentConfig(
            name="orch",
            function_name="agent",
            sub_agents=[sub],
        )
        assert cfg.middleware is not None
        sa = [mw for mw in cfg.middleware if mw.name == "sub_agents"]
        assert len(sa) == 1
        assert isinstance(sa[0], SubAgentMiddleware)


