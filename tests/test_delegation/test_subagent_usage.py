"""Sub-agent billing rides home on ``CallToolResult.meta``.

A delegated sub-agent runs its own :class:`Run` with its own cost tracking, so
the parent must learn the child's billing some other way.  The transport is the
otherwise-unused ``meta`` field of the ``CallToolResult`` the
:class:`SubAgentToolProvider` returns: it carries the child Run's
:class:`RunUsage`, encoded by :func:`encode_usage_meta`.

This module covers the *writer* half (provider attaches meta) and the
encode/decode contract.  The orchestrator *reader* half — folding that billing
into the parent Run while leaving the parent footprint untouched — is exercised
in ``TestParentBillingRollup``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Text, ToolCall

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import (
    USAGE_META_KEY,
    RunUsage,
    ToolProvider,
    decode_usage_meta,
    encode_usage_meta,
)
from t0_agents.delegation.provider import SubAgentToolProvider
from t0_agents.delegation.routing import default_tool_name
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0, FakeUsage


def _sub_agent_with_usage(
    usage: FakeUsage,
    *,
    name: str = "researcher",
    middleware: list[Any] | None = None,
) -> Agent:
    """A sub-agent whose one text turn carries ``usage``.

    ``middleware=[]`` drops the child's ``CostTrackingMiddleware`` so its
    ``Run.usage()`` is ``None`` (the untracked-child shape).
    """
    config = (
        AgentConfig(name=name, function_name=f"{name}_fn", middleware=middleware)
        if middleware is not None
        else AgentConfig(name=name, function_name=f"{name}_fn")
    )
    t0 = FakeT0(
        [
            FakeChatResponse(
                content=[Text(type="text", text="sub-answer")], usage=usage,
            ),
        ],
    )
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


class TestProviderAttachesMeta:
    async def test_meta_carries_child_billing(self) -> None:
        child = _sub_agent_with_usage(
            FakeUsage(
                input_tokens=15, output_tokens=4, cost=0.003,
                provider_cache_read_input_tokens=6,
            ),
        )
        provider = SubAgentToolProvider([child])

        result = await provider.call_tool(default_tool_name(child), {"message": "hi"})

        assert result.is_error is False
        assert result.content[0].text == "sub-answer"
        usage = decode_usage_meta(result.meta)
        assert usage is not None
        assert usage.input_tokens == 15
        assert usage.output_tokens == 4
        assert usage.cost == pytest.approx(0.003)
        assert usage.provider_cache_read_input_tokens == 6
        assert usage.cost_is_priced is True

    async def test_meta_none_when_child_untracked(self) -> None:
        child = _sub_agent_with_usage(
            FakeUsage(input_tokens=15, output_tokens=4, cost=0.003),
            middleware=[],
        )
        provider = SubAgentToolProvider([child])

        result = await provider.call_tool(default_tool_name(child), {"message": "hi"})

        assert result.is_error is False
        assert result.meta is None


class TestUsageMetaCodec:
    def test_roundtrip(self) -> None:
        ru = RunUsage(
            input_tokens=5, output_tokens=2, cost=0.1,
            provider_cache_read_input_tokens=3,
            provider_cache_write_input_tokens=1, cost_is_priced=True,
        )
        assert decode_usage_meta(encode_usage_meta(ru)) == ru

    def test_encode_uses_namespaced_key(self) -> None:
        encoded = encode_usage_meta(RunUsage(input_tokens=1))
        assert set(encoded) == {USAGE_META_KEY}

    def test_none_and_empty_decode_to_none(self) -> None:
        assert decode_usage_meta(None) is None
        assert decode_usage_meta({}) is None

    def test_foreign_meta_decodes_to_none(self) -> None:
        assert decode_usage_meta({"some/other": {"input_tokens": 9}}) is None

    @pytest.mark.parametrize("payload", ["not-a-dict", 123, None, []])
    def test_non_mapping_payload_decodes_to_none(self, payload: Any) -> None:
        assert decode_usage_meta({USAGE_META_KEY: payload}) is None

    @pytest.mark.parametrize("meta", [[1, 2], "nope", 123])
    def test_non_dict_meta_decodes_to_none(self, meta: Any) -> None:
        # A truthy non-dict ``meta`` must fold to None, not raise AttributeError.
        assert decode_usage_meta(meta) is None

    def test_unknown_keys_ignored_and_missing_default(self) -> None:
        decoded = decode_usage_meta(
            {USAGE_META_KEY: {"input_tokens": 7, "future_field": 99}}
        )
        assert decoded == RunUsage(input_tokens=7)


def _orchestrator(child: Agent, turn1: FakeUsage, turn2: FakeUsage) -> Agent:
    """A parent that delegates to ``child`` on turn 1, then answers on turn 2.

    Turn 1 is a tool call to the child's synthetic tool (so the loop dispatches
    the sub-agent); turn 2 is text (so the loop ends).
    """
    child_tool = default_tool_name(child)
    parent_t0 = FakeT0(
        [
            FakeChatResponse(
                content=[
                    ToolCall(
                        id="tc-1",
                        raw_name=child_tool,
                        name=child_tool,
                        raw_arguments='{"message": "go"}',
                        arguments={"message": "go"},
                    ),
                ],
                usage=turn1,
            ),
            FakeChatResponse(
                content=[Text(type="text", text="final")], usage=turn2,
            ),
        ],
    )
    return Agent(
        AgentConfig(name="orch", function_name="orch_fn", sub_agents=[child]),
        t0=cast(AsyncTensorZeroGateway, parent_t0),
        tool_provider=None,
    )


class TestParentBillingRollup:
    """The orchestrator-side fold (acceptance §7): billing up, footprint not."""

    async def test_parent_usage_includes_child_footprint_excludes(self) -> None:
        child = _sub_agent_with_usage(
            FakeUsage(input_tokens=30, output_tokens=5, cost=0.003),
        )
        parent = _orchestrator(
            child,
            FakeUsage(input_tokens=100, output_tokens=10, cost=0.01),
            FakeUsage(input_tokens=120, output_tokens=8, cost=0.012),
        )

        run = parent.invoke("delegate please")
        result = await run.result()
        assert result.response == "final"

        usage = run.usage()
        assert usage is not None
        # Billing rolls up: parent turns (100 + 120) PLUS the child's 30.
        # If the child were excluded this would be 220 — the +30 is the proof.
        assert usage.input_tokens == 250
        assert usage.output_tokens == 23
        assert usage.cost == pytest.approx(0.025)

        # Footprint stays parent-only: the parent's latest turn (120), neither
        # the billing sum (250) nor inflated by the child's 30-token prompt.
        footprint = run.context_footprint()
        assert footprint is not None
        assert footprint.input_tokens == 120

    async def test_child_priced_flag_rolls_up_to_unpriced_parent(self) -> None:
        # Child reports a real $0.00 (priced); both parent turns are unpriced.
        child = _sub_agent_with_usage(
            FakeUsage(input_tokens=10, output_tokens=2, cost=0.0),
        )
        parent = _orchestrator(
            child,
            FakeUsage(input_tokens=50, output_tokens=5, cost=None),
            FakeUsage(input_tokens=60, output_tokens=4, cost=None),
        )

        run = parent.invoke("go")
        await run.result()

        usage = run.usage()
        assert usage is not None
        # The child's priced $0.00 makes the whole run priced (OR rolls up).
        assert usage.cost_is_priced is True
        assert usage.cost == pytest.approx(0.0)
        # Billing tokens still include the child (50 + 60 + 10).
        assert usage.input_tokens == 120
