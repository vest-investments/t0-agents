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
from tensorzero.types import Text

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

    def test_unknown_keys_ignored_and_missing_default(self) -> None:
        decoded = decode_usage_meta(
            {USAGE_META_KEY: {"input_tokens": 7, "future_field": 99}}
        )
        assert decoded == RunUsage(input_tokens=7)
