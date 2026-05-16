"""Tests for :meth:`Run.replay`.

Resolves open question Q7: replay creates a NEW TensorZero episode by
default; ``replay(episode_id=original.episode_id)`` opts in to
preserving the original correlation.  Also covers the weakref-GC
failure mode — if the original Agent has been garbage-collected and
no explicit ``agent=`` is passed, the call raises ``RuntimeError``.
"""

from __future__ import annotations

import gc
from typing import Any, cast

import pytest
from tensorzero import AsyncTensorZeroGateway, Text

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0


def _resp(text: str = "ok") -> FakeChatResponse:
    return FakeChatResponse(content=[Text(type="text", text=text)])


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "primary", "function_name": "agent"}
    defaults.update(config_overrides)
    config = AgentConfig(**defaults)
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_resp()])),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


class TestReplaySameAgent:
    async def test_replay_mints_new_episode_by_default(self) -> None:
        # Q7 resolution: replay defaults to NEW episode_id.  Scripts
        # two responses so each Run gets a fresh inference.
        fake_t0 = FakeT0([_resp("first"), _resp("second")])
        agent = _make_agent(t0=fake_t0)
        original = agent.invoke("hello")
        await original.result()
        replayed = await original.replay()
        await replayed.result()
        # Both runs hit the fake; episode_ids are distinct.
        assert original.episode_id is not None
        assert replayed.episode_id is not None
        assert original.episode_id != replayed.episode_id

    async def test_replay_with_explicit_episode_id_preserves_correlation(
        self,
    ) -> None:
        fake_t0 = FakeT0([_resp("first"), _resp("second")])
        agent = _make_agent(t0=fake_t0)
        original = agent.invoke("hello")
        await original.result()
        replayed = await original.replay(episode_id=original.episode_id)
        # The new Run's input carries the explicit episode_id.
        assert replayed.input.episode_id == original.episode_id
        await replayed.result()


class TestReplayDifferentAgent:
    async def test_replay_against_a_different_agent(self) -> None:
        fake_t0_a = FakeT0([_resp("from-a")])
        fake_t0_b = FakeT0([_resp("from-b")])
        agent_a = _make_agent(t0=fake_t0_a)
        agent_b = _make_agent(t0=fake_t0_b, name="secondary")
        original = agent_a.invoke("hello")
        await original.result()
        replayed = await original.replay(agent=agent_b)
        result = await replayed.result()
        # The replay ran against agent_b's gateway, not agent_a's.
        assert result.response == "from-b"
        assert len(fake_t0_b.calls) == 1


class TestWeakrefGC:
    async def test_replay_after_agent_gc_raises_runtime_error(self) -> None:
        # Hold the gateway in the test's outer scope so it stays alive,
        # but discard every reference to the Agent so the weakref
        # inside the Run is the only path back.
        fake_t0 = FakeT0([_resp("hello")])
        agent = _make_agent(t0=fake_t0)
        run = agent.invoke("hi")
        await run.result()
        del agent
        gc.collect()
        with pytest.raises(RuntimeError, match="garbage-collected"):
            await run.replay()


class TestVariantTag:
    """variant= is forwarded as a TensorZero tag for replay audit."""

    async def test_variant_kwarg_lands_on_input_tags(self) -> None:
        fake_t0 = FakeT0([_resp("first"), _resp("second")])
        agent = _make_agent(t0=fake_t0)
        original = agent.invoke("hello", tags={"feature": "x"})
        await original.result()
        replayed = await original.replay(variant="experiment-42")
        # Original tags are preserved; variant lands under
        # ``t0_agents.variant``.
        assert replayed.input.tags is not None
        assert dict(replayed.input.tags) == {
            "feature": "x",
            "t0_agents.variant": "experiment-42",
        }
        await replayed.result()
