"""Per-run ``Agent.invoke(tool_provider=…, memory=…)`` overrides.

The injection seam for conversation-scoped identity: a shared Agent can
run one invocation against a different tool provider (e.g. an
identity-bound runtime view) and a different memory provider (e.g. a
tenant-scoped ``TensorZeroMemory``) without mutating the agent.  These
tests pin the override semantics:

- the override reaches the loop (and only for that run);
- ``tool_provider=None`` is an explicit toolless run;
- the delegation surface (``SubAgentToolProvider``) is re-wrapped around
  an override instead of silently dropped;
- ``memory=`` swaps the provider for load, messages-hydration, and save;
- validation fails loud (protocol mismatch, memory-less agent);
- the ``json_mode='tool'`` safety check verifies an override without
  poisoning the per-agent latch.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pytest
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import Message, Text

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import JsonOutput
from t0_agents.core.types import ToolProvider
from t0_agents.memory.base import MemoryRecord
from t0_agents.testing import MockGateway
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeChatResponse, FakeT0

_EPISODE = UUID("00000000-0000-0000-0000-000000000004")


def _text_response(text: str) -> FakeChatResponse:
    return FakeChatResponse(
        content=[Text(type="text", text=text)], episode_id=_EPISODE
    )


class RecordingMemory:
    """MemoryProvider fake recording every load/save/load_history call."""

    def __init__(self) -> None:
        self.loads: list[str] = []
        self.saves: list[MemoryRecord] = []
        self.history_loads: list[str] = []

    async def load(self, episode_id: str) -> list[MemoryRecord]:
        self.loads.append(episode_id)
        return []

    async def save(self, record: MemoryRecord) -> None:
        self.saves.append(record)

    async def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        return []

    async def load_history(self, episode_id: str) -> list[Message] | None:
        self.history_loads.append(episode_id)
        return None


def _make_agent(
    *,
    t0: FakeT0 | None = None,
    client: FakeMCPClient | None = None,
    **config_overrides: Any,
) -> Agent:
    defaults: dict[str, Any] = {"name": "test", "function_name": "agent"}
    defaults.update(config_overrides)
    return Agent(
        AgentConfig(**defaults),
        t0=cast(AsyncTensorZeroGateway, t0 or FakeT0([_text_response("ok")])),
        tool_provider=cast(ToolProvider, client) if client is not None else None,
    )


class TestToolProviderOverride:
    async def test_override_reaches_the_loop(self) -> None:
        """The run lists tools from the override, not the agent's provider."""

        class RecordingClient(FakeMCPClient):
            def __init__(self, tool_name: str) -> None:
                super().__init__()
                self.listed = 0
                self._tool_name = tool_name

            async def list_tools(self) -> Any:
                self.listed += 1
                from mcp.types import Tool as MCPTool

                return [
                    MCPTool(
                        name=self._tool_name,
                        description="d",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ]

        base = RecordingClient("base_tool")
        override = RecordingClient("override_tool")
        t0 = FakeT0([_text_response("done")])
        agent = _make_agent(t0=t0, client=base)

        result = await agent.invoke("go", tool_provider=cast(ToolProvider, override))

        assert result.response == "done"
        assert override.listed == 1
        assert base.listed == 0
        sent = t0.calls[0]["additional_tools"]
        assert [tool["name"] for tool in sent] == ["override_tool"]

    async def test_override_is_per_run_not_sticky(self) -> None:
        """A later plain invoke uses the agent's own provider again."""
        base = FakeMCPClient()
        override = FakeMCPClient(tools=[])
        t0 = FakeT0([_text_response("one"), _text_response("two")])
        agent = _make_agent(t0=t0, client=base)

        await agent.invoke("first", tool_provider=cast(ToolProvider, override))
        await agent.invoke("second")

        assert agent._tool_provider is cast(ToolProvider, base)
        first_tools = t0.calls[0].get("additional_tools")
        second_tools = t0.calls[1]["additional_tools"]
        assert not first_tools
        assert [tool["name"] for tool in second_tools] == ["add"]

    async def test_override_none_is_a_toolless_run(self) -> None:
        """``tool_provider=None`` skips list_tools entirely for this run."""

        class ExplodingClient(FakeMCPClient):
            async def list_tools(self) -> Any:
                raise AssertionError("agent provider must not be consulted")

        t0 = FakeT0([_text_response("done")])
        agent = _make_agent(t0=t0, client=ExplodingClient())

        result = await agent.invoke("go", tool_provider=None)

        assert result.response == "done"
        assert not t0.calls[0].get("additional_tools")

    async def test_override_rewraps_sub_agents(self) -> None:
        """An override keeps the delegation surface: synthetic ``<name>_ask``
        tools are appended after the override's own tools."""
        sub = _make_agent(name="helper", t0=FakeT0([]))
        t0 = FakeT0([_text_response("done")])
        agent = _make_agent(t0=t0, sub_agents=[sub])
        override = FakeMCPClient()

        await agent.invoke("go", tool_provider=cast(ToolProvider, override))

        sent = [tool["name"] for tool in t0.calls[0]["additional_tools"]]
        assert sent == ["add", "helper_ask"]

    async def test_invalid_override_raises(self) -> None:
        agent = _make_agent()
        with pytest.raises(ConfigurationError, match="ToolProvider"):
            agent.invoke("go", tool_provider=cast(Any, object()))


class TestMemoryOverride:
    async def test_override_swaps_load_and_save(self) -> None:
        """Load + save go through the override; the agent's provider is idle."""
        base = RecordingMemory()
        override = RecordingMemory()
        t0 = FakeT0([_text_response("noted")])
        agent = _make_agent(t0=t0, memory=base)

        result = await agent.invoke(
            "remember", episode_id=str(_EPISODE), memory=override
        )

        assert result.response == "noted"
        assert override.loads and override.saves
        assert not base.loads and not base.saves

    async def test_override_swaps_messages_hydration(self) -> None:
        """``memory_strategy='messages'`` hydrates via the override's
        ``load_history``."""
        base = RecordingMemory()
        override = RecordingMemory()
        t0 = FakeT0([_text_response("hi")])
        agent = _make_agent(t0=t0, memory=base, memory_strategy="messages")
        episode = str(_EPISODE)

        await agent.invoke("hello", episode_id=episode, memory=override)

        assert override.history_loads == [episode]
        assert not base.history_loads

    async def test_override_requires_memory_enabled_agent(self) -> None:
        agent = _make_agent()
        with pytest.raises(ConfigurationError, match="memory-enabled"):
            agent.invoke("go", memory=RecordingMemory())

    async def test_override_rejects_non_provider(self) -> None:
        agent = _make_agent(memory=RecordingMemory())
        with pytest.raises(ConfigurationError, match="MemoryProvider"):
            agent.invoke("go", memory=cast(Any, None))


def _json_tool_agent(t0: MockGateway) -> Agent:
    config = AgentConfig(
        name="extractor",
        function_name="extract",
        output=JsonOutput(json_mode=cast(Any, "tool")),
    )
    return Agent(
        config,
        t0=cast(AsyncTensorZeroGateway, t0),
        tool_provider=cast(ToolProvider, FakeMCPClient(tools=[])),
    )


class TestJsonToolModeSafetyWithOverride:
    async def test_tool_advertising_override_rejected(self) -> None:
        """json_mode='tool' checks the override's advertised tools."""
        t0 = MockGateway.script_json([{"raw": '{"a": 1}', "parsed": {"a": 1}}])
        agent = _json_tool_agent(t0)

        with pytest.raises(ConfigurationError, match=r"§11\.1"):
            await agent.invoke(
                "go", tool_provider=cast(ToolProvider, FakeMCPClient())
            )

    async def test_override_check_does_not_poison_the_latch(self) -> None:
        """A failing override check must not latch ``verified_empty`` — the
        agent's own (empty) provider is still verified independently on the
        next plain run."""
        t0 = MockGateway.script_json([{"raw": '{"a": 1}', "parsed": {"a": 1}}])
        agent = _json_tool_agent(t0)

        with pytest.raises(ConfigurationError, match=r"§11\.1"):
            await agent.invoke(
                "go", tool_provider=cast(ToolProvider, FakeMCPClient())
            )
        assert agent._tool_provider_verified_empty is False

        result = await agent.invoke("go")
        assert result.parsed == {"a": 1}
        assert agent._tool_provider_verified_empty is True
