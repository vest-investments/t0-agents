"""Unit tests for :class:`AgentConfig`'s structured-output surface.

v1.0 removed the v0.12 ``function_kind``/``json_mode``/``output_schema``/
``permissive_json_parsing``/``max_json_retries`` cluster (Tier 2) and
replaced it with a single ``output: OutputMode`` field.  These tests
cover:

- ``output`` defaults to :class:`ChatOutput`.
- The 5 legacy kwargs raise ``TypeError`` when passed.
- ``sub_agents`` with a single-shot output (e.g. ``JsonOutput``) raises
  ``ConfigurationError`` — single-shot mode would render sub-agent
  tools unreachable.
- ``memory_strategy='messages'`` is incompatible with single-shot
  output modes.

Path-based schema loading (v0.12 ``output_schema: Path``) is no longer
supported — users load the JSON themselves and pass a ``dict`` to
``JsonOutput(schema=...)``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import ChatOutput, JsonOutput


class TestDefaults:
    def test_default_output_is_chat_output(self) -> None:
        cfg = AgentConfig(name="a", function_name="f")
        assert isinstance(cfg.output, ChatOutput)
        assert cfg.output.expects_tool_loop is True
        assert cfg.output.supports_streaming is True
        # Read-side properties on AgentResult still work — v0.12 byte-
        # identity preserved by ``@property`` aliases.
        assert cfg.output.name == "chat"


class TestLegacyKwargsRemoved:
    """The 5 v0.12 legacy kwargs raise TypeError when passed."""

    def test_function_kind_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="function_kind"):
            AgentConfig(
                name="a",
                function_name="f",
                function_kind="json",  # type: ignore[call-arg]
            )

    def test_json_mode_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="json_mode"):
            AgentConfig(
                name="a",
                function_name="f",
                json_mode="strict",  # type: ignore[call-arg]
            )

    def test_output_schema_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="output_schema"):
            AgentConfig(
                name="a",
                function_name="f",
                output_schema={"type": "object"},  # type: ignore[call-arg]
            )

    def test_permissive_json_parsing_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="permissive_json_parsing"):
            AgentConfig(
                name="a",
                function_name="f",
                permissive_json_parsing=True,  # type: ignore[call-arg]
            )

    def test_max_json_retries_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="max_json_retries"):
            AgentConfig(
                name="a",
                function_name="f",
                max_json_retries=3,  # type: ignore[call-arg]
            )


class TestSingleShotForbidsSubAgents:
    """Single-shot output modes are incompatible with sub-agents."""

    def test_jsonoutput_forbids_sub_agents(self) -> None:
        # The sub_agents-must-be-Agent check fires *before* the
        # single-shot-vs-sub_agents check, so we need a real Agent instance.
        from tensorzero import AsyncTensorZeroGateway

        from t0_agents.core.agent import Agent
        from t0_agents.core.types import ToolProvider
        from tests._fake_mcp import FakeMCPClient
        from tests._fake_t0 import FakeT0

        sub = Agent(
            AgentConfig(name="sub", function_name="agent"),
            t0=cast(AsyncTensorZeroGateway, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        with pytest.raises(ConfigurationError, match="incompatible"):
            AgentConfig(
                name="orch",
                function_name="extract",
                output=JsonOutput(json_mode="strict"),
                sub_agents=[sub],
            )


class TestSingleShotForbidsMessagesMemory:
    """Single-shot output modes are incompatible with ``messages`` memory strategy."""

    def test_jsonoutput_forbids_messages_strategy(self) -> None:
        class _StubMemory:
            async def load(self, episode_id: str) -> list[Any]:
                return []

            async def save(self, record: Any) -> None: ...

            async def search(
                self, query: str, *, limit: int = 5
            ) -> list[Any]:
                return []

            async def load_history(self, episode_id: str) -> list[Any]:
                return []

        with pytest.raises(ConfigurationError, match="messages"):
            AgentConfig(
                name="extract",
                function_name="f",
                output=JsonOutput(json_mode="strict"),
                memory=cast(Any, _StubMemory()),
                memory_strategy="messages",
            )
