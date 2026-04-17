"""Module-level agent factories used by ``test_loader`` and ``test_info_cmd``.

The CLI's ``load_agent_factory`` resolves ``module:attr`` via
``importlib.import_module``, so the target must be importable by its
fully-qualified name.  Tests use these factories as the ``attr`` half of
the spec — e.g. ``tests.test_cli._factories:fake_factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.stream.events import AgentEvent
from tests._fake_mcp import FakeMCPClient
from tests._fake_t0 import FakeT0

# Module-level registries consumed by the scripted factories below.
# Tests mutate these before calling `CliRunner.invoke(...)` and clear
# them with the `scripted_state` fixture in `conftest.py`.
SCRIPTED_TURNS: list[list[AgentEvent]] = []
SCRIPTED_CALLS: list[dict[str, Any]] = []
SCRIPTED_EXCEPTIONS: list[BaseException | None] = []


def build_agent(
    *,
    name: str = "test-agent",
    function_name: str = "agent",
    max_turns: int = 5,
    tags: dict[str, str] | None = None,
) -> Agent:
    """Construct an Agent over FakeT0 + FakeMCPClient."""
    t0 = FakeT0(responses=[])
    mcp = FakeMCPClient()
    return Agent(
        AgentConfig(
            name=name,
            function_name=function_name,
            max_turns=max_turns,
            tags=tags,
        ),
        t0=cast(Any, t0),
        tool_provider=cast(ToolProvider, mcp),
    )


@asynccontextmanager
async def fake_factory() -> AsyncIterator[Agent]:
    """Valid factory: yields a fresh Agent, no external resources."""
    yield build_agent()


def not_a_factory() -> str:
    """Non-Agent, callable — used to verify the loader still accepts callables.

    ``load_agent_factory`` does not attempt to call the object; it only
    checks that it is callable and not a bare Agent.  The CLI will fail
    later when trying to ``async with not_a_factory() as agent:``.
    """
    return "not an agent"


fake_agent_instance: Agent = build_agent(name="bare-agent")
"""Module-level bare Agent — used to prove the loader rejects it."""


not_callable: dict[str, int] = {"foo": 1}
"""Non-callable attribute — used to verify the loader rejects non-callables."""


class _ScriptedAgent(Agent):
    """Agent subclass whose ``stream`` yields scripted events.

    Reads turns from :data:`SCRIPTED_TURNS`, records each call into
    :data:`SCRIPTED_CALLS`, and raises from :data:`SCRIPTED_EXCEPTIONS`
    when that slot is populated for a turn.
    """

    async def stream(  # type: ignore[override]
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        SCRIPTED_CALLS.append(
            {
                "user_message": user_message,
                "episode_id": episode_id,
                "tags": tags,
            },
        )
        exc: BaseException | None = None
        if SCRIPTED_EXCEPTIONS:
            exc = SCRIPTED_EXCEPTIONS.pop(0)
        if exc is not None:
            raise exc
        if not SCRIPTED_TURNS:
            raise AssertionError("_ScriptedAgent: no scripted turn queued")
        for event in SCRIPTED_TURNS.pop(0):
            yield event


@asynccontextmanager
async def scripted_factory() -> AsyncIterator[Agent]:
    """Factory used by ``test_run_cmd`` — yields a fresh scripted agent.

    The script is driven by :data:`SCRIPTED_TURNS` /
    :data:`SCRIPTED_EXCEPTIONS` — tests populate these before invoking
    the CLI.
    """
    t0 = FakeT0(responses=[])
    mcp = FakeMCPClient()
    agent = _ScriptedAgent(
        AgentConfig(name="scripted-agent", function_name="agent", max_turns=3),
        t0=cast(Any, t0),
        tool_provider=cast(ToolProvider, mcp),
    )
    yield agent


def emit_agent_result(response: str = "ok", episode_id: str | None = None) -> AgentResult:
    """Shorthand for building an AgentResult in test scripts."""
    return AgentResult(
        response=response,
        episode_id=episode_id,
        inference_ids=[],
        turns=1,
    )
