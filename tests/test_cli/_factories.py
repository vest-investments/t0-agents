"""Module-level agent factories used by ``test_loader`` and ``test_info_cmd``.

The CLI's ``load_agent_factory`` resolves ``module:attr`` via
``importlib.import_module``, so the target must be importable by its
fully-qualified name.  Tests use these factories as the ``attr`` half of
the spec — e.g. ``tests.test_cli._factories:fake_factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.output import OutputMode
from t0_agents.core.run import Run, RunInput
from t0_agents.core.types import AgentResult, ToolProvider
from t0_agents.glow import App as GlowApp
from t0_agents.stream.events import AgentEvent, DoneEvent
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


class _ScriptedRun(Run[Any]):
    """:class:`Run` subclass whose ``events()`` yields scripted events.

    Built non-eagerly (no background drive task) so ``events()`` can
    return the scripted iterator directly without contending with the
    middleware chain.
    """

    def __init__(
        self,
        agent: Agent | None,
        input_: RunInput,
        output: OutputMode[Any],
        events_or_exc: tuple[list[AgentEvent], BaseException | None],
    ) -> None:
        super().__init__(agent, input_, output, eager=False)
        self._scripted_events, self._scripted_exception = events_or_exc

    async def events(self, replay: bool = False) -> AsyncIterator[AgentEvent]:  # type: ignore[override]
        if self._scripted_exception is not None:
            raise self._scripted_exception
        for event in self._scripted_events:
            if isinstance(event, DoneEvent):
                self._result = event.result
                self._state = "succeeded"
            yield event


class _ScriptedAgent(Agent):
    """:class:`Agent` subclass whose ``invoke()`` returns a scripted Run.

    Reads turns from :data:`SCRIPTED_TURNS`, records each call into
    :data:`SCRIPTED_CALLS`, and raises from :data:`SCRIPTED_EXCEPTIONS`
    when that slot is populated for a turn.
    """

    def invoke(  # type: ignore[override]
        self,
        user_message: str,
        *,
        output: OutputMode[Any] | None = None,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        system: str | None = None,
        prior_messages: Sequence[Mapping[str, Any]] | None = None,
        middleware: Sequence[Any] | None = None,
        eager: bool = True,
        context: Mapping[str, Any] | None = None,
    ) -> Run[Any]:
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
        events: list[AgentEvent] = []
        if exc is None:
            if not SCRIPTED_TURNS:
                raise AssertionError("_ScriptedAgent: no scripted turn queued")
            events = SCRIPTED_TURNS.pop(0)
        input_ = RunInput(
            user_message=user_message,
            function_name=self.config.function_name,
            episode_id=episode_id,
            tags=tags,
        )
        return _ScriptedRun(self, input_, self.config.output, (events, exc))


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


# ---------------------------------------------------------------------------
# Glow App specs (M7) — module-level Apps the loader must adapt.
# Overrides make them gateway-free; the scripted agents read the same
# SCRIPTED_* registries as `scripted_factory`, so the conftest reset
# fixture covers them too.
# ---------------------------------------------------------------------------


def _scripted(name: str) -> _ScriptedAgent:
    return _ScriptedAgent(
        AgentConfig(name=name, function_name="agent", max_turns=3),
        t0=cast(Any, FakeT0(responses=[])),
        tool_provider=cast(ToolProvider, FakeMCPClient()),
    )


glow_app = GlowApp()
"""Single-agent glow App — the loader's sole-agent default path."""


@glow_app.agent("solo")
async def _solo(query: str) -> str:  # type: ignore[empty-body]
    """Scripted solo agent."""


glow_app.agent_overrides["solo"] = _scripted("solo")


glow_app_two = GlowApp()
"""Two-agent glow App — exercises --agent selection and its errors."""


@glow_app_two.agent("alpha")
async def _alpha(query: str) -> str:  # type: ignore[empty-body]
    """Alpha."""


@glow_app_two.agent("beta")
async def _beta(query: str) -> str:  # type: ignore[empty-body]
    """Beta."""


glow_app_two.agent_overrides["alpha"] = _scripted("alpha")
glow_app_two.agent_overrides["beta"] = _scripted("beta")


glow_app_empty = GlowApp()
"""Glow App with no registered agents — the loader must reject it."""
