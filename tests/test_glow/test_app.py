"""Unit tests for :class:`t0_agents.glow.App` and ``@app.agent`` (M2).

Ten cases (no TZ gateway), covering:

* Registration: decorator inserts an ``_AgentSpec`` in
  ``app._agent_registry`` and the spec reflects the function's
  docstring / return-type / function_name.
* Duplicate-name registration raises ``ConfigurationError``.
* Forward-ref failure on the return annotation surfaces as
  ``ConfigurationError`` at decoration time (not invocation).
* ``wrapper("msg")`` returns a :class:`FluentRun`.
* ``await wrapper("msg")`` returns the unwrapped value (mock ``Agent``
  whose ``invoke()`` returns a fake :class:`Run`).
* ``wrapper._agent`` is ``None`` before first build; populated after.
* ``wrapper._spec`` exposes the underlying ``_AgentSpec``.
* ``await app.build()`` constructs every registered agent once
  (subsequent invocations reuse).
* ``list[BaseModel]`` returns: the synthetic-wrapper ``.items`` is
  unwrapped to ``list[T]`` (M1 carry-forward §4(d)).
* ``typing.assert_type`` confirms the ParamSpec/TypeVar typing surface
  (mypy reveal-type proxy — runtime no-op, strict-mode check at
  ``uv run mypy`` time).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, ClassVar, assert_type, cast

import pytest
from pydantic import BaseModel
from tensorzero import AsyncTensorZeroGateway

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow import App, FluentRun
from t0_agents.stream.events import AgentEvent


class _Trip(BaseModel):
    destinations: list[str]
    cost_usd: float


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_t0() -> AsyncTensorZeroGateway:
    """Return an opaque sentinel cast to ``AsyncTensorZeroGateway``.

    :class:`Agent.__init__` does not touch ``t0`` (it only stores the
    reference), so a bare sentinel suffices for construction-only
    tests.  Tests that actually invoke patch :class:`Agent` in
    ``glow.app`` to a fake class.
    """
    return cast(AsyncTensorZeroGateway, object())


class _FakeRun:
    """Duck-typed stand-in for :class:`Run` consumed by :class:`FluentRun`.

    Same shape as ``tests/test_glow/test_fluent_run._FakeRun`` — the
    constants here are deliberately small so test assertions can be
    exact.
    """

    def __init__(
        self,
        *,
        value: Any = "hello-from-fake",
        episode_id: str | None = "ep_app_test",
    ) -> None:
        self._value = value
        self._episode_id = episode_id

    async def result(self) -> AgentResult[Any]:
        return AgentResult(
            value=self._value,
            episode_id=self._episode_id,
            inference_ids=["if_app_test"],
            turns=1,
        )

    async def events(self) -> AsyncIterator[AgentEvent]:
        if False:  # pragma: no cover - generator-shape marker
            yield


class _FakeAgent:
    """Fake :class:`Agent` patched into ``glow.app`` for invoke-path tests.

    Records the constructor args (for the build-once assertion) and
    returns a configurable :class:`_FakeRun` from ``invoke()``.
    """

    construct_count: ClassVar[int] = 0
    last_invoke_kwargs: ClassVar[dict[str, Any]] = {}
    fake_value: ClassVar[Any] = "hello-from-fake"

    def __init__(self, config: Any, *, t0: Any, tool_provider: Any = None) -> None:
        type(self).construct_count += 1
        self._config = config
        self._t0 = t0

    def invoke(self, user_message: str, **kwargs: Any) -> Run[Any]:
        type(self).last_invoke_kwargs = {"user_message": user_message, **kwargs}
        return cast(Run[Any], _FakeRun(value=type(self).fake_value))


@pytest.fixture
def app_with_fake_agent(monkeypatch: pytest.MonkeyPatch) -> App:
    """Provide an :class:`App` whose internal :class:`Agent` is the fake.

    Resets the fake's class-level counters before yielding so each
    test sees a clean slate.
    """
    _FakeAgent.construct_count = 0
    _FakeAgent.last_invoke_kwargs = {}
    _FakeAgent.fake_value = "hello-from-fake"
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    return App(t0=_fake_t0())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_decorator_registers_agent_in_registry() -> None:
    """``@app.agent`` populates ``app._agent_registry`` and the spec
    reflects the function's docstring / return-type / function_name.
    """
    app = App(t0=_fake_t0())

    @app.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    assert "tutor" in app._agent_registry
    spec = app._agent_registry["tutor"]
    assert spec.name == "tutor"
    # The "agent" fallback moved to AgentConfig (M3: model XOR
    # function_name; neither → "agent") — the spec records what the
    # caller actually passed.
    assert spec.function_name is None
    assert spec.system_prompt == "An English tutor."
    assert isinstance(spec.output_mode, ChatOutput)
    assert spec.agent is None  # lazy build — not constructed yet
    assert spec.value_transform is None  # not a list[BaseModel] return


def test_wrapper_call_returns_fluent_run(app_with_fake_agent: App) -> None:
    """``wrapper("msg")`` returns a :class:`FluentRun` instance."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    handle = tutor("How do I say hi formally in Japanese?")
    assert isinstance(handle, FluentRun)


async def test_await_wrapper_returns_unwrapped_value(
    app_with_fake_agent: App,
) -> None:
    """The Pythonic fast path: ``await wrapper("msg")`` returns ``T``."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    _FakeAgent.fake_value = "OK!"
    out = await tutor("hi")
    assert out == "OK!"


async def test_wrapper_underscore_agent_exposes_underlying_agent(
    app_with_fake_agent: App,
) -> None:
    """``wrapper._agent`` is ``None`` pre-build, populated post-build."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    assert tutor._agent is None
    await tutor("hi")  # triggers lazy build via _lookup_or_build
    assert tutor._agent is not None
    assert isinstance(tutor._agent, _FakeAgent)


def test_wrapper_underscore_spec_exposes_underlying_spec(
    app_with_fake_agent: App,
) -> None:
    """``wrapper._spec`` is the same ``_AgentSpec`` stored in the registry."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    spec = tutor._spec
    assert spec is app_with_fake_agent._agent_registry["tutor"]
    assert spec.name == "tutor"


async def test_app_build_constructs_every_registered_agent_once(
    app_with_fake_agent: App,
) -> None:
    """``await app.build()`` is idempotent; subsequent invokes reuse.

    Wraps the test body in ``async with app.lifespan():`` so M3's
    App-owned ``Client`` is connected at ``build()`` and cleanly
    disconnected on exit (otherwise the in-process FastMCP transport
    task lingers across tests and trips asyncio resource warnings).
    """

    @app_with_fake_agent.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app_with_fake_agent.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    assert _FakeAgent.construct_count == 0
    async with app_with_fake_agent.lifespan():
        assert _FakeAgent.construct_count == 2

        # Subsequent invokes do NOT construct again (cache on spec.agent).
        await a("hi")
        await b("hi")
        assert _FakeAgent.construct_count == 2

        # Calling build() again is also a no-op.
        await app_with_fake_agent.build()
        assert _FakeAgent.construct_count == 2


def test_duplicate_agent_name_raises_configuration_error() -> None:
    """Two ``@app.agent`` registrations with the same name fail eagerly."""
    app = App(t0=_fake_t0())

    @app.agent("dup")
    async def first(q: str) -> str:  # type: ignore[empty-body]
        """First."""

    with pytest.raises(ConfigurationError, match="Duplicate agent name 'dup'"):

        @app.agent("dup")
        async def _second(q: str) -> str:  # type: ignore[empty-body]  # pragma: no cover
            """Second."""


def test_forward_ref_failure_raises_at_decoration_time() -> None:
    """A return annotation referring to an undefined name raises eagerly.

    Mirrors ``test_output_inference.test_forward_ref_to_missing_name_raises_configuration_error``
    but exercises the *decorator* path: the failure must happen as soon
    as ``@app.agent`` is applied, not on first invocation.
    """
    app = App(t0=_fake_t0())

    with pytest.raises(ConfigurationError):

        @app.agent("broken")
        async def _broken(q: str) -> Missing:  # type: ignore[name-defined]  # noqa: F821
            """Broken."""


async def test_list_of_pydantic_unwraps_value_to_items_list(
    app_with_fake_agent: App,
) -> None:
    """``list[BaseModel]`` returns flow through ``value_transform``.

    The synthetic wrapper produced by M1's ``_wrap_list_model`` has a
    single ``items: list[T]`` field; the wrapper layer hands the user
    the inner list so the user-visible value matches ``list[T]``.
    """

    @app_with_fake_agent.agent("trips")
    async def trips(q: str) -> list[_Trip]:  # type: ignore[empty-body]
        """List of trips."""

    spec = app_with_fake_agent._agent_registry["trips"]
    assert spec.value_transform is not None  # M1 §4(d) honored

    # Build a synthetic wrapper instance like StructuredOutput would
    # produce; the value_transform must extract `.items` cleanly.
    sample = [
        _Trip(destinations=["Tokyo"], cost_usd=1500.0),
        _Trip(destinations=["Lisbon"], cost_usd=900.0),
    ]
    wrapper_cls = type(spec.output_mode).__name__  # sanity: it's StructuredOutput
    assert wrapper_cls == "StructuredOutput"

    # Drive the fake to return a synthetic wrapper instance.
    class _WrapperShape(BaseModel):
        items: list[_Trip]

    _FakeAgent.fake_value = _WrapperShape(items=sample)
    out = await trips("3 trips please")
    assert out == sample  # list[_Trip], not _WrapperShape


def test_assert_type_of_call_matches_return_annotation() -> None:
    """Mypy-strict reveal-type proxy via :func:`typing.assert_type`.

    ``assert_type`` is a runtime no-op (returns its first argument
    unchanged) but mypy verifies the type at static-check time.  This
    test runs cleanly; the real signal is ``uv run mypy``.
    """
    app = App(t0=_fake_t0())

    @app.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    # NOTE: mypy verifies ``tutor("hi")`` is ``FluentRun[str]``;
    # without the ``app.agent`` decorator the call would be
    # ``Coroutine[Any, Any, str]``.  Runtime: returns a FluentRun.
    handle = tutor("hi")
    assert_type(handle, FluentRun[str])
    assert isinstance(handle, FluentRun)


def test_call_rejects_kwargs_and_extra_positionals(
    app_with_fake_agent: App,
) -> None:
    """M3 wrappers accept exactly one positional ``str`` argument.

    `Depends(agent)` parameters are compiled into ``sub_agents`` at
    decoration time and have no runtime call-side representation, so
    kwargs are still rejected — but the error wording points the user
    at the right place.
    """

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    with pytest.raises(TypeError, match="exactly one positional argument"):
        tutor()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="exactly one positional argument"):
        tutor("hi", "bye")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match=r"Depends.*compiled into"):
        tutor("hi", mode="json")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="expects a str"):
        tutor(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# M3 — signature walk and sub-agents-via-Depends
# ---------------------------------------------------------------------------


async def test_subagent_via_depends_populates_sub_agent_specs(
    app_with_fake_agent: App,
) -> None:
    """Orchestrator declaring ``Depends(researcher_wrapper)`` records the spec."""
    from t0_agents.glow import Depends

    @app_with_fake_agent.agent("researcher")
    async def researcher(q: str) -> str:  # type: ignore[empty-body]
        """A researcher."""

    @app_with_fake_agent.agent("orchestrator")
    async def orchestrator(  # type: ignore[empty-body]
        q: str,
        research: Any = Depends(researcher),
    ) -> str:
        """An orchestrator."""

    spec = orchestrator._spec
    assert len(spec.sub_agent_specs) == 1
    assert spec.sub_agent_specs[0] is researcher._spec


async def test_subagent_specs_built_depth_first_before_orchestrator(
    app_with_fake_agent: App,
) -> None:
    """``_lookup_or_build`` constructs sub-agents BEFORE the orchestrator.

    AgentConfig's strict ``isinstance(sa, Agent)`` validation
    (`core/config.py:177`) rejects the test fake, so we cannot
    inspect the orchestrator's actual ``AgentConfig.sub_agents``.
    The integration test in commit 5 exercises that wiring against a
    real :class:`Agent`.  This unit test verifies the depth-first
    construction ordering via ``spec.agent`` population: when
    ``_lookup_or_build`` runs for the orchestrator, its sub-agent
    specs must have ``agent`` already populated.
    """
    from t0_agents.glow import Depends

    @app_with_fake_agent.agent("researcher")
    async def researcher(q: str) -> str:  # type: ignore[empty-body]
        """A researcher."""

    @app_with_fake_agent.agent("writer")
    async def writer(q: str) -> str:  # type: ignore[empty-body]
        """A writer."""

    @app_with_fake_agent.agent("orchestrator")
    async def orchestrator(  # type: ignore[empty-body]
        q: str,
        research: Any = Depends(researcher),
        write: Any = Depends(writer),
    ) -> str:
        """An orchestrator."""

    # Monkey-patch ``AgentConfig.__post_init__`` to skip the strict
    # ``isinstance(sa, Agent)`` validation for this unit test only —
    # the real :class:`Agent` path is exercised in commit 5.
    import t0_agents.core.config as config_mod

    captured_sub_agents: list[Any] = []
    original_post_init = config_mod.AgentConfig.__post_init__

    def patched_post_init(self: Any) -> None:
        if self.sub_agents:
            captured_sub_agents[:] = list(self.sub_agents)
            self.sub_agents = []
        original_post_init(self)

    config_mod.AgentConfig.__post_init__ = patched_post_init  # type: ignore[method-assign]
    try:
        async with app_with_fake_agent.lifespan():
            orch_spec = orchestrator._spec
            assert len(orch_spec.sub_agent_specs) == 2
            # Sub-agents constructed depth-first.
            for sub_spec in orch_spec.sub_agent_specs:
                assert sub_spec.agent is not None
            # And those constructed instances were what
            # ``_lookup_or_build`` threaded into ``AgentConfig.sub_agents``.
            assert {sa for sa in captured_sub_agents} == {
                sub_spec.agent for sub_spec in orch_spec.sub_agent_specs
            }
    finally:
        config_mod.AgentConfig.__post_init__ = original_post_init  # type: ignore[method-assign]


def test_subagent_param_not_in_runtime_call_signature(
    app_with_fake_agent: App,
) -> None:
    """The wrapper still accepts exactly one positional argument."""
    from t0_agents.glow import Depends

    @app_with_fake_agent.agent("researcher")
    async def researcher(q: str) -> str:  # type: ignore[empty-body]
        """A researcher."""

    @app_with_fake_agent.agent("orchestrator")
    async def orchestrator(  # type: ignore[empty-body]
        q: str,
        research: Any = Depends(researcher),
    ) -> str:
        """An orchestrator."""

    # Passing the sub-agent param as a kwarg is rejected — sub-agents
    # are compiled into ``AgentConfig.sub_agents`` at decoration time
    # and have no runtime call-side slot.
    with pytest.raises(TypeError, match=r"Depends.*compiled into"):
        orchestrator("hi", research=researcher)


def test_depends_on_unregistered_wrapper_raises_eagerly() -> None:
    """Using ``Depends(wrapper)`` from a different App raises at decoration."""
    from t0_agents.glow import Depends

    app_a = App(t0=_fake_t0())
    app_b = App(t0=_fake_t0())

    @app_a.agent("from_a")
    async def from_a(q: str) -> str:  # type: ignore[empty-body]
        """Registered on app_a only."""

    with pytest.raises(ConfigurationError, match="not registered on this App"):

        @app_b.agent("orchestrator")
        async def _orch(  # type: ignore[empty-body]  # pragma: no cover
            q: str,
            other: Any = Depends(from_a),
        ) -> str:
            """Tries to depend on an agent from a different App."""


def test_value_style_depends_on_agent_raises_with_pointer_to_tool() -> None:
    """``Depends(get_db)`` on ``@app.agent`` raises with a pointer to ``@app.tool``."""
    from t0_agents.glow import Depends

    app = App(t0=_fake_t0())

    def get_db() -> str:
        return "db"

    with pytest.raises(
        ConfigurationError, match=r"value-style provider.*@app.tool"
    ):

        @app.agent("orchestrator")
        async def _orch(  # type: ignore[empty-body]  # pragma: no cover
            q: str,
            db: str = Depends(get_db),
        ) -> str:
            """Tries value-style Depends on @app.agent."""


def test_first_parameter_typed_non_str_raises() -> None:
    """``@app.agent`` requires the first positional parameter to be ``str``."""
    app = App(t0=_fake_t0())

    with pytest.raises(ConfigurationError, match="must be annotated as ``str``"):

        @app.agent("orchestrator")
        async def _orch(q: int) -> str:  # type: ignore[empty-body]  # pragma: no cover
            """Wrong annotation on the message param."""


def test_first_parameter_annotated_str_is_accepted() -> None:
    """``Annotated[str, <meta>]`` on the message param is accepted.

    Forward-compat with M4's guardrail-on-message pattern (plan
    Example 9): ``Annotated[str, Guardrail.short]`` should land cleanly
    without rewriting the signature walk.  The Annotated wrapper is
    peeled to ``str`` and accepted; the metadata payload is ignored
    at M3 (M4 will extract guardrails from it).
    """
    app = App(t0=_fake_t0())

    # No exception means the decorator accepted the annotation.
    @app.agent("orchestrator")
    async def orch(q: Annotated[str, "short"]) -> str:  # type: ignore[empty-body]
        """Accepts an Annotated[str, ...] message."""

    assert "orchestrator" in app._agent_registry


def test_non_message_param_without_default_raises() -> None:
    """A non-message parameter without ``Depends`` default raises."""
    app = App(t0=_fake_t0())

    with pytest.raises(ConfigurationError, match="has no default"):

        @app.agent("orchestrator")
        async def _orch(  # type: ignore[empty-body]  # pragma: no cover
            q: str,
            extra: int,
        ) -> str:
            """Extra positional with no default."""


def test_non_message_param_with_plain_default_raises() -> None:
    """A non-message parameter whose default is NOT ``Depends`` raises."""
    app = App(t0=_fake_t0())

    with pytest.raises(ConfigurationError, match="non-Depends default"):

        @app.agent("orchestrator")
        async def _orch(  # type: ignore[empty-body]  # pragma: no cover
            q: str,
            extra: int = 7,
        ) -> str:
            """Extra positional with a plain default."""


# ---------------------------------------------------------------------------
# M2: AgentConfig knob passthrough + App-level defaults + .agent accessor
# ---------------------------------------------------------------------------


async def test_config_knobs_reach_agentconfig(app_with_fake_agent: App) -> None:
    """Every ``@app.agent`` knob lands on the constructed ``AgentConfig``."""

    @app_with_fake_agent.agent(
        "cs",
        max_turns=40,
        tool_timeout=60.0,
        parallel_tools=True,
        dryrun=True,
        tags={"team": "cs"},
        namespace="prod",
        memory_strategy="prepended_message",
    )
    async def cs(q: str) -> str:  # type: ignore[empty-body]
        """A CS agent."""

    await cs("hi")  # trigger lazy build
    cfg = cs.agent._config
    assert cfg.max_turns == 40
    assert cfg.tool_timeout == 60.0
    assert cfg.parallel_tools is True
    assert cfg.dryrun is True
    assert cfg.tags == {"team": "cs"}
    assert cfg.namespace == "prod"
    assert cfg.memory_strategy == "prepended_message"


async def test_omitted_knobs_keep_agentconfig_defaults(
    app_with_fake_agent: App,
) -> None:
    """Knobs the caller omits keep their ``AgentConfig`` defaults."""

    @app_with_fake_agent.agent("plain")
    async def plain(q: str) -> str:  # type: ignore[empty-body]
        """A plain agent."""

    await plain("hi")
    cfg = plain.agent._config
    assert cfg.max_turns == 20  # AgentConfig default, not overridden
    assert cfg.tool_timeout == 30.0
    assert cfg.parallel_tools is False
    assert cfg.tags is None
    assert cfg.namespace is None


async def test_app_level_defaults_and_agent_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App-level defaults apply; agent-level overrides win; tags dict-merge."""
    _FakeAgent.construct_count = 0
    _FakeAgent.last_invoke_kwargs = {}
    _FakeAgent.fake_value = "hello-from-fake"
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    app = App(t0=_fake_t0(), tags={"env": "prod"}, namespace="ns_app")

    @app.agent("inherits")
    async def inherits(q: str) -> str:  # type: ignore[empty-body]
        """Inherits App defaults."""

    @app.agent("overrides", namespace="ns_agent", tags={"env": "dev", "team": "cs"})
    async def overrides(q: str) -> str:  # type: ignore[empty-body]
        """Overrides App defaults."""

    await inherits("hi")
    await overrides("hi")

    inherits_cfg = inherits.agent._config
    assert inherits_cfg.namespace == "ns_app"
    assert inherits_cfg.tags == {"env": "prod"}

    overrides_cfg = overrides.agent._config
    assert overrides_cfg.namespace == "ns_agent"  # agent-level wins
    # tags dict-merge; the agent-level key ("env") wins over the App default.
    assert overrides_cfg.tags == {"env": "dev", "team": "cs"}


async def test_explicit_agent_tags_none_opts_out_of_app_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``@app.agent(tags=None)`` overrides the App default to None.

    Regression: an explicit ``None`` must stay distinct from omission — it
    must not be silently replaced by the App-level default tags.
    """
    _FakeAgent.construct_count = 0
    _FakeAgent.last_invoke_kwargs = {}
    _FakeAgent.fake_value = "hello-from-fake"
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    app = App(t0=_fake_t0(), tags={"env": "prod"})

    @app.agent("no_tags", tags=None)
    async def no_tags(q: str) -> str:  # type: ignore[empty-body]
        """Explicitly opts out of the App-level tags."""

    await no_tags("hi")
    assert no_tags.agent._config.tags is None  # explicit None, not the App default


async def test_per_call_with_reaches_invoke(app_with_fake_agent: App) -> None:
    """``.with_(episode_id=, prior_messages=, tags=)`` reaches ``Agent.invoke``."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    await tutor("turn 2").with_(
        episode_id="ep_multi",
        prior_messages=[{"role": "user", "content": "turn 1"}],
        tags={"customer_id": "c1"},
    )

    kw = _FakeAgent.last_invoke_kwargs
    assert kw["episode_id"] == "ep_multi"
    assert kw["prior_messages"] == [{"role": "user", "content": "turn 1"}]
    assert kw["tags"] == {"customer_id": "c1"}


async def test_system_arguments_suppresses_docstring_system(
    app_with_fake_agent: App,
) -> None:
    """``system_arguments`` drops the implicit docstring ``system``."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    await tutor("hi").with_(system_arguments={"level": "beginner"})

    kw = _FakeAgent.last_invoke_kwargs
    assert kw["system_arguments"] == {"level": "beginner"}
    assert "system" not in kw  # docstring suppressed (mutually exclusive)


async def test_system_arguments_with_explicit_system_raises(
    app_with_fake_agent: App,
) -> None:
    """Supplying both ``system_arguments`` and ``system`` raises."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        await tutor("hi").with_(system_arguments={"a": 1}, system="raw")


async def test_agent_accessor_pre_build_raises_post_build_returns(
    app_with_fake_agent: App,
) -> None:
    """``wrapper.agent`` raises before build, returns the Agent after."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """An English tutor."""

    assert tutor._agent is None  # deprecated alias stays None pre-build
    with pytest.raises(ConfigurationError, match="not built yet"):
        _ = tutor.agent

    await tutor("hi")  # trigger build
    assert isinstance(tutor.agent, _FakeAgent)
    assert tutor.agent is tutor._agent


# ---------------------------------------------------------------------------
# M3: model= functionless routing knobs
# ---------------------------------------------------------------------------


async def test_model_knobs_reach_agentconfig(app_with_fake_agent: App) -> None:
    """model/params/cache ride the config channel onto AgentConfig."""

    @app_with_fake_agent.agent(
        "tutor-m",
        model="groq::llama-3.3-70b-versatile",
        params={"chat_completion": {"temperature": 0.2}},
        cache={"enabled": "on", "max_age_s": 3600},
    )
    async def tutor_m(q: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    await tutor_m("hi")
    cfg = tutor_m.agent._config
    assert cfg.model == "groq::llama-3.3-70b-versatile"
    assert cfg.function_name is None
    assert cfg.params == {"chat_completion": {"temperature": 0.2}}
    assert cfg.cache == {"enabled": "on", "max_age_s": 3600}


async def test_variant_knob_reaches_agentconfig(app_with_fake_agent: App) -> None:
    """The prod path — function_name + variant + cache — stays intact."""

    @app_with_fake_agent.agent(
        "prod-a", function_name="prod_agent", variant="gpt4o_v2"
    )
    async def prod_a(q: str) -> str:  # type: ignore[empty-body]
        """Prod agent."""

    await prod_a("hi")
    cfg = prod_a.agent._config
    assert cfg.function_name == "prod_agent"
    assert cfg.variant == "gpt4o_v2"


async def test_default_function_name_resolves_to_agent(
    app_with_fake_agent: App,
) -> None:
    """Neither model= nor function_name= → AgentConfig's "agent" fallback
    (hello-world against a stock TOML keeps working)."""

    @app_with_fake_agent.agent("plain-m3")
    async def plain_m3(q: str) -> str:  # type: ignore[empty-body]
        """Plain."""

    await plain_m3("hi")
    cfg = plain_m3.agent._config
    assert cfg.function_name == "agent"
    assert cfg.model is None


async def test_model_and_function_name_raise_at_build(
    app_with_fake_agent: App,
) -> None:
    @app_with_fake_agent.agent("bad-xor", function_name="f", model="groq::x")
    async def bad_xor(q: str) -> str:  # type: ignore[empty-body]
        """Bad."""

    with pytest.raises(ConfigurationError, match="XOR"):
        await bad_xor("hi")


async def test_model_with_structured_return_raises_with_stanza(
    app_with_fake_agent: App,
) -> None:
    """A structured return annotation infers a JSON-function output mode,
    which the model= path cannot satisfy — the error carries the TOML fix."""

    @app_with_fake_agent.agent("planner-m3", model="groq::x")
    async def planner_m3(q: str) -> _Trip:  # type: ignore[empty-body]
        """Planner."""

    with pytest.raises(ConfigurationError) as excinfo:
        await planner_m3("hi")
    assert "[functions.planner-m3]" in str(excinfo.value)
    assert 'type = "json"' in str(excinfo.value)
