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
from typing import Any, ClassVar, assert_type, cast

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
    assert spec.function_name == "agent"
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
    """M2 wrappers accept exactly one positional ``str`` argument."""

    @app_with_fake_agent.agent("tutor")
    async def tutor(q: str) -> str:  # type: ignore[empty-body]
        """A tutor."""

    with pytest.raises(TypeError, match="exactly one positional argument"):
        tutor()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="exactly one positional argument"):
        tutor("hi", "bye")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="keyword arguments will be supported in M3"):
        tutor("hi", mode="json")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="expects a str"):
        tutor(123)  # type: ignore[arg-type]
