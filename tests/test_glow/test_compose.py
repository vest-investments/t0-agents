"""Unit tests for :func:`t0_agents.glow.compose.compose`.

Cover the native-type dispatch (list/tuple/set/frozenset), the default
merge reducer for set ensembles, error paths for invalid spec types and
node types, glow-wrapper unwrapping, and composer-in-composer nesting.

The dict branch is exercised by ``test_compose_dict.py`` (landed in
commit 3); this module asserts the placeholder ``NotImplementedError``
only.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar, cast

import pytest
from tensorzero import AsyncTensorZeroGateway

from t0_agents.compose.builtins import ParallelComposer, SequentialComposer
from t0_agents.compose.topology import Composer, ComposerError
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs.tool_call import ToolCallOutput
from t0_agents.core.run import Run
from t0_agents.core.types import AgentResult
from t0_agents.glow import App, compose
from t0_agents.glow.compose import _default_merge, _LLMRoutedComposer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_t0() -> AsyncTensorZeroGateway:
    """Opaque ``AsyncTensorZeroGateway`` sentinel — ``Agent.__init__`` only
    stores the reference, so a bare ``object()`` survives unit tests."""
    return cast(AsyncTensorZeroGateway, object())


class _FakeAgent:
    """Stub :class:`~t0_agents.core.agent.Agent` patched into
    ``glow.app.Agent`` so wrapper-unwrap tests can resolve to a real
    object satisfying the v1.0 composer factories.
    """

    construct_count: ClassVar[int] = 0

    def __init__(self, config: Any, *, t0: Any, tool_provider: Any = None) -> None:
        type(self).construct_count += 1
        self.config = config
        self._t0 = t0
        self._tool_provider = tool_provider

    def invoke(self, user_message: str, **kwargs: Any) -> Run[Any]:  # pragma: no cover
        # The compose() unit tests never invoke the underlying agent —
        # they only check the returned composer's shape.  Raise loudly if
        # a test slips through and tries.
        raise NotImplementedError("compose() unit tests must not invoke agents")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> App:
    """Provide an :class:`App` whose ``Agent`` constructor is stubbed."""
    _FakeAgent.construct_count = 0
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    return App(t0=_fake_t0())


# ---------------------------------------------------------------------------
# list / tuple → sequential
# ---------------------------------------------------------------------------


def test_compose_list_returns_sequential_composer(app: App) -> None:
    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    result = compose([a, b])
    assert isinstance(result, Composer)
    assert isinstance(result, SequentialComposer)
    assert result.name == "sequential"
    assert len(result.children) == 2


def test_compose_tuple_returns_sequential_composer(app: App) -> None:
    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    result = compose((a, b))
    assert isinstance(result, SequentialComposer)
    assert len(result.children) == 2


# ---------------------------------------------------------------------------
# set / frozenset → parallel(error_policy="continue")
# ---------------------------------------------------------------------------


def test_compose_set_returns_parallel_continue_policy(app: App) -> None:
    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    def my_merge(items: list[Any]) -> str:
        return "|".join(str(item) for item in items)

    result = compose({a, b}, merge=my_merge)
    assert isinstance(result, ParallelComposer)
    assert result.error_policy == "continue"
    assert result.reducer is my_merge
    assert len(result.children) == 2


def test_compose_frozenset_returns_parallel_continue_policy(app: App) -> None:
    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    result = compose(frozenset({a, b}))
    assert isinstance(result, ParallelComposer)
    assert result.error_policy == "continue"
    # No merge passed → reducer is _default_merge.
    assert result.reducer is _default_merge


# ---------------------------------------------------------------------------
# _default_merge — direct exercise
# ---------------------------------------------------------------------------


def _result(value: Any) -> AgentResult[Any]:
    return AgentResult(
        value=value, episode_id="ep_test", inference_ids=["if_test"], turns=1,
    )


def test_default_merge_concatenates_string_values() -> None:
    out = _default_merge([_result("hello"), _result("world")])
    assert out == "hello\n---\nworld"


def test_default_merge_skips_exceptions_and_concatenates_successes() -> None:
    out = _default_merge([_result("hi"), RuntimeError("oops"), _result("there")])
    assert out == "hi\n---\nthere"


def test_default_merge_mixed_types_raises_with_merge_hint() -> None:
    with pytest.raises(TypeError, match=r"explicit `merge=` reducer"):
        _default_merge([_result("hello"), _result({"a": 1})])


def test_default_merge_all_failures_raises_composer_error() -> None:
    with pytest.raises(ComposerError):
        _default_merge([RuntimeError("a"), ValueError("b")])


# ---------------------------------------------------------------------------
# Invalid spec / node types
# ---------------------------------------------------------------------------


def test_compose_invalid_spec_type_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="unsupported spec type"):
        compose(42)


def test_compose_list_with_non_composable_node_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="not a Composable"):
        compose([1, 2])


def test_compose_list_with_string_node_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="not a Composable"):
        compose(["string-not-an-agent"])


def test_compose_empty_dict_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="routes cannot be empty"):
        compose({})


def test_compose_dict_with_non_composable_value_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="not a Composable"):
        compose({"k": object()})


# ---------------------------------------------------------------------------
# Composer-in-composer nesting (v1.0 ``Composable`` protocol acceptance)
# ---------------------------------------------------------------------------


def test_compose_nested_composer_is_accepted_as_node(app: App) -> None:
    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    @app.agent("c")
    async def c(q: str) -> str:  # type: ignore[empty-body]
        """C."""

    inner = compose([a, b])
    outer = compose([inner, c])
    assert isinstance(outer, SequentialComposer)
    assert outer.children[0] is inner
    # ``inner`` is itself a SequentialComposer — proves the
    # ``Composable`` protocol carries through unchanged.
    assert isinstance(outer.children[0], SequentialComposer)


# ---------------------------------------------------------------------------
# Glow wrapper unwrap
# ---------------------------------------------------------------------------


def test_compose_unwraps_glow_wrapper_to_underlying_agent(app: App) -> None:
    """A wrapper is converted to the underlying :class:`Agent` before
    being passed to the v1.0 composer factory — the children tuple
    contains real Agent instances (the _FakeAgent stub), not wrappers."""

    @app.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    result = compose([a])
    assert isinstance(result, SequentialComposer)
    assert len(result.children) == 1
    # Build was triggered lazily by compose() unwrap.
    assert _FakeAgent.construct_count == 1
    assert isinstance(result.children[0], _FakeAgent)


# ---------------------------------------------------------------------------
# LLM-routed dict composer construction
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_config_aware_fake(monkeypatch: pytest.MonkeyPatch) -> App:
    """``App`` whose Agent constructor is the stub on BOTH the
    ``glow.app`` and ``glow.compose`` import sites.

    The dict-route classifier is built by
    ``glow.compose._make_router_agent`` (not by the App), so the patch
    has to land on the ``compose`` module's ``Agent`` symbol too — the
    integration test exercises a real ``ToolCallOutput`` Agent against a
    live gateway; the unit test only needs the construction shape.

    The double-patch goes through ``importlib.import_module`` because
    ``t0_agents.glow.__init__`` re-exports ``compose`` as a function,
    which shadows the submodule on attribute access from the parent
    package — monkeypatch's dotted-path resolver would otherwise land
    on the function instead of the module.
    """
    _FakeAgent.construct_count = 0
    monkeypatch.setattr("t0_agents.glow.app.Agent", _FakeAgent)
    compose_module = importlib.import_module("t0_agents.glow.compose")
    monkeypatch.setattr(compose_module, "Agent", _FakeAgent)
    return App(t0=_fake_t0())


def test_compose_dict_returns_llm_routed_composer(
    app_with_config_aware_fake: App,
) -> None:
    @app_with_config_aware_fake.agent("math")
    async def math_route(q: str) -> str:  # type: ignore[empty-body]
        """Math route."""

    @app_with_config_aware_fake.agent("code")
    async def code_route(q: str) -> str:  # type: ignore[empty-body]
        """Code route."""

    result = compose({"math": math_route, "code": code_route})
    assert isinstance(result, _LLMRoutedComposer)
    assert set(result.routes) == {"math", "code"}
    # Classifier inherits the leaf Agent's function_name (the M2/M3
    # default ``agent`` chat function in glow).  The classifier itself
    # is built per-invocation (concurrency-safety) — the spec/config
    # is what the composer stores at construction time.
    assert result.classifier_config.function_name == "agent"
    # Classifier uses ToolCallOutput with tool_choice="required".
    assert isinstance(result.classifier_config.output, ToolCallOutput)
    assert result.classifier_config.output.tool_choice == "required"


def test_compose_dict_no_reachable_agent_raises_configuration_error() -> None:
    """A composer-only route graph with no leaf ``Agent`` cannot supply
    ``t0`` / ``function_name`` to the synthetic classifier; we surface
    that mismatch at compose() time, not on first invocation."""

    class _BareComposer(Composer):
        """Composer with no children that satisfies neither Agent nor
        the v1.0 builtin shapes — exercises the no-leaf path."""

        name: str = "bare"

        @property
        def children(self) -> tuple[Composer, ...]:
            return ()

    with pytest.raises(ConfigurationError, match="no Agent is reachable"):
        compose({"k": _BareComposer()})


def test_compose_dict_unwraps_glow_wrappers(
    app_with_config_aware_fake: App,
) -> None:
    @app_with_config_aware_fake.agent("a")
    async def a(q: str) -> str:  # type: ignore[empty-body]
        """A."""

    @app_with_config_aware_fake.agent("b")
    async def b(q: str) -> str:  # type: ignore[empty-body]
        """B."""

    result = compose({"a": a, "b": b})
    assert isinstance(result, _LLMRoutedComposer)
    # Routes are unwrapped Agent (the fake stub), not wrappers.
    assert all(
        isinstance(route, _FakeAgent) for route in result.routes.values()
    )

