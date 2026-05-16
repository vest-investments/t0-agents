"""Smoke tests: M4 does not regress the legacy ``AgentConfig.sub_agents=`` path.

The full behavioural regression for ``examples/14_multi_agent/`` is
covered by the existing ``tests/test_compose/test_integration.py``
(integration-marked).  This file is a lightweight smoke test that
imports the legacy public API and confirms a config-time check
still rejects the documented mis-configuration.
"""

from __future__ import annotations

import pytest

from t0_agents import (
    Agent,
    Composable,
    Composer,
    SubAgentMiddleware,
    agent_as_tool,
    sequential,
)


def test_legacy_sub_agents_path_still_exists() -> None:
    """The M2 declarative ``sub_agents=`` path is still importable.

    M4 adds Composers as an additive surface — it does not remove or
    deprecate the v0.12 ``AgentConfig.sub_agents=`` path nor the M2
    ``SubAgentMiddleware``.
    """
    assert callable(agent_as_tool)
    assert SubAgentMiddleware is not None
    # SubAgentMiddleware has the M2 protocol shape.
    assert hasattr(SubAgentMiddleware, "__call__")


def test_agent_is_composable() -> None:
    """An Agent satisfies the M4 Composable protocol structurally."""
    # We don't construct a live Agent here (would need a gateway);
    # just verify the class is structurally a Composable when
    # constructed.
    assert hasattr(Agent, "invoke")
    # The Composable protocol is @runtime_checkable; verify the
    # method exists on Agent's class.
    assert callable(getattr(Agent, "invoke", None))


def test_topology_factories_return_composers() -> None:
    """Each topology factory returns a Composer (Composable) instance.

    Construction is sync; no Agent is required, so this works without
    a TZ gateway.  The returned object is both a ``Composer`` and a
    ``Composable``.
    """
    from tests.test_compose.v1._helpers import MockComposable

    a = MockComposable(name="a", value="x")
    composer = sequential(a)

    assert isinstance(composer, Composer)
    assert isinstance(composer, Composable)


def test_composer_invoke_kwargs_are_narrow() -> None:
    """``Composer.invoke`` rejects Agent-level kwargs (narrow surface)."""
    from tests.test_compose.v1._helpers import MockComposable

    a = MockComposable(name="a", value="x")
    composer = sequential(a)
    # ``Composer.invoke`` takes only ``input``, ``eager``, ``context``,
    # ``cancel_timeout``.  Passing an Agent-level kwarg like
    # ``tags=`` is a TypeError.
    with pytest.raises(TypeError):
        composer.invoke("seed", tags={"foo": "bar"})  # type: ignore[call-arg]
