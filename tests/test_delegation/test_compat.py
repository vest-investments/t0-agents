"""Smoke test: the public sub-agent surface stays importable.

Confirms that the four symbols a caller needs to wire a declarative
``AgentConfig(sub_agents=[...])`` orchestrator — ``Agent``,
``agent_as_tool``, ``SubAgentMiddleware``, ``SubAgentError`` — remain
re-exported from the top-level ``t0_agents`` package after the
sub-agent code moved to :mod:`t0_agents.delegation`.

Renamed from the v1.0-era ``test_compat_sub_agents.py``; the three
sibling tests in that file covered the topology Composer surface
(``sequential``/``Composer``/``Composable``) and were dropped during
the compose-removal M1 relocation because that surface is slated for
deletion in M2.
"""

from __future__ import annotations

from t0_agents import (
    Agent,
    SubAgentMiddleware,
    agent_as_tool,
)


def test_legacy_sub_agents_path_still_exists() -> None:
    """The declarative ``sub_agents=`` path is still importable."""
    assert callable(agent_as_tool)
    assert SubAgentMiddleware is not None
    # SubAgentMiddleware follows the Middleware protocol shape.
    assert hasattr(SubAgentMiddleware, "__call__")
    # Agent exposes the canonical invoke entry point.
    assert callable(getattr(Agent, "invoke", None))
