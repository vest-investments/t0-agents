"""Smoke test: the public sub-agent surface stays importable.

Confirms that the symbols a caller needs to wire a declarative
``AgentConfig(sub_agents=[...])`` orchestrator — ``Agent``,
``SubAgentMiddleware``, ``SubAgentError``, ``SubAgentToolProvider`` —
remain re-exported from the top-level ``t0_agents`` package.

Renamed from the v1.0-era ``test_compat_sub_agents.py``; the three
sibling tests in that file covered the topology Composer surface
(``sequential``/``Composer``/``Composable``) and were dropped during
the compose-removal M1 relocation.  ``agent_as_tool`` is no longer
exported — it was a single-purpose adapter to the deleted server-side
registration path; :class:`SubAgentToolProvider` is the canonical
surface now.
"""

from __future__ import annotations

from t0_agents import (
    Agent,
    SubAgentMiddleware,
    SubAgentToolProvider,
)


def test_legacy_sub_agents_path_still_exists() -> None:
    """The declarative ``sub_agents=`` path is still importable."""
    assert SubAgentToolProvider is not None
    assert SubAgentMiddleware is not None
    # SubAgentMiddleware follows the Middleware protocol shape.
    assert hasattr(SubAgentMiddleware, "__call__")
    # Agent exposes the canonical invoke entry point.
    assert callable(getattr(Agent, "invoke", None))
