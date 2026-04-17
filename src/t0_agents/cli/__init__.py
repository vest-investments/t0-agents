"""Typer + Rich CLI for ``t0-agents``.

The ``[cli]`` extra adds three subcommands:

- ``t0-agents run <spec>`` — interactive REPL with streaming + Rich
  rendering + elicitation prompts.
- ``t0-agents serve <spec>`` — wraps :func:`t0_agents.serve.create_app`
  in uvicorn for a one-shot dev server.
- ``t0-agents info <spec>`` — pretty-prints the loaded
  :class:`~t0_agents.core.config.AgentConfig`.

Public symbols re-exported here are stable extension points; anything
else is internal.

Install with::

    pip install 't0-agents[cli]'
"""

from __future__ import annotations

try:
    import rich as _rich  # noqa: F401
    import typer as _typer  # noqa: F401
except ImportError as exc:  # pragma: no cover - defensive
    from t0_agents.core.exceptions import T0AgentsError

    raise T0AgentsError(
        "t0_agents.cli requires the 'cli' extra. "
        "Install with: pip install 't0-agents[cli]'"
    ) from exc

from t0_agents.cli.app import main
from t0_agents.cli.elicit_tui import (
    make_rich_elicitation_handler,
    rich_elicitation_handler,
)
from t0_agents.cli.loader import AgentFactory, load_agent_factory
from t0_agents.cli.tui import TUIRenderer, get_active_renderer

__all__ = [
    "AgentFactory",
    "TUIRenderer",
    "get_active_renderer",
    "load_agent_factory",
    "main",
    "make_rich_elicitation_handler",
    "rich_elicitation_handler",
]
