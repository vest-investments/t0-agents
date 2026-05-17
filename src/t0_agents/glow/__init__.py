"""t0_agents.glow — the Pythonic skin over t0-agents v1.0.

EXPERIMENTAL. Public surface is unstable. This sub-package adds a
FastAPI-style decorator surface, type-hint-driven OutputMode inference,
fluent invocation handles, dependency injection, native-collection
composition, and typed MCP wiring on top of the v1.0 primitives.

Nothing in t0-agents core is modified by this layer — every `@app.agent`
compiles to a normal :class:`~t0_agents.Agent` and can be inspected via
the public ``._agent`` attribute on the returned callable.

The public surface will be populated incrementally as the
``experimental/glow`` branch grows.  See the plan at
``/home/makarenko/.claude/plans/la-idea-es-tener-glowing-gem.md`` for
the implementation roadmap.
"""

from __future__ import annotations

from t0_agents.glow.handle import FluentRun

__all__ = ["FluentRun"]
