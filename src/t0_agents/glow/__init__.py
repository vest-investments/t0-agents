"""t0_agents.glow — the Pythonic skin over t0-agents v1.0.

EXPERIMENTAL. Public surface is unstable. This sub-package adds a
FastAPI-style decorator surface, type-hint-driven OutputMode inference,
fluent invocation handles, dependency injection, native-collection
composition, and typed MCP wiring on top of the v1.0 primitives.

Nothing in t0-agents core is modified by this layer — every `@app.agent`
compiles to a normal :class:`~t0_agents.Agent` and can be inspected via
the public ``._agent`` attribute on the returned callable.
"""

from __future__ import annotations

from t0_agents.glow.app import App
from t0_agents.glow.di import Depends
from t0_agents.glow.guardrails_typing import regex_guardrail
from t0_agents.glow.handle import FluentRun
from t0_agents.glow.mcp_typing import MCPServer
from t0_agents.glow.selectors import (
    Done,
    Error,
    Selector,
    Text,
    Thought,
    ToolCall,
    ToolResult,
    TurnEnd,
)

__all__ = [
    "App",
    "Depends",
    "Done",
    "Error",
    "FluentRun",
    "MCPServer",
    "Selector",
    "Text",
    "Thought",
    "ToolCall",
    "ToolResult",
    "TurnEnd",
    "regex_guardrail",
]
