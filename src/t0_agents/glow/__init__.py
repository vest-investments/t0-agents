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

from t0_agents.core.agent import AGENT_TAG
from t0_agents.core.conversation import ConversationIndex, ConversationSummary
from t0_agents.glow.agent_handle import AgentHandle, llm
from t0_agents.glow.app import App
from t0_agents.glow.conversation import Conversation
from t0_agents.glow.di import Depends
from t0_agents.glow.guardrails_typing import regex_guardrail
from t0_agents.glow.handle import FluentRun
from t0_agents.glow.mcp_typing import MCPServer
from t0_agents.glow.selectors import (
    Done,
    ElicitationRequested,
    ElicitationResolved,
    Error,
    RunError,
    Selector,
    Text,
    TextEnd,
    Thought,
    ToolCall,
    ToolProgress,
    ToolResult,
    TurnEnd,
)
from t0_agents.mcp.runtime import ServerCredentials
from t0_agents.usage import UsageLedger

__all__ = [
    "AGENT_TAG",
    "AgentHandle",
    "App",
    "Conversation",
    "ConversationIndex",
    "ConversationSummary",
    "Depends",
    "Done",
    "ElicitationRequested",
    "ElicitationResolved",
    "Error",
    "FluentRun",
    "MCPServer",
    "RunError",
    "Selector",
    "ServerCredentials",
    "Text",
    "TextEnd",
    "Thought",
    "ToolCall",
    "ToolProgress",
    "ToolResult",
    "TurnEnd",
    "UsageLedger",
    "llm",
    "regex_guardrail",
]
