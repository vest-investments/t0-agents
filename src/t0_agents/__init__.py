"""t0-agents: The Pythonic agentic framework for TensorZero."""

import logging

from tensorzero import ToolChoice

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    ElicitationRequest,
)
from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
    MaxTurnsExceededError,
    T0AgentsError,
    T0AgentsFatalError,
)
from t0_agents.core.types import (
    AgentResult,
    Guardrail,
    GuardrailLogEntry,
    MemoryProvider,
    ToolProvider,
)
from t0_agents.mcp import run_agent

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ConfigurationError",
    "ElicitationForwarder",
    "ElicitationRequest",
    "Guardrail",
    "GuardrailBlockedError",
    "GuardrailLogEntry",
    "MaxTurnsExceededError",
    "MemoryProvider",
    "T0AgentsError",
    "T0AgentsFatalError",
    "ToolChoice",
    "ToolProvider",
    "run_agent",
]

# Library best practice: prevent "No handler found" warnings.
logging.getLogger("t0_agents").addHandler(logging.NullHandler())
