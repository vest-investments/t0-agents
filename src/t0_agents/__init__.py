"""t0-agents: The Pythonic agentic framework for TensorZero."""

import logging

from tensorzero import ToolChoice

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import (
    ConfigurationError,
    MaxTurnsExceededError,
    T0AgentsError,
)
from t0_agents.core.types import AgentResult, MemoryProvider, ToolProvider
from t0_agents.mcp import run_agent

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "ConfigurationError",
    "MaxTurnsExceededError",
    "MemoryProvider",
    "T0AgentsError",
    "ToolChoice",
    "ToolProvider",
    "run_agent",
]

# Library best practice: prevent "No handler found" warnings.
logging.getLogger("t0_agents").addHandler(logging.NullHandler())
