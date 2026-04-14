"""t0-agents: The Pythonic agentic framework for TensorZero."""

import logging

from t0_agents.core.exceptions import MaxTurnsExceededError
from t0_agents.core.types import AgentResult
from t0_agents.mcp import run_agent

__all__ = ["AgentResult", "MaxTurnsExceededError", "run_agent"]

# Library best practice: prevent "No handler found" warnings.
logging.getLogger("t0_agents").addHandler(logging.NullHandler())
