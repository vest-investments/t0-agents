"""t0-agents: The Pythonic agentic framework for TensorZero."""

import logging

from tensorzero import ToolChoice

from t0_agents.compose import (
    Composable,
    Composer,
    ComposerError,
    circuit_breaker,
    conditional,
    consensus,
    map_reduce,
    parallel,
    race,
    retry_chain,
    saga,
    sequential,
)
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
    RunCancelledError,
    RunFailedError,
    SubAgentError,
    T0AgentsError,
    T0AgentsFatalError,
)
from t0_agents.core.middleware import (
    CostTrackingMiddleware,
    GuardrailsMiddleware,
    JsonRetryMiddleware,
    MemoryMiddleware,
    Middleware,
    ObservabilityMiddleware,
    RunContext,
    SubAgentMiddleware,
)
from t0_agents.core.output import (
    Attempt,
    InferenceRequest,
    OutputDiagnostics,
    OutputMode,
    ParseFailure,
    Recovery,
)
from t0_agents.core.outputs import (
    ChatOutput,
    JsonOutput,
    StreamingOutput,
    StructuredOutput,
    ToolCallOutput,
)
from t0_agents.core.run import Run, RunInput, RunState
from t0_agents.core.structured import StructuredInferenceResult, permissive_parse
from t0_agents.core.types import (
    AgentResult,
    Guardrail,
    GuardrailLogEntry,
    MemoryProvider,
    RunUsage,
    ToolProvider,
)
from t0_agents.delegation import agent_as_tool
from t0_agents.mcp import run_agent
from t0_agents.stream import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    StreamingAgentLoop,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStreamAssembler,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "AgentResult",
    "Attempt",
    "ChatOutput",
    "Composable",
    "Composer",
    "ComposerError",
    "ConfigurationError",
    "CostTrackingMiddleware",
    "DoneEvent",
    "ElicitationForwarder",
    "ElicitationRequest",
    "ErrorEvent",
    "Guardrail",
    "GuardrailBlockedError",
    "GuardrailLogEntry",
    "GuardrailsMiddleware",
    "InferenceRequest",
    "JsonOutput",
    "JsonRetryMiddleware",
    "MaxTurnsExceededError",
    "MemoryMiddleware",
    "MemoryProvider",
    "Middleware",
    "ObservabilityMiddleware",
    "OutputDiagnostics",
    "OutputMode",
    "ParseFailure",
    "Recovery",
    "Run",
    "RunCancelledError",
    "RunContext",
    "RunFailedError",
    "RunInput",
    "RunState",
    "RunUsage",
    "StreamingAgentLoop",
    "StreamingOutput",
    "StructuredInferenceResult",
    "StructuredOutput",
    "SubAgentError",
    "SubAgentMiddleware",
    "T0AgentsError",
    "T0AgentsFatalError",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThoughtDeltaEvent",
    "ThoughtEvent",
    "ToolCallDeltaEvent",
    "ToolCallEvent",
    "ToolCallOutput",
    "ToolChoice",
    "ToolProvider",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStreamAssembler",
    "agent_as_tool",
    "circuit_breaker",
    "conditional",
    "consensus",
    "map_reduce",
    "parallel",
    "permissive_parse",
    "race",
    "retry_chain",
    "run_agent",
    "saga",
    "sequential",
]

# Library best practice: prevent "No handler found" warnings.
logging.getLogger("t0_agents").addHandler(logging.NullHandler())
