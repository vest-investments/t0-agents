"""t0-agents: The Pythonic agentic framework for TensorZero."""

import logging

from tensorzero import ToolChoice

__version__ = "2.6.0"

from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.conversation import ConversationUsage
from t0_agents.core.elicitation_forwarder import (
    ElicitationForwarder,
    ElicitationRequest,
)
from t0_agents.core.exceptions import (
    ConfigurationError,
    GuardrailBlockedError,
    MaxTurnsExceededError,
    MemoryHistoryUnavailableError,
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
from t0_agents.delegation import SubAgentToolProvider
from t0_agents.mcp import run_agent
from t0_agents.stream import (
    CODEC_VERSION,
    AgentEvent,
    DoneEvent,
    ElicitationRequestedEvent,
    ElicitationResolvedEvent,
    ErrorEvent,
    RunErrorEvent,
    Status,
    StreamingAgentLoop,
    TextDeltaEvent,
    TextEndEvent,
    ThoughtDeltaEvent,
    ThoughtEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolProgressEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStreamAssembler,
    event_from_dict,
    event_to_dict,
    split_tool_identity,
    synthesize_statuses,
)
from t0_agents.tooling import FilteredToolProvider

__all__ = [
    "CODEC_VERSION",
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "AgentResult",
    "Attempt",
    "ChatOutput",
    "ConfigurationError",
    "ConversationUsage",
    "CostTrackingMiddleware",
    "DoneEvent",
    "ElicitationForwarder",
    "ElicitationRequest",
    "ElicitationRequestedEvent",
    "ElicitationResolvedEvent",
    "ErrorEvent",
    "FilteredToolProvider",
    "Guardrail",
    "GuardrailBlockedError",
    "GuardrailLogEntry",
    "GuardrailsMiddleware",
    "InferenceRequest",
    "JsonOutput",
    "JsonRetryMiddleware",
    "MaxTurnsExceededError",
    "MemoryHistoryUnavailableError",
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
    "RunErrorEvent",
    "RunFailedError",
    "RunInput",
    "RunState",
    "RunUsage",
    "Status",
    "StreamingAgentLoop",
    "StreamingOutput",
    "StructuredInferenceResult",
    "StructuredOutput",
    "SubAgentError",
    "SubAgentMiddleware",
    "SubAgentToolProvider",
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
    "ToolProgressEvent",
    "ToolProvider",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStreamAssembler",
    "__version__",
    "event_from_dict",
    "event_to_dict",
    "permissive_parse",
    "run_agent",
    "split_tool_identity",
    "synthesize_statuses",
]

# Library best practice: prevent "No handler found" warnings.
logging.getLogger("t0_agents").addHandler(logging.NullHandler())
