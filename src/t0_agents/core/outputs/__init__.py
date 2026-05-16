"""Built-in :class:`~t0_agents.core.output.OutputMode` implementations.

Five modes ship in v1.0:

- :class:`ChatOutput` — boundary assistant text from the MCP tool loop.
- :class:`JsonOutput` — single-shot JSON inference with permissive parsing
  and bounded retry-with-hint.
- :class:`StructuredOutput` — Pydantic-validated JSON.  Lazy import of
  ``pydantic``; install via ``pip install 't0-agents[pydantic]'``.
- :class:`ToolCallOutput` — router pattern: extract tool calls from a
  chat-style response without executing them.
- :class:`StreamingOutput` — placeholder forwarding to ``base``.  Only
  ``StreamingOutput(base=ChatOutput())`` is supported in v1.0 (TZ
  upstream does not yet emit ``output.parsed`` mid-stream — open
  question Q11).

Every mode satisfies the :class:`~t0_agents.core.output.OutputMode`
protocol; users can plug in custom modes without subclassing.
"""

from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.streaming import StreamingOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.core.outputs.tool_call import ToolCallOutput

__all__ = [
    "ChatOutput",
    "JsonOutput",
    "StreamingOutput",
    "StructuredOutput",
    "ToolCallOutput",
]
