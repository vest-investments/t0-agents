from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    """Structural type for tool discovery and execution.

    Matches the slice of ``fastmcp.Client`` that the agent loop uses:

    - ``list_tools()`` — discover available tools at loop start.
    - ``call_tool(name, arguments, *, timeout=...)`` — execute a single tool.

    Any object implementing these two async methods satisfies the protocol,
    enabling testing with fakes and future non-FastMCP providers.
    """

    async def list_tools(self) -> list[Any]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any: ...


@dataclass
class AgentResult:
    """Result of an agent run."""

    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
