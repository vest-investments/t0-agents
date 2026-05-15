"""Shared fake for ``fastmcp.Client`` in unit tests.

Hand-written stub rather than ``unittest.mock`` so the suite stays
mypy-strict-clean. ``run_agent`` only calls ``list_tools()`` and
``call_tool()`` on the client, so the duck-typed ``FakeMCPClient``
satisfies the runtime contract without subclassing the real Client.

Extracted from ``tests/test_mcp/test_loop.py`` so that multiple test
modules can share a single source of truth for the MCP client mock
contract.
"""

from __future__ import annotations

from typing import Any

from fastmcp.client.client import CallToolResult
from mcp.types import Tool as MCPTool


class FakeMCPClient:
    """Stand-in for ``fastmcp.Client``.

    ``call_results`` maps tool name -> either a ``CallToolResult`` (returned) or
    an ``Exception`` (raised). For tests that need to script per-call ordering
    independently, use ``call_outcomes`` (a list popped in order).
    """

    def __init__(
        self,
        *,
        tools: list[MCPTool] | None = None,
        call_results: dict[str, CallToolResult | Exception] | None = None,
        call_outcomes: list[CallToolResult | Exception] | None = None,
        side_effect: Any = None,
    ) -> None:
        # ``tools is None`` falls back to a default one-tool list for
        # tests that don't care about advertisement; an explicit
        # ``tools=[]`` must be honoured so safety checks (e.g. the
        # ``json_mode='tool'`` conflict guard) can verify the
        # empty-provider path.
        self._tools = (
            tools
            if tools is not None
            else [
                MCPTool(
                    name="add",
                    description="add tool",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )
        self._call_results = call_results or {}
        self._call_outcomes = list(call_outcomes) if call_outcomes else None
        self._side_effect = side_effect
        self.call_tool_invocations: list[
            tuple[str | None, dict[str, Any] | None, float | None]
        ] = []

    async def list_tools(self) -> list[MCPTool]:
        return self._tools

    async def call_tool(
        self,
        name: str | None,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        **_: Any,
    ) -> CallToolResult:
        self.call_tool_invocations.append((name, arguments, timeout))
        if self._side_effect is not None:
            await self._side_effect(name)
        if self._call_outcomes is not None:
            outcome: CallToolResult | Exception = self._call_outcomes.pop(0)
        else:
            key = name or ""
            if key not in self._call_results:
                raise AssertionError(f"FakeMCPClient: no canned result for {name!r}")
            outcome = self._call_results[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
