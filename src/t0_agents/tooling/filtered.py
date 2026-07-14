"""Name-set tool filtering — :class:`FilteredToolProvider`.

Standalone and reusable: imports only from :mod:`t0_agents.core`, so it sits
below ``glow`` / ``mcp`` in the dependency graph (mirrors the ``mcp/``
standalone rule).  Glow's ``@app.agent(tools=[...])`` is the primary caller —
it resolves the heterogeneous spec (``@app.tool`` callables, ``MCPServer``
classes, glob strings) against the live tool list, then hands the resolved
names here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from t0_agents.core.exceptions import T0AgentsError
from t0_agents.core.types import ToolProvider

__all__ = ["FilteredToolProvider"]


class FilteredToolProvider:
    """A :class:`~t0_agents.core.types.ToolProvider` exposing only an
    allow-listed set of tool names.

    ``list_tools`` forwards to ``inner`` and drops every tool whose ``name``
    is not in ``allow``.  ``call_tool`` raises :class:`~t0_agents.core.exceptions.T0AgentsError`
    for a name outside the set — a listing-only filter would still let a
    forged tool call reach ``inner``, so refusing here makes the filter a real
    boundary (the "order-placement tools stay disabled" guarantee is
    safety-critical for the vest-ai migration).  An empty ``allow`` is a
    genuinely toolless provider.
    """

    def __init__(self, inner: ToolProvider, allow: Iterable[str]) -> None:
        self._inner = inner
        self._allow = frozenset(allow)

    async def list_tools(self) -> list[Any]:
        return [
            tool
            for tool in await self._inner.list_tools()
            if tool.name in self._allow
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        if name not in self._allow:
            raise T0AgentsError(
                f"Tool {name!r} is not permitted for this agent "
                f"(allowed: {sorted(self._allow)!r})"
            )
        return await self._inner.call_tool(name, arguments, timeout=timeout)
