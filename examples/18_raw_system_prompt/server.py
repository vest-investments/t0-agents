"""FastMCP server for the raw-system-prompt demo.

The single ``echo`` tool exists because chat-mode ``Agent`` requires a
non-None ``tool_provider`` (see CHANGELOG v0.12.0).  The demo's
personalities (haiku / pirate / lawyer) are written so the model
answers in plain text without ever invoking it — which is itself part
of the point: the runtime ``system="..."`` prompt steers the model
strongly enough that the noop tool is never reached.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("RawSystemPromptDemo")


@mcp.tool
def echo(text: str) -> str:
    """Return the input text unchanged.

    Noop — only here so ``Agent`` (chat mode) has a tool_provider.
    """
    return text
