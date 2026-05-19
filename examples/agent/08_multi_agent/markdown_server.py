"""Toy markdown-formatting MCP server for the writer sub-agent.

The writer hits ``format_markdown`` to produce a final deliverable in a
consistent shape.  Report / bullets / summary are the three options so
the LLM has a non-trivial choice to make per delegation.
"""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

mcp = FastMCP("markdown-writer")

Structure = Literal["report", "bullets", "summary"]


@mcp.tool
def format_markdown(text: str, structure: Structure = "report") -> str:
    """Format ``text`` as markdown in the requested ``structure``.

    - ``report``: adds an H1 title, splits ``text`` into paragraphs on
      double-newlines, prefixes each paragraph block with a short
      heading.
    - ``bullets``: one markdown bullet per line of ``text``.
    - ``summary``: a one-paragraph blockquote.

    The function is intentionally mechanical — the delegation example
    cares about *who decides to call this tool*, not the formatting
    quality.
    """
    if structure == "bullets":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"- {line}" for line in lines)
    if structure == "summary":
        single_line = " ".join(text.split())
        return f"> {single_line}"
    # Default: report
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    body = "\n\n".join(f"## Finding\n\n{p}" for p in paragraphs) or text
    return f"# Report\n\n{body}"
