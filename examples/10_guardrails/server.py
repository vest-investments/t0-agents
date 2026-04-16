"""MCP server for the guardrails example.

Three agent-visible tools plus one intentionally-dangerous stub:

- ``read_calendar`` — always-safe; no guardrail intercepts it.
- ``send_email`` — gated by ``require_approval``.
- ``delete_task`` — gated by ``require_approval``; uses ``ctx.elicit()``
  server-side so the example also demonstrates how ``require_approval``
  works alongside existing elicitation patterns.
- ``drop_database`` — target of ``block_tools`` — the LLM may attempt to
  call it, the guardrail never lets it through.
"""

from __future__ import annotations

from fastmcp import Context, FastMCP

mcp = FastMCP("guardrails-demo")

_INBOX: list[dict[str, str]] = []
_TASKS: dict[str, str] = {
    "T-101": "Set up CI pipeline",
    "T-102": "Write unit tests",
    "T-103": "Update README",
}


@mcp.tool
def read_calendar() -> str:
    """Read today's calendar events (always safe to invoke)."""
    return "\n".join(
        [
            "09:00 - Standup",
            "11:00 - Design review",
            "14:00 - 1:1 with Alice",
        ]
    )


@mcp.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (gated by require_approval at the client)."""
    _INBOX.append({"to": to, "subject": subject, "body": body})
    return f"Email sent to {to!r} with subject {subject!r}"


@mcp.tool
async def delete_task(task_id: str, ctx: Context) -> str:
    """Delete a project task (also calls ctx.elicit server-side)."""
    if task_id not in _TASKS:
        return f"Task {task_id!r} not found"
    title = _TASKS[task_id]
    server_ack = await ctx.elicit(
        f"Server-side ack: really delete {task_id} ({title})?",
        response_type=None,
    )
    if server_ack.action != "accept":
        return f"Task {task_id!r} deletion declined at server"
    _TASKS.pop(task_id)
    return f"Deleted task {task_id!r} ({title})"


@mcp.tool
def drop_database() -> str:
    """Dangerous stub — should never actually run in this example."""
    return "dropped EVERYTHING, hope you had a backup"
