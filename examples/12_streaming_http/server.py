"""FastMCP server for the streaming HTTP demo.

Combines two flavours of tools so the browser client can exercise both
event kinds coming over the WS channel:

* ``add`` / ``get_weather`` — plain tools, no elicitation.  These emit
  ``tool_call`` / ``tool_result`` events but nothing HITL.
* ``delete_task`` / ``create_task`` — elicitation tools (copied from
  ``examples/07_elicitation/server.py``) that exercise the
  :class:`~t0_agents.core.ElicitationForwarder` round-trip over the WS.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp import Context, FastMCP

mcp = FastMCP("StreamingAgentDemo")

TASKS = {
    "T-101": {"title": "Ship streaming docs", "priority": "high", "assignee": "Alice"},
    "T-102": {"title": "Rewrite onboarding", "priority": "medium", "assignee": "Bob"},
    "T-103": {"title": "Clean old caches", "priority": "low", "assignee": "Alice"},
}


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data)."""
    return f"The weather in {city} is 22°C and sunny."


@mcp.tool
def list_tasks() -> str:
    """List the current project tasks."""
    lines = []
    for task_id, info in TASKS.items():
        lines.append(
            f"{task_id}: {info['title']} "
            f"(priority: {info['priority']}, assignee: {info['assignee']})"
        )
    return "\n".join(lines)


@mcp.tool
async def delete_task(task_id: str, ctx: Context) -> str:
    """Delete a task by ID. Call this tool directly — the client handles the
    approval UI automatically. Do NOT ask the user to confirm in chat first;
    the approval dialog is built into this tool.
    """
    if task_id not in TASKS:
        return f"Task {task_id!r} not found"
    task = TASKS[task_id]
    result = await ctx.elicit(
        f"Delete task {task_id} '{task['title']}'?",
        response_type=None,
    )
    if result.action == "accept":
        return f"Deleted task: {task_id} ({task['title']})"
    return f"Deletion cancelled for task: {task_id}"


@dataclass
class TaskInput:
    """Structured input for creating a task (flat primitive fields only)."""

    title: str
    priority: str
    assignee: str


@mcp.tool
async def create_task(ctx: Context) -> str:
    """Create a new task. Call this tool directly — it collects title,
    priority, and assignee from the user through a built-in form dialog.
    Do NOT ask the user for those details in chat first; this tool opens
    the form itself.
    """
    result = await ctx.elicit(
        "Provide the new task details",
        response_type=TaskInput,
    )
    if result.action == "accept":
        data = result.data
        return (
            f"Created task: '{data.title}' "
            f"(priority: {data.priority}, assignee: {data.assignee})"
        )
    return "Task creation cancelled"
