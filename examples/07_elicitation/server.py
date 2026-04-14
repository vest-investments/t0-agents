"""MCP server for the elicitation example (project task manager).

Three tools:
- list_tasks: no elicitation, returns hardcoded tasks
- delete_task: binary approval via ctx.elicit(response_type=None)
- create_task: structured input via ctx.elicit(response_type=TaskInput)
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp import Context, FastMCP

mcp = FastMCP("TaskManager")

TASKS = {
    "T-101": {"title": "Set up CI pipeline", "priority": "high", "assignee": "Alice"},
    "T-102": {"title": "Write unit tests", "priority": "medium", "assignee": "Bob"},
    "T-103": {"title": "Update README", "priority": "low", "assignee": "Alice"},
    "T-104": {"title": "Fix login bug", "priority": "high", "assignee": "Carol"},
}


@mcp.tool
def list_tasks() -> str:
    """List all project tasks."""
    lines = []
    for task_id, info in TASKS.items():
        lines.append(
            f"{task_id}: {info['title']} "
            f"(priority: {info['priority']}, assignee: {info['assignee']})"
        )
    return "\n".join(lines)


@mcp.tool
async def delete_task(task_id: str, ctx: Context) -> str:
    """Delete a task by ID (requires human approval)."""
    if task_id not in TASKS:
        return f"Task '{task_id}' not found"
    task = TASKS[task_id]
    result = await ctx.elicit(
        f"Delete task {task_id} '{task['title']}'?", response_type=None
    )
    if result.action == "accept":
        return f"Deleted task: {task_id} ({task['title']})"
    return f"Deletion cancelled for task: {task_id}"


@dataclass
class TaskInput:
    """Structured input for creating a task (flat fields only)."""

    title: str
    priority: str
    assignee: str


@mcp.tool
async def create_task(ctx: Context) -> str:
    """Create a new task (collects details from the user via elicitation)."""
    result = await ctx.elicit(
        "Please provide the new task details", response_type=TaskInput
    )
    if result.action == "accept":
        data = result.data
        return (
            f"Created task: '{data.title}' "
            f"(priority: {data.priority}, assignee: {data.assignee})"
        )
    return "Task creation cancelled"
