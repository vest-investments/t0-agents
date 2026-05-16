"""FastMCP server for the TUI demo.

Two tools covering the event kinds the Rich panel renders:

* ``add`` — plain tool that exercises ``tool_call`` / ``tool_result``
  events with visible integer arguments.
* ``set_reminder`` — elicitation tool. Calls ``ctx.elicit`` with a
  structured :class:`ReminderConfirmation` dataclass so the CLI's
  Rich-based elicitation handler renders a schema-driven prompt
  (primitive + ``enum`` + ``default`` + ``description``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import Field

mcp = FastMCP("TUIDemo")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@dataclass
class ReminderConfirmation:
    """Structured elicitation payload for :func:`set_reminder`.

    Kept flat (primitive + ``Literal``/``enum``) so the Rich
    elicitation handler can render each field with the appropriate
    prompt type.
    """

    confirm: Annotated[
        bool,
        Field(description="Confirm the reminder?"),
    ]
    channel: Annotated[
        str,
        Field(
            description="Delivery channel",
            json_schema_extra={"enum": ["email", "sms", "push"]},
        ),
    ] = "email"
    note: Annotated[
        str,
        Field(description="Optional note for the reminder"),
    ] = ""


@mcp.tool
async def set_reminder(when: str, what: str, ctx: Context) -> str:
    """Schedule a reminder. Call this tool directly — it will open a
    confirmation form for the channel and an optional note. Do NOT ask
    the user to confirm in chat first; the form handles it.
    """
    result = await ctx.elicit(
        f"Reminder at {when}: {what}",
        response_type=ReminderConfirmation,
    )
    if result.action != "accept":
        return f"Reminder cancelled ({result.action})."
    data = result.data
    if not data.confirm:
        return "Reminder declined by the user."
    note_suffix = f" — note: {data.note}" if data.note else ""
    return (
        f"Scheduled reminder for {when!r} via {data.channel}: "
        f"{what}{note_suffix}"
    )
