"""Reference elicitation handlers for testing and prototyping."""

from __future__ import annotations

from typing import Any

from fastmcp.client.elicitation import ElicitResult


async def auto_approve_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
    """Auto-approve binary approval requests; decline structured input.

    Approves when ``response_type is None`` (binary approval) and declines
    otherwise, since no meaningful data can be provided automatically.

    Intended for **testing only** — not a substitute for a real handler in
    production.
    """
    if response_type is None:
        return ElicitResult(action="accept", content={})
    return ElicitResult(action="decline")


async def auto_decline_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
    """Auto-decline all elicitation requests.

    Intended for **testing only** — useful for verifying that tools handle
    rejection gracefully.
    """
    return ElicitResult(action="decline")
