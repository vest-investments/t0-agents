"""Agent loop: inference -> tool calls -> execution -> result -> next inference."""

from __future__ import annotations


async def run_agent() -> None:
    raise NotImplementedError


async def execute_tool_calls() -> None:
    raise NotImplementedError


def has_tool_calls() -> None:
    raise NotImplementedError
