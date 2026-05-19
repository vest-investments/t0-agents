"""Advanced TUI customisation: audit-logging elicitation handler.

This factory ships the same assistant as ``agent.py``, but wires a
custom elicitation handler that appends every elicitation request and
outcome to ``tui-audit.jsonl`` in the current working directory.

It demonstrates two patterns documented in the Advanced tiers of
``docs/cli.md``:

* Composing a custom :class:`~fastmcp.client.elicitation.ElicitationHandler`
  on top of :func:`t0_agents.cli.make_rich_elicitation_handler`, so the
  pause / resume coordination with the live Rich panel stays
  automatic — the wrapper only adds behaviour, the stock handler still
  drives the actual prompts.
* Using :func:`t0_agents.cli.get_active_renderer` to reach the
  currently-running :class:`~t0_agents.cli.TUIRenderer` from inside the
  wrapper.  FastMCP's ``ElicitationHandler`` signature is fixed, so the
  renderer cannot be passed as a parameter; the module-global binding
  is the supported hook for that kind of side-band coordination.

Invoke via::

    uv run t0-agents run examples.10_cli.advanced:assistant_with_audit

Every run appends to ``tui-audit.jsonl``; delete the file between runs
if you want a clean log.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import ElicitRequestFormParams, ElicitRequestParams
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig
from t0_agents.cli import (
    get_active_renderer,
    make_rich_elicitation_handler,
)
from t0_agents.memory import InMemoryMemory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

AUDIT_PATH = Path("tui-audit.jsonl")


AuditHandler = Callable[
    [
        str,
        type[Any] | None,
        ElicitRequestParams,
        RequestContext[Any, LifespanContextT],
    ],
    Awaitable[ElicitResult[dict[str, Any]]],
]


def make_audit_elicitation_handler(
    audit_path: Path = AUDIT_PATH,
) -> AuditHandler:
    """Return an elicitation handler that logs every request + outcome.

    The returned handler delegates to
    :func:`make_rich_elicitation_handler` with no explicit renderer, so
    the stock handler auto-discovers the active ``TUIRenderer`` via
    :func:`get_active_renderer` and keeps the pause / resume
    coordination working.

    Before and after the delegated call, the wrapper writes a JSONL
    record describing the request mode (``form`` / ``url``), the
    prompt message, the resulting action (``accept`` / ``decline`` /
    ``cancel``), whether the user supplied content, and a monotonic
    wall-clock timestamp.  We also look up the active renderer in the
    wrapper itself — just so the audit log can tag each entry with the
    agent name the renderer is driving.
    """
    inner = make_rich_elicitation_handler()

    async def _handler(
        message: str,
        response_type: type[Any] | None,
        params: ElicitRequestParams,
        context: RequestContext[Any, LifespanContextT],
    ) -> ElicitResult[dict[str, Any]]:
        renderer = get_active_renderer()
        agent_name = renderer.agent_name if renderer is not None else None
        mode = "form" if isinstance(params, ElicitRequestFormParams) else "url"

        _append_audit(
            audit_path,
            {
                "kind": "elicit_request",
                "mode": mode,
                "message": message,
                "agent": agent_name,
                "ts": time.time(),
            },
        )
        result = await inner(message, response_type, params, context)
        _append_audit(
            audit_path,
            {
                "kind": "elicit_response",
                "action": result.action,
                "has_content": result.content is not None,
                "ts": time.time(),
            },
        )
        return result

    return _handler


def _append_audit(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as fp:
        fp.write(json.dumps(entry) + "\n")


@asynccontextmanager
async def assistant_with_audit() -> AsyncIterator[Agent]:
    """Build an Agent whose elicitation decisions are logged to JSONL.

    Same tools as ``agent.py`` (``add``, ``set_reminder``).  The
    ``Client`` is wired with the audit handler rather than the
    zero-config ``rich_elicitation_handler`` so every ``ctx.elicit``
    call leaves a trail on disk.
    """
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(
        mcp,
        elicitation_handler=make_audit_elicitation_handler(),
    )
    async with await gateway as t0, client:
        yield Agent(
            AgentConfig(
                name="audited-assistant",
                function_name="agent",
                max_turns=10,
                memory=InMemoryMemory(),
                tags={"example": "10_cli", "variant": "audit"},
            ),
            t0=t0,
            tool_provider=client,
        )
