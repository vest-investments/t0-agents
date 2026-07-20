"""Elicit-survival regression gates for the two tool-call timeout fuses.

Two independent fuses govern a glow tool call:

1. The **per-call deadline** — ``tool_timeout`` / ``call_tool(timeout=)``:
   a wall-clock limit on the response future, transport-independent, NOT
   reset by server activity.  ``None`` disables it (M1 fix — the ``Agent``
   terminals used to coerce ``None`` back to 30 s, silently killing long
   elicitation waits).
2. The **httpx read timeout** on a remote hop — the maximum silent gap
   between bytes on the wire; ANY SSE frame the server emits (progress,
   log message, elicitation request) resets it.  ``MCPServer(timeout=)``
   (new in M1) pins it per server; the SDK default is 300 s.

The in-memory pair proves fuse 1 through the real FastMCP boundary: an
explicit 0.5 s timeout kills a 1.5 s elicit wait, ``None`` lets the same
wait complete.  Fuse 2 is proven end-to-end through an ephemeral uvicorn
server by ``_timeout_survival_script.py``, driven here as a subprocess —
see ``test_read_window_survival_scenarios_in_subprocess`` for why the
isolation is load-bearing.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.elicitation import ElicitResult
from mcp import McpError
from tensorzero import AsyncTensorZeroGateway

from t0_agents.glow import App, MCPServer


def _fake_t0() -> AsyncTensorZeroGateway:
    # No inference runs in these tests — only tool round-trips through the
    # owned Client — so a bare object stands in.
    return cast(AsyncTensorZeroGateway, object())


def _slow_approve(delay: float) -> Any:
    """Elicitation handler that deliberates for ``delay`` seconds, then accepts."""

    async def handler(
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]:
        await asyncio.sleep(delay)
        return ElicitResult(action="accept", content={})

    return handler


# --- Fuse 1: the per-call deadline (in-memory, no httpx involved) -----------


def _make_mem_server() -> FastMCP[Any]:
    """Fresh in-memory server per test.

    Every server in this file is per-test: a FastMCP instance reused
    across tests lives across event loops, and this file deliberately
    leaves timed-out calls behind — shared instances turn that into
    order-dependent hangs.
    """
    server: FastMCP[Any] = FastMCP("slow-hitl-mem")

    @server.tool
    async def confirm(ctx: Context) -> str:
        """Binary approval that blocks until the human decides."""
        result = await ctx.elicit("Proceed?", response_type=None)
        return "confirmed" if result.action == "accept" else "declined"

    return server


async def test_explicit_timeout_still_fires_on_slow_elicit() -> None:
    """Control: a 0.5 s per-call timeout kills a 1.5 s elicit wait."""
    app = App(
        t0=_fake_t0(),
        servers=[MCPServer(server=_make_mem_server(), namespace="mem")],
        elicitation_handler=_slow_approve(1.5),
    )
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        with pytest.raises(McpError, match=r"[Tt]imed out"):
            await client.call_tool("mem_confirm", {}, timeout=0.5)
        # Let the handler finish deliberating so teardown doesn't cancel a
        # mid-flight elicitation resolution.
        await asyncio.sleep(1.2)


async def test_none_timeout_survives_slow_elicit_in_memory() -> None:
    """``timeout=None`` lets the same 1.5 s elicit wait complete.

    Paired with the control above this proves ``None`` truly disables the
    per-call deadline — there is no residual hidden fuse (the old 30 s
    coercion would be caught by the loop-level spy tests before this).
    """
    app = App(
        t0=_fake_t0(),
        servers=[MCPServer(server=_make_mem_server(), namespace="mem")],
        elicitation_handler=_slow_approve(1.5),
    )
    async with app.lifespan():
        client = app._owned_client
        assert client is not None
        result = await client.call_tool("mem_confirm", {}, timeout=None)
    assert result.data == "confirmed"


# --- Fuse 2: the httpx read window (real HTTP, subprocess-isolated) ---------


def test_read_window_survival_scenarios_in_subprocess() -> None:
    """Run the real-HTTP elicit-survival scenarios in a pristine process.

    Scenario A: a 5 s elicit survives a 2 s ``MCPServer(timeout=)`` read
    window because ``ctx.info`` heartbeats reset the gap timer.
    Scenario B: without heartbeats the window kills the wire at ~2 s and
    the call fails at its 8 s per-call deadline instead of completing at
    ~5 s — the differential with A proves the knob reaches the wire.

    Subprocess isolation is deliberate, not convenience: fastmcp 3.2.3's
    ``create_proxy`` per-call clients have an unbounded connect race when
    several client/uvicorn cycles share one process
    (``client_init_timeout`` defaults to ``None``), and these scenarios
    kill streams mid-elicit — run in-process they poison later, unrelated
    ``App``+uvicorn tests into order-dependent hangs (observed ~1-in-4
    full-suite runs before the split).  See the script's docstring.
    """
    pytest.importorskip("uvicorn")
    script = Path(__file__).with_name("_timeout_survival_script.py")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"survival script failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "SURVIVAL-OK A" in proc.stdout
    assert "SURVIVAL-OK B" in proc.stdout
