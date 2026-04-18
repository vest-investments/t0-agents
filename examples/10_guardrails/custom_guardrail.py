"""Custom guardrail: domain allowlist with a custom fatal error.

Builds on the four built-ins shown in ``main.py`` by adding one policy
guardrail written directly against the ``Guardrail`` Protocol.  The
demo exercises every Advanced-tier pattern the documentation describes:

- defining a :class:`T0AgentsFatalError` subclass for a domain-specific
  hard abort,
- inspecting ``tool_call.arguments`` inside ``before_tool``,
- accumulating per-run state on ``GuardrailContext.extra``,
- reporting the accumulated audit in ``after_run`` (which fires even when
  the run aborts — the audit survives).

Scenario: the agent is asked to send a marketing email to a recipient
whose domain is not on the allowlist.  The custom guardrail blocks the
tool dispatch and raises ``UnauthorizedRecipientError``, which propagates
past the agent loop (``T0AgentsFatalError`` semantics).  ``after_run``
still fires with ``result=None``; it prints the attempted recipients so
operators see the evidence even on aborted runs.

Requirements:
    docker compose up -d          # boots TensorZero gateway + ClickHouse
    GROQ_API_KEY in .env
    uv run python examples/10_guardrails/custom_guardrail.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig, Guardrail, T0AgentsFatalError
from t0_agents.guardrails import GuardrailContext, max_tool_calls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import mcp  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("custom_guardrail")


class UnauthorizedRecipientError(T0AgentsFatalError):
    """Raised when an agent targets an email domain outside the allowlist.

    Inherits :class:`T0AgentsFatalError` so the agent loop's
    ``_execute_single_tool_call`` re-raises it past its broad
    ``except Exception`` trap — the exception surfaces verbatim to the
    caller of ``Agent.run()`` instead of being stringified into a tool
    error.
    """


def enforce_email_allowlist(
    *,
    tools: list[str],
    field: str = "to",
    domains: list[str],
) -> Guardrail:
    """Return a guardrail that hard-aborts on off-allowlist recipients.

    Args:
        tools: Tool names whose calls should be inspected.  Non-listed
            tools pass through untouched.
        field: Name of the argument that carries the recipient.  Defaults
            to ``"to"`` to match the example server's ``send_email``.
        domains: Lower-case domains the agent is allowed to target.  The
            ``@`` separator is optional — ``"example.com"`` and
            ``"@example.com"`` are equivalent.
    """
    gated = frozenset(tools)
    allowed = frozenset(d.lower().lstrip("@") for d in domains)

    class _Allowlist:
        name = f"email_allowlist(domains={sorted(allowed)!r})"

        async def before_run(self, ctx: GuardrailContext) -> None:
            ctx.extra["email_attempts"] = []

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            if tool_call.name not in gated:
                return None

            target = str(tool_call.arguments.get(field, "")).strip()
            domain = target.split("@", 1)[1].lower() if "@" in target else ""
            ctx.extra["email_attempts"].append(
                {"to": target, "domain": domain, "allowed": domain in allowed}
            )
            if domain not in allowed:
                raise UnauthorizedRecipientError(
                    f"recipient {target!r} is not in the allowlist "
                    f"{sorted(allowed)}"
                )
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self, ctx: GuardrailContext, result: Any
        ) -> None:
            attempts = ctx.extra.get("email_attempts", [])
            if not attempts:
                return
            status = "aborted" if result is None else "completed"
            logger.info(
                "Run %s — %d email attempt(s) observed:", status, len(attempts)
            )
            for entry in attempts:
                verdict = "allowed" if entry["allowed"] else "BLOCKED"
                logger.info(
                    "  %s  to=%r  domain=%r",
                    verdict, entry["to"], entry["domain"],
                )

    return _Allowlist()


async def main() -> None:
    gateway = cast(
        Awaitable[AsyncTensorZeroGateway],
        AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000"),
    )
    client: Client[Any] = Client(mcp)
    config = AgentConfig(
        name="email_agent",
        function_name="agent",
        max_turns=6,
        tool_timeout=10.0,
        guardrails=[
            enforce_email_allowlist(
                tools=["send_email"],
                domains=["example.com"],
            ),
            max_tool_calls(per_run=5),  # belt-and-braces safety cap
        ],
    )

    async with await gateway as t0, client:
        agent = Agent(config, t0=t0, tool_provider=client)

        try:
            result = await agent.run(
                "Send a brief marketing email to eve@evil.io announcing "
                "our product launch. Subject: 'Big news'. Body: one sentence."
            )
        except UnauthorizedRecipientError as exc:
            print(f"\nBlocked by custom guardrail: {exc}")
            return
        except T0AgentsFatalError as exc:
            print(f"\nFatal guardrail: {exc}")
            return

    print(f"\nResponse: {result.response}")
    print(f"Turns: {result.turns}")
    print(f"Guardrail entries: {len(result.guardrail_log)}")


if __name__ == "__main__":
    asyncio.run(main())
