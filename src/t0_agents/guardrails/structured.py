"""Result-shape guardrails for JSON / structured-output agents.

Today's single occupant — :func:`require_parsed` — is the opt-in
counterpart to the permissive-parsing layer.  Adding the guardrail to
``AgentConfig.guardrails`` upgrades the "``parsed is None`` is a runtime
non-error" contract into a hard :class:`GuardrailBlockedError` abort,
for applications that cannot proceed without a parseable structured
response.

For chat (text) runs the guardrail is a no-op — ``parsed`` is
permanently ``None`` for the chat path, so an unconditional fail
would be wrong.  Detection happens via :attr:`AgentResult.function_kind`.
"""

from __future__ import annotations

from typing import Any

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import AgentResult, Guardrail
from t0_agents.guardrails.base import GuardrailContext

__all__ = ["require_parsed"]


def require_parsed(*, name: str = "require_parsed") -> Guardrail:
    """Hard-abort when a JSON run completes with ``parsed is None``.

    Opt-in: add to ``AgentConfig(guardrails=[require_parsed()])`` when
    your application cannot tolerate an unparseable JSON output.  The
    guardrail consults :attr:`AgentResult.function_kind` and only fires
    on ``"json"`` runs — chat runs pass through unchanged.

    No schema validation is performed: the check is exclusively
    "did any layer (strict / permissive / retry) populate ``parsed``"
    (see :attr:`AgentResult.parse_strategy`).  Schema conformance is
    the caller's responsibility (TZ guide §11.15, structured-outputs
    Q2).

    Args:
        name: Human-readable identifier surfaced in
            :class:`GuardrailLogEntry.guardrail_name` and the abort
            message.  Override when stacking several
            ``require_parsed`` instances (rare).
    """

    guardrail_name = name  # closed-over rebind so the inner class attr survives

    class _RequireParsed:
        name = guardrail_name

        async def before_run(self, ctx: GuardrailContext) -> None:
            return None

        async def before_tool(
            self, ctx: GuardrailContext, tool_call: Any
        ) -> str | None:
            return None

        async def after_tool(
            self, ctx: GuardrailContext, tool_call: Any, result_str: str
        ) -> None:
            return None

        async def after_run(
            self,
            ctx: GuardrailContext,
            result: AgentResult[Any] | None,
        ) -> None:
            # ``result`` is ``None`` on a fatal-abort teardown (the run
            # never produced an :class:`AgentResult`); chat runs short-
            # circuit because ``parsed`` is structurally unreachable
            # for them; only the JSON branch with a missing ``parsed``
            # triggers the hard block.
            if result is None:
                return
            if result.function_kind != "json":
                return
            if result.parsed is None:
                raise GuardrailBlockedError(
                    f"{name}: JSON run completed with parsed=None "
                    f"(parse_strategy={result.parse_strategy!r}, "
                    f"attempts={result.json_parse_attempts})"
                )

    return _RequireParsed()
