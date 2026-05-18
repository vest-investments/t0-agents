""":func:`regex_guardrail` + ``Annotated[T, Guardrail(...)]`` extraction.

Two pieces:

1. :func:`regex_guardrail` — convenience factory returning an object that
   satisfies the v1.0 :class:`~t0_agents.core.types.Guardrail` Protocol.
   The existing :mod:`t0_agents.guardrails` package ships ``max_cost``,
   ``max_tool_calls``, ``block_tools``, ``require_approval``, and
   ``require_parsed``; a regex-based blocker was missing, and the M4
   spec requires a regex test.

   **Carry-forward (M5 or post-M5):** migrate this factory to
   ``src/t0_agents/guardrails/regex.py`` so every Agent consumer
   benefits (analog to the M2 D5 toolless-chat unlock).  Until
   migrated, the function is glow-private.

2. :func:`_extract_guardrails` — walks ``@app.agent`` parameter
   annotations and the return annotation; harvests every
   :class:`~t0_agents.core.types.Guardrail` Protocol instance carried
   in ``Annotated[T, ...]`` metadata.  Folded into
   ``_AgentSpec.guardrails`` at registration time; runtime path
   (``GuardrailsMiddleware``) is unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from t0_agents.core.exceptions import GuardrailBlockedError
from t0_agents.core.types import AgentResult, Guardrail

__all__ = ["regex_guardrail"]


def regex_guardrail(
    pattern: str | re.Pattern[str],
    message: str,
    *,
    phase: Literal["before_run", "after_run"] = "after_run",
) -> Guardrail:
    """Construct a :class:`Guardrail` that blocks when ``pattern`` matches.

    ``phase="after_run"`` (default) checks ``result.value`` — string
    values are tested directly; non-string values are JSON-encoded via
    ``json.dumps(..., default=str)`` so dicts and Pydantic models still
    get scanned consistently.

    ``phase="before_run"`` checks ``ctx.user_message`` — the
    pre-inference user query.  This is what
    :class:`GuardrailsMiddleware` populates the context with at the
    start of the run (before memory injection or any tool call).

    Matching uses :func:`re.search` so the pattern can be unanchored
    (the typical case for blockers — "block if the response *contains*
    an SSN").  Raises :class:`GuardrailBlockedError` with the supplied
    ``message`` on a hit; the middleware propagates this as a hard
    abort (``T0AgentsFatalError`` subclass).
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    class _RegexGuardrail:
        """Concrete Guardrail Protocol implementation backed by a regex."""

        name: str = f"regex_{phase}"

        async def before_run(self, ctx: Any) -> None:
            if phase != "before_run":
                return
            user_message = getattr(ctx, "user_message", "")
            if compiled.search(user_message):
                raise GuardrailBlockedError(message)

        async def before_tool(
            self, ctx: Any, tool_call: Any,
        ) -> str | None:
            return None

        async def after_tool(
            self, ctx: Any, tool_call: Any, result_str: str,
        ) -> None:
            return None

        async def after_run(
            self, ctx: Any, result: AgentResult[Any] | None,
        ) -> None:
            if phase != "after_run" or result is None:
                return
            text = _stringify_result_value(result.value)
            if compiled.search(text):
                raise GuardrailBlockedError(message)

    return _RegexGuardrail()


def _stringify_result_value(value: Any) -> str:
    """Render an ``AgentResult.value`` for regex matching.

    Strings pass through; everything else is JSON-encoded with
    ``default=str`` so dicts and Pydantic models become inspectable
    text without forcing the user to write a custom predicate.  Errors
    in JSON encoding fall back to ``repr`` — the worst case is the
    regex misses a hit; we never want the stringifier itself to raise.

    Catches the broad ``Exception`` because ``json.dumps`` can surface
    ``TypeError`` (default), ``ValueError`` (NaN / inf without
    ``allow_nan=False`` workarounds), ``OverflowError`` (circular
    references), and ``RecursionError`` (deeply nested structures).
    A guardrail's stringifier must never crash the run — the docstring
    contract is "best-effort string for regex matching", and ``repr()``
    always succeeds.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def _extract_guardrails(
    fn: Callable[..., Any],
    hints: dict[str, Any] | None = None,
) -> list[Guardrail]:
    """Harvest every :class:`Guardrail` Protocol instance from ``fn``'s annotations.

    Walks ``Annotated[T, *meta]`` entries on both parameters AND the
    return annotation, collects metadata satisfying the runtime
    ``Guardrail`` Protocol (any object with ``.name`` plus the four
    async hooks).  Other Annotated metadata (MCPServer instances, plain
    markers) is silently ignored — they belong to other extractors.

    ``hints`` may be passed pre-resolved by the caller (the decorator
    has already paid the ``get_type_hints`` cost once); otherwise we
    resolve here.  Resolution failures (unresolvable forward refs)
    short-circuit to ``[]`` so this helper composes safely with the
    decorator's own forward-ref enforcement.
    """
    if hints is None:
        try:
            hints = get_type_hints(fn, include_extras=True)
        except (NameError, TypeError):
            return []

    guardrails: list[Guardrail] = []
    for annotation in hints.values():
        guardrails.extend(_guardrails_from_annotation(annotation))
    return guardrails


def _guardrails_from_annotation(annotation: Any) -> list[Guardrail]:
    """Return every Guardrail instance reachable from one annotation.

    Only ``Annotated[T, *meta]`` carries guardrail metadata in the
    glow surface; plain types pass through with no harvest.  We do
    not recurse into nested ``Annotated`` — :func:`_peel_to_core` in
    ``output_inference`` handles that for the output-mode path; for
    guardrails the convention is one ``Annotated`` layer per
    parameter / return.
    """
    if get_origin(annotation) is not Annotated:
        return []
    metadata = get_args(annotation)[1:]
    return [m for m in metadata if isinstance(m, Guardrail)]
