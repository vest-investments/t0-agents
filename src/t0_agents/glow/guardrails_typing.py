""":func:`regex_guardrail` re-export + ``Annotated[T, Guardrail(...)]`` extraction.

Two pieces:

1. :func:`regex_guardrail` — re-exported from :mod:`t0_agents.guardrails`
   so the glow surface stays drop-in for callers who write
   ``Annotated[str, regex_guardrail(...)]`` directly on a parameter or
   return type.  The factory itself lives at
   ``src/t0_agents/guardrails/regex.py`` and is the canonical home —
   every ``Agent`` consumer can use it, not just glow agents.  The
   v1.0 module migration landed in M5 (analog to the M2 D5
   toolless-chat unlock).

2. :func:`_extract_guardrails` — walks ``@app.agent`` parameter
   annotations and the return annotation; harvests every
   :class:`~t0_agents.core.types.Guardrail` Protocol instance carried
   in ``Annotated[T, ...]`` metadata.  Folded into
   ``_AgentSpec.guardrails`` at registration time; runtime path
   (``GuardrailsMiddleware``) is unchanged.  This walker is
   glow-specific — the typed-metadata pattern is what glow adds; v1.0
   passes guardrails through ``AgentConfig(guardrails=[...])``
   explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from t0_agents.core.types import Guardrail
from t0_agents.guardrails import regex_guardrail

__all__ = ["regex_guardrail"]


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
