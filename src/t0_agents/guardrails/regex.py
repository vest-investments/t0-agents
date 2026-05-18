"""Regex-based content guardrail for input/output validation.

:func:`regex_guardrail` constructs a :class:`~t0_agents.core.types.Guardrail`
Protocol implementation that blocks a run when :func:`re.search` matches:

- ``phase="after_run"`` (default) scans :attr:`AgentResult.value`.  String
  values are tested directly; dicts are JSON-encoded via
  :func:`json.dumps` (with ``default=str``); Pydantic v2 models are
  unwrapped through :meth:`model_dump` first so the regex matches
  against the actual JSON payload rather than the model's
  ``__str__``.
- ``phase="before_run"`` scans :attr:`GuardrailContext.user_message` — the
  user's original query, before any inference or tool call.

Matching uses :func:`re.search` so the pattern can be unanchored — the
typical case for blockers (*"block if the response contains an SSN"*).
A match raises :class:`~t0_agents.core.exceptions.GuardrailBlockedError`
with the supplied ``message``; the
:class:`~t0_agents.core.middleware.GuardrailsMiddleware` propagates this
as a hard abort.

This factory was originally introduced for the
:mod:`t0_agents.glow` experimental surface in v1.0 M4 (the
``Annotated[T, Guardrail(...)]`` extraction test required a regex
blocker, and the existing guardrails subsystem had ``max_cost``,
``max_tool_calls``, ``block_tools``, ``require_approval``, and
``require_parsed`` but nothing regex-based).  M5 promotes it to the
canonical guardrails namespace so every ``Agent`` consumer can use it
directly, not just glow agents.  The :mod:`t0_agents.glow` package
re-exports the same callable, so existing glow imports keep working.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

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

    Args:
        pattern: Regex source (compiled with :func:`re.compile`) or a
            pre-compiled :class:`re.Pattern`.  Matching is unanchored
            (:func:`re.search`).
        message: Error message attached to the
            :class:`GuardrailBlockedError` raised on a hit.
        phase: ``"after_run"`` (default) inspects
            :attr:`AgentResult.value` at the end of the run;
            ``"before_run"`` inspects :attr:`GuardrailContext.user_message`
            before any inference fires.

    Returns:
        A :class:`Guardrail` Protocol instance ready to plug into
        :attr:`AgentConfig.guardrails`.
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
    in JSON encoding fall back to :func:`repr` — the worst case is the
    regex misses a hit; we never want the stringifier itself to raise.

    Pydantic v2 models are unwrapped via ``model_dump()`` before JSON
    encoding so the regex sees the field payload
    (``'{"ssn": "123-45-6789"}'``) rather than the model's ``__str__``
    (``"Model(ssn='123-45-6789')"``).  Without the unwrap,
    ``json.dumps(model, default=str)`` would call ``str(model)`` on
    the model itself and a regex anchored on JSON quoting would miss.
    The check is duck-typed on a callable ``model_dump`` attribute, so
    any object exposing the same interface (Pydantic, attrs-style,
    custom dataclasses with a manually defined ``model_dump``)
    benefits.  Pydantic v1's ``.dict()`` is intentionally NOT a
    fallback because the project requires v2 (see ``pyproject.toml``)
    and a generic ``hasattr(value, "dict")`` test would collide with
    unrelated third-party objects whose ``.dict()`` method means
    something different.

    Catches the broad :class:`Exception` because :func:`json.dumps` can
    surface :class:`TypeError` (default), :class:`ValueError` (NaN / inf
    without ``allow_nan=False`` workarounds), :class:`OverflowError`
    (circular references), and :class:`RecursionError` (deeply nested
    structures).  A guardrail's stringifier must never crash the run —
    the docstring contract is "best-effort string for regex matching",
    and :func:`repr` always succeeds.
    """
    if isinstance(value, str):
        return value
    try:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            value = model_dump()
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)
