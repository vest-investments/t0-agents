"""Map a function's return annotation to a v1.0 :class:`OutputMode`.

The bridge between glow's type-hint-driven DX and v1.0's polymorphic
output surface.  M2's ``@app.agent`` calls
:func:`output_mode_from_return_type` at decoration time to derive the
``OutputMode`` for the wrapped function; failures (e.g. unresolvable
forward refs) surface as :class:`ConfigurationError` so the user sees
them at registration time rather than during invocation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast, get_type_hints

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.output import OutputMode
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.glow._typing import (
    _is_dict_alias,
    _is_list_of_pydantic,
    _is_pydantic_model,
    _strip_annotated,
    _unwrap_optional,
    _wrap_list_model,
)

__all__ = ["output_mode_from_return_type"]

logger = logging.getLogger(__name__)


def output_mode_from_return_type(fn: Callable[..., Any]) -> OutputMode[Any]:
    """Derive an :class:`OutputMode` from ``fn``'s return annotation.

    Mapping:

    - ``str`` / ``type(None)`` / missing annotation →
      :class:`~t0_agents.core.outputs.chat.ChatOutput`.  Missing-annotation
      case logs DEBUG so the implicit fallback is observable.
    - ``dict`` / ``dict[K, V]`` / ``Mapping[K, V]`` →
      :class:`~t0_agents.core.outputs.json.JsonOutput`.
    - Pydantic ``BaseModel`` subclass →
      :class:`~t0_agents.core.outputs.structured.StructuredOutput`.
    - ``list[T]`` with ``T`` a Pydantic model →
      :class:`StructuredOutput` over a synthesised wrapper (M2's
      ``@app.agent`` unwraps ``result.value.items``).
    - ``Optional[T]`` / ``T | None`` → as ``T``.
    - Everything else → :class:`ChatOutput` (the helper is also called
      for explicit chat agents; no warning).

    Forward-ref failures (``NameError`` / ``TypeError`` from
    :func:`typing.get_type_hints`) re-raise as :class:`ConfigurationError`
    so M2's ``@app.agent`` can surface them at registration time.
    """
    try:
        hints = get_type_hints(
            fn,
            include_extras=True,
            localns=fn.__globals__,
        )
    except (NameError, TypeError) as exc:
        raise ConfigurationError(
            f"Cannot resolve return annotation for {fn.__qualname__!r}: {exc}"
        ) from exc

    if "return" not in hints:
        logger.debug(
            "Function %r has no return annotation; defaulting to ChatOutput.",
            fn.__qualname__,
        )
        return ChatOutput()

    rt = _strip_annotated(_unwrap_optional(hints["return"]))

    if rt is str or rt is type(None):
        return ChatOutput()
    if _is_dict_alias(rt):
        return JsonOutput()
    # `StructuredOutput.name` is a `@property` while
    # :class:`OutputMode.name` is declared as a settable attribute, so
    # mypy strict needs the explicit cast to satisfy the Protocol
    # conformance check.  Other v1.0 sites (e.g.
    # `AgentConfig(output=StructuredOutput(...))`) hit the same coercion
    # at the call-site, outside `src/`.
    if _is_pydantic_model(rt):
        return cast(OutputMode[Any], StructuredOutput(rt))
    if _is_list_of_pydantic(rt):
        return cast(OutputMode[Any], StructuredOutput(_wrap_list_model(rt)))
    return ChatOutput()
