"""Internal type-introspection helpers used by :mod:`output_inference`.

Six small predicates / peelers operating on the runtime type objects that
:func:`typing.get_type_hints` returns.  They underpin the return-type →
:class:`OutputMode` mapping; M4 will extend this surface with metadata
extraction for :class:`Annotated[T, Guardrail(...)]`.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, create_model


def _strip_annotated(tp: Any) -> Any:
    """Return the underlying type if ``tp`` is ``Annotated[X, ...]``.

    Annotated aliases expose ``__metadata__`` and place the underlying
    type as ``get_args(tp)[0]``.  Non-Annotated inputs pass through
    unchanged so the helper composes safely with the other peelers.
    """
    if hasattr(tp, "__metadata__"):
        return get_args(tp)[0]
    return tp


def _unwrap_optional(tp: Any) -> Any:
    """Peel ``Optional[T]`` / ``T | None`` to ``T``.

    Covers ``typing.Union[T, None]`` (origin ``typing.Union``) and PEP 604
    ``T | None`` (origin ``types.UnionType``).  Returns ``T`` only when
    the union has exactly one non-``None`` arg; richer unions
    (``Union[T, S]``) pass through unchanged.
    """
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _peel_to_core(tp: Any) -> Any:
    """Iteratively strip ``Annotated[...]`` and ``Optional[...]`` wrappers.

    Resolves the M1 §3.1 carry-forward: a single composition of
    :func:`_strip_annotated` and :func:`_unwrap_optional` only handles one
    ordering — ``Optional[Annotated[T, m]]`` works under ``strip(unwrap(...))``
    but ``Annotated[Optional[T], m]`` does not (the ``Annotated`` wrapper
    hides the ``Union`` from ``_unwrap_optional`` on the first pass).
    Looping until convergence handles every nesting shape including
    ``Optional[Annotated[Optional[T], m]]``.
    """
    while True:
        peeled = _strip_annotated(_unwrap_optional(tp))
        if peeled is tp:
            return tp
        tp = peeled


def _is_dict_alias(tp: Any) -> bool:
    """Match ``dict``, ``dict[K, V]``, and ``Mapping[K, V]``."""
    if tp is dict:
        return True
    origin = get_origin(tp)
    return origin is dict or origin is Mapping


def _is_pydantic_model(tp: Any) -> bool:
    """``True`` when ``tp`` is a Pydantic ``BaseModel`` subclass."""
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_list_of_pydantic(tp: Any) -> bool:
    """``True`` when ``tp`` is ``list[T]`` with ``T`` a Pydantic model."""
    if get_origin(tp) is not list:
        return False
    args = get_args(tp)
    return bool(args) and _is_pydantic_model(args[0])


def _wrap_list_model(tp: Any) -> type[BaseModel]:
    """Synthesise a ``BaseModel`` subclass with a single ``items: list[T]`` field.

    :class:`~t0_agents.core.outputs.structured.StructuredOutput` needs a
    Pydantic class to derive the JSON schema from; a bare ``list[T]``
    annotation has no model, so we mint one at registration time.  M2's
    ``@app.agent`` will unwrap ``result.value.items`` so the user-facing
    return matches the original ``list[T]`` annotation.

    The synthesised class is named ``<T>List`` (e.g. ``ItemList`` for
    ``list[Item]``) so schema titles read sensibly.
    """
    inner = get_args(tp)[0]
    # `list[inner]` is a runtime generic-alias construction; mypy cannot
    # introspect `inner` (typed Any here) as a static type, so silence
    # the valid-type check.
    return create_model(
        f"{inner.__name__}List",
        items=(list[inner], ...),  # type: ignore[valid-type]
    )
