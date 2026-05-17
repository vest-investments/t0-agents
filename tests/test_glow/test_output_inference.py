"""Unit tests for :func:`t0_agents.glow.output_inference.output_mode_from_return_type`.

Ten cases covering the full spec: ``str``, bare ``dict``, parameterised
``dict``, Pydantic ``BaseModel``, ``list[BaseModel]`` (synthesised
wrapper), ``Optional[T]``, PEP-604 ``T | None``, forward-ref-resolves,
forward-ref-missing → ``ConfigurationError``, missing-annotation →
``ChatOutput`` + DEBUG log.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs.chat import ChatOutput
from t0_agents.core.outputs.json import JsonOutput
from t0_agents.core.outputs.structured import StructuredOutput
from t0_agents.glow.output_inference import output_mode_from_return_type


class _Item(BaseModel):
    """Module-level Pydantic model — anchors the forward-ref-resolves case."""

    name: str


class _Other(BaseModel):
    """Second module-level model — anchors the same-module forward ref test."""

    value: int


def _no_annotation_fn():  # type: ignore[no-untyped-def]
    """Helper function with no return annotation (defined at module level
    so the strict-mypy ignore can be local)."""
    return None


def test_str_return_yields_chat_output() -> None:
    def fn() -> str:
        return "hi"

    assert isinstance(output_mode_from_return_type(fn), ChatOutput)


def test_bare_dict_yields_json_output() -> None:
    def fn() -> dict:  # type: ignore[type-arg]
        return {}

    assert isinstance(output_mode_from_return_type(fn), JsonOutput)


def test_parameterised_dict_yields_json_output() -> None:
    def fn() -> dict[str, Any]:
        return {}

    assert isinstance(output_mode_from_return_type(fn), JsonOutput)


def test_pydantic_model_yields_structured_output() -> None:
    def fn() -> _Item:  # noqa: RUF100
        ...

    out = output_mode_from_return_type(fn)
    assert isinstance(out, StructuredOutput)
    assert out.model is _Item


def test_list_of_pydantic_wraps_with_synthetic_model() -> None:
    def fn() -> list[_Item]:
        ...

    out = output_mode_from_return_type(fn)
    assert isinstance(out, StructuredOutput)
    wrapper = out.model
    assert issubclass(wrapper, BaseModel)
    assert "items" in wrapper.model_fields
    # Pydantic v2 records the field annotation on `FieldInfo.annotation`.
    assert wrapper.model_fields["items"].annotation == list[_Item]


def test_optional_pydantic_unwraps_to_inner_model() -> None:
    # Optional[_Item] exercises the typing.Union origin path of
    # _unwrap_optional; the PEP 604 sibling test below covers
    # types.UnionType.  Both must collapse to the inner model.
    def fn() -> Optional[_Item]:  # noqa: UP045
        ...

    out = output_mode_from_return_type(fn)
    assert isinstance(out, StructuredOutput)
    assert out.model is _Item


def test_pep604_optional_pydantic_unwraps_to_inner_model() -> None:
    def fn() -> _Item | None:
        ...

    out = output_mode_from_return_type(fn)
    assert isinstance(out, StructuredOutput)
    assert out.model is _Item


def test_forward_ref_to_same_module_class_resolves() -> None:
    # `from __future__ import annotations` already turns every
    # annotation into a string at runtime, so a bare `_Other` IS the
    # forward-ref form get_type_hints must resolve.
    def fn() -> _Other:
        ...

    out = output_mode_from_return_type(fn)
    assert isinstance(out, StructuredOutput)
    assert out.model is _Other


def test_forward_ref_to_missing_name_raises_configuration_error() -> None:
    # Same PEP 563 reasoning: the annotation is stored as the string
    # "MissingType" and only resolved (and fails) inside get_type_hints.
    def fn() -> MissingType:  # type: ignore[name-defined]  # noqa: F821
        ...

    with pytest.raises(ConfigurationError, match="Cannot resolve return annotation"):
        output_mode_from_return_type(fn)


def test_no_annotation_returns_chat_with_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="t0_agents.glow.output_inference"):
        out = output_mode_from_return_type(_no_annotation_fn)

    assert isinstance(out, ChatOutput)
    assert any(
        record.name == "t0_agents.glow.output_inference"
        and record.levelname == "DEBUG"
        and "no return annotation" in record.message
        for record in caplog.records
    )
