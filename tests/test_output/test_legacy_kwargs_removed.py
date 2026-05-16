"""Tier 2 removal verification: legacy AgentConfig JSON kwargs raise TypeError.

The five v0.12 fields (``function_kind``, ``json_mode``,
``output_schema``, ``permissive_json_parsing``, ``max_json_retries``)
were removed in v1.0 (contract §2.4 Tier 2).  Passing any of them to
:class:`AgentConfig` now raises a ``TypeError`` from the dataclass
constructor — no quiet acceptance, no silent translation.
"""

from __future__ import annotations

import pytest

from t0_agents.core.config import AgentConfig


def test_function_kind_kwarg_removed() -> None:
    with pytest.raises(TypeError, match="function_kind"):
        AgentConfig(
            name="x",
            function_name="y",
            function_kind="json",  # type: ignore[call-arg]
        )


def test_json_mode_kwarg_removed() -> None:
    with pytest.raises(TypeError, match="json_mode"):
        AgentConfig(
            name="x",
            function_name="y",
            json_mode="strict",  # type: ignore[call-arg]
        )


def test_output_schema_kwarg_removed() -> None:
    with pytest.raises(TypeError, match="output_schema"):
        AgentConfig(
            name="x",
            function_name="y",
            output_schema={"type": "object"},  # type: ignore[call-arg]
        )


def test_permissive_json_parsing_kwarg_removed() -> None:
    with pytest.raises(TypeError, match="permissive_json_parsing"):
        AgentConfig(
            name="x",
            function_name="y",
            permissive_json_parsing=False,  # type: ignore[call-arg]
        )


def test_max_json_retries_kwarg_removed() -> None:
    with pytest.raises(TypeError, match="max_json_retries"):
        AgentConfig(
            name="x",
            function_name="y",
            max_json_retries=5,  # type: ignore[call-arg]
        )
