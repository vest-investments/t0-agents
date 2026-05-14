"""Unit tests for the M1 structured-output fields on :class:`AgentConfig`.

Covers the three new fields (``function_kind``, ``output_schema``,
``json_mode``) and the four ``__post_init__`` branches that protect them
from incoherent combinations.  Uses :class:`AgentConfig` directly — no
gateway, no MCP client, no fakes — because every assertion here is a
config-shape contract, not a runtime behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError


class TestFunctionKindValidation:
    """``function_kind`` drives the rest of the structured-output surface."""

    def test_default_function_kind_is_chat(self) -> None:
        cfg = AgentConfig(name="a", function_name="f")
        assert cfg.function_kind == "chat"
        assert cfg.json_mode is None
        assert cfg.output_schema is None

    def test_function_kind_json_requires_json_mode(self) -> None:
        with pytest.raises(ConfigurationError, match="json_mode"):
            AgentConfig(
                name="a",
                function_name="f",
                function_kind="json",
            )

    def test_function_kind_chat_forbids_json_mode(self) -> None:
        with pytest.raises(ConfigurationError, match="forbids json_mode"):
            AgentConfig(
                name="a",
                function_name="f",
                json_mode="strict",
            )

    def test_function_kind_chat_forbids_output_schema(self) -> None:
        with pytest.raises(ConfigurationError, match="forbids output_schema"):
            AgentConfig(
                name="a",
                function_name="f",
                output_schema={"type": "object"},
            )

    def test_function_kind_json_with_valid_combo_constructs(self) -> None:
        cfg = AgentConfig(
            name="a",
            function_name="f",
            function_kind="json",
            json_mode="strict",
            output_schema={"type": "object", "required": ["x"]},
        )
        assert cfg.function_kind == "json"
        assert cfg.json_mode == "strict"
        assert cfg.output_schema == {"type": "object", "required": ["x"]}


class TestFunctionKindJsonForbidsSubAgents:
    """Contract §3.6 — JSON functions are single-shot, sub-agents are unreachable."""

    def test_function_kind_json_forbids_sub_agents(self) -> None:
        # The sub_agents-must-be-Agent check fires *before* the
        # json-vs-sub_agents check, so we need a real Agent instance.
        from typing import cast

        from tensorzero import AsyncTensorZeroGateway

        from t0_agents.core.agent import Agent
        from t0_agents.core.types import ToolProvider
        from tests._fake_mcp import FakeMCPClient
        from tests._fake_t0 import FakeT0

        sub = Agent(
            AgentConfig(name="sub", function_name="agent"),
            t0=cast(AsyncTensorZeroGateway, FakeT0([])),
            tool_provider=cast(ToolProvider, FakeMCPClient()),
        )

        with pytest.raises(ConfigurationError, match="incompatible with sub_agents"):
            AgentConfig(
                name="orch",
                function_name="extract",
                function_kind="json",
                json_mode="strict",
                sub_agents=[sub],
            )


class TestOutputSchemaPath:
    """``output_schema: Path`` is read at construction time (Q3 resolution)."""

    def test_output_schema_loads_from_absolute_path(self, tmp_path: Path) -> None:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema), encoding="utf-8")

        cfg = AgentConfig(
            name="extract",
            function_name="f",
            function_kind="json",
            json_mode="strict",
            output_schema=schema_file,
        )
        assert cfg.output_schema == schema

    def test_output_schema_loads_from_relative_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        schema = {"type": "object", "additionalProperties": False}
        schema_file = tmp_path / "rel-schema.json"
        schema_file.write_text(json.dumps(schema), encoding="utf-8")

        # Q3 resolution: relative paths are resolved against cwd.
        monkeypatch.chdir(tmp_path)
        cfg = AgentConfig(
            name="extract",
            function_name="f",
            function_kind="json",
            json_mode="strict",
            output_schema=Path("rel-schema.json"),
        )
        assert cfg.output_schema == schema

    def test_output_schema_missing_file_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ConfigurationError, match="could not be read"):
            AgentConfig(
                name="extract",
                function_name="f",
                function_kind="json",
                json_mode="strict",
                output_schema=tmp_path / "does-not-exist.json",
            )

    def test_output_schema_invalid_json_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            AgentConfig(
                name="extract",
                function_name="f",
                function_kind="json",
                json_mode="strict",
                output_schema=bad,
            )

    def test_output_schema_non_object_raises_configuration_error(
        self,
        tmp_path: Path,
    ) -> None:
        # Top-level array — not a valid JSON Schema root for TZ.
        arr = tmp_path / "array.json"
        arr.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="JSON object at the top level"):
            AgentConfig(
                name="extract",
                function_name="f",
                function_kind="json",
                json_mode="strict",
                output_schema=arr,
            )
