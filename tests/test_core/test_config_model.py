"""Unit tests for :class:`AgentConfig`'s functionless-routing surface.

M3 (glow v2) adds ``model``/``variant``/``params``/``cache``:

- ``model`` XOR ``function_name``; neither falls back to ``"agent"``.
- ``variant`` requires the function path — the gateway rejects
  ``variant_name`` for ``model``-routed inference.
- ``model`` + a JSON-function output mode raises at construction with a
  ready-to-paste tensorzero.toml stanza (the functionless passthrough
  ignores ``output_schema``; verified against gateway 2026.4.0).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from t0_agents.core.config import AgentConfig
from t0_agents.core.exceptions import ConfigurationError
from t0_agents.core.outputs import (
    ChatOutput,
    JsonOutput,
    StructuredOutput,
    ToolCallOutput,
)
from t0_agents.core.outputs.streaming import StreamingOutput

MODEL = "groq::llama-3.3-70b-versatile"


class TestModelFunctionNameXor:
    def test_model_alone_leaves_function_name_none(self) -> None:
        cfg = AgentConfig(name="a", model=MODEL)
        assert cfg.model == MODEL
        assert cfg.function_name is None

    def test_function_name_alone_unchanged(self) -> None:
        cfg = AgentConfig(name="a", function_name="f")
        assert cfg.function_name == "f"
        assert cfg.model is None

    def test_neither_falls_back_to_agent(self) -> None:
        cfg = AgentConfig(name="a")
        assert cfg.function_name == "agent"
        assert cfg.model is None

    def test_both_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="XOR"):
            AgentConfig(name="a", function_name="f", model=MODEL)

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="model"):
            AgentConfig(name="a", model="")

    def test_empty_function_name_still_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="function_name"):
            AgentConfig(name="a", function_name="")


class TestVariant:
    def test_variant_with_function_name_ok(self) -> None:
        cfg = AgentConfig(name="a", function_name="f", variant="gpt4o_v2")
        assert cfg.variant == "gpt4o_v2"

    def test_variant_with_default_function_ok(self) -> None:
        cfg = AgentConfig(name="a", variant="pinned")
        assert cfg.function_name == "agent"
        assert cfg.variant == "pinned"

    def test_variant_with_model_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="variant"):
            AgentConfig(name="a", model=MODEL, variant="default")


class TestModelOutputModeCompatibility:
    def test_model_with_chat_output_ok(self) -> None:
        cfg = AgentConfig(name="a", model=MODEL, output=ChatOutput())
        assert cfg.model == MODEL

    def test_model_with_tool_call_output_ok(self) -> None:
        cfg = AgentConfig(name="a", model=MODEL, output=ToolCallOutput())
        assert cfg.model == MODEL

    def test_model_with_streaming_chat_output_ok(self) -> None:
        cfg = AgentConfig(
            name="a", model=MODEL, output=StreamingOutput(base=ChatOutput())
        )
        assert cfg.model == MODEL

    def test_model_with_json_output_raises_with_stanza(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            AgentConfig(name="planner", model=MODEL, output=JsonOutput())
        message = str(excinfo.value)
        assert "[functions.planner]" in message
        assert 'type = "json"' in message
        assert f'model = "{MODEL}"' in message

    def test_model_with_structured_output_raises_with_stanza(self) -> None:
        class Plan(BaseModel):
            step: str

        with pytest.raises(ConfigurationError) as excinfo:
            AgentConfig(name="planner", model=MODEL, output=StructuredOutput(Plan))
        assert "[functions.planner]" in str(excinfo.value)

    def test_json_output_without_model_still_ok(self) -> None:
        cfg = AgentConfig(name="a", function_name="f", output=JsonOutput())
        assert cfg.output.requires_json_function is True


class TestParamsCache:
    def test_params_and_cache_stored_verbatim(self) -> None:
        params = {"chat_completion": {"temperature": 0.2}}
        cache = {"enabled": "on", "max_age_s": 3600}
        cfg = AgentConfig(name="a", model=MODEL, params=params, cache=cache)
        assert cfg.params == params
        assert cfg.cache == cache

    def test_params_cache_default_none(self) -> None:
        cfg = AgentConfig(name="a")
        assert cfg.params is None
        assert cfg.cache is None
