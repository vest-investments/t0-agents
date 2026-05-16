"""Unit tests for :class:`AgentResult` v0.12 read-side compat properties.

Verifies the Tier 0 byte-identity guarantee: every v0.12 read-side
property (``response``, ``parsed``, ``parse_strategy``,
``json_parse_attempts``, ``function_kind``) remains readable on the
generic ``AgentResult[T]`` in v1.0.
"""

from __future__ import annotations

from t0_agents.core.output import OutputDiagnostics
from t0_agents.core.types import AgentResult, GuardrailLogEntry


class TestChatShape:
    def test_chat_result_via_value_kwarg(self) -> None:
        result: AgentResult[str] = AgentResult(value="hello")
        assert result.value == "hello"
        assert result.response == "hello"
        assert result.parsed is None
        assert result.function_kind == "chat"
        assert result.parse_strategy == "strict"  # default
        assert result.json_parse_attempts == 0

    def test_chat_result_via_response_kwarg_alias(self) -> None:
        # v0.12 byte-identity: mcp/loop.py constructs AgentResult(response=...).
        result: AgentResult[str] = AgentResult(response="hello")
        assert result.value == "hello"
        assert result.response == "hello"


class TestJsonShape:
    def test_json_result_with_diagnostics(self) -> None:
        result = AgentResult(
            value={"a": 1},
            diagnostics=OutputDiagnostics(
                parse_strategy="permissive",
                attempts=1,
                raw_text='{"a": 1}',
            ),
        )
        assert result.value == {"a": 1}
        assert result.response == '{"a": 1}'
        assert result.parsed == {"a": 1}
        assert result.parse_strategy == "permissive"
        assert result.json_parse_attempts == 1
        assert result.function_kind == "json"

    def test_json_result_with_none_value(self) -> None:
        result = AgentResult(
            value=None,
            diagnostics=OutputDiagnostics(
                parse_strategy="none",
                attempts=2,
                raw_text="garbage",
            ),
        )
        assert result.value is None
        assert result.parsed is None
        assert result.parse_strategy == "none"
        assert result.json_parse_attempts == 2
        # function_kind is "json" because value is not a str
        assert result.function_kind == "json"


class TestConstructorContract:
    def test_neither_value_nor_response_raises(self) -> None:
        try:
            AgentResult()  # type: ignore[call-overload]
        except TypeError as exc:
            assert "value" in str(exc).lower()
        else:
            raise AssertionError("AgentResult() should have raised TypeError")

    def test_both_value_and_response_raises(self) -> None:
        try:
            AgentResult(value="x", response="y")  # type: ignore[call-overload]
        except TypeError as exc:
            assert "both" in str(exc).lower() or "either" in str(exc).lower()
        else:
            raise AssertionError("Both args should have raised TypeError")


class TestFieldsRetained:
    def test_guardrail_log_preserved(self) -> None:
        entry = GuardrailLogEntry(
            guardrail_name="g",
            hook="before_run",
            kind="pass",
        )
        result = AgentResult(value="hi", guardrail_log=[entry])
        assert result.guardrail_log == [entry]

    def test_inference_ids_preserved(self) -> None:
        result = AgentResult(
            value="hi", inference_ids=["abc", "def"]
        )
        assert result.inference_ids == ["abc", "def"]
