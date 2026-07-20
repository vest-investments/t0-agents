"""Unit tests for the ``tool_call`` content-block builder."""

from __future__ import annotations

import json

import pytest
from tensorzero import ToolCall

from t0_agents.testing import tool_call


class TestName:
    def test_positional_name_sets_both_name_fields(self) -> None:
        tc = tool_call("db_lookup", q="x")

        assert isinstance(tc, ToolCall)
        assert tc.raw_name == "db_lookup"
        assert tc.name == "db_lookup"
        assert tc.type == "tool_call"

    def test_arguments_serialize_to_raw_arguments(self) -> None:
        tc = tool_call("search", query="hello", limit=10)

        assert tc.arguments == {"query": "hello", "limit": 10}
        assert json.loads(tc.raw_arguments) == {"query": "hello", "limit": 10}

    def test_no_arguments_is_empty_object(self) -> None:
        tc = tool_call("ping")

        assert tc.arguments == {}
        assert tc.raw_arguments == "{}"

    def test_nested_arguments_round_trip(self) -> None:
        tc = tool_call("place", order={"symbol": "NVDA", "qty": 1})

        assert json.loads(tc.raw_arguments) == {
            "order": {"symbol": "NVDA", "qty": 1}
        }

    def test_arguments_mirror_the_parsed_raw_arguments(self) -> None:
        # A real ToolCall's `arguments` is TensorZero's parse of
        # `raw_arguments`, so a tuple the caller passed must surface as the
        # list JSON delivers — not stay a tuple.
        tc = tool_call("t", coords=(1, 2))

        assert tc.arguments == {"coords": [1, 2]}
        assert json.loads(tc.raw_arguments) == tc.arguments

    def test_non_json_argument_fails_loud(self) -> None:
        with pytest.raises(TypeError):
            tool_call("t", when=object())


class TestServerToolForm:
    def test_server_and_tool_compose_prefixed_name(self) -> None:
        tc = tool_call(server="db", tool="lookup", q="x")

        assert tc.raw_name == "db_lookup"
        assert tc.name == "db_lookup"
        assert tc.arguments == {"q": "x"}

    def test_tool_without_server_is_bare_name(self) -> None:
        tc = tool_call(tool="lookup", q="x")

        assert tc.name == "lookup"


class TestId:
    def test_id_defaults_to_unique_value(self) -> None:
        first = tool_call("t")
        second = tool_call("t")

        assert first.id != second.id

    def test_id_override_is_honored(self) -> None:
        tc = tool_call("t", id="mock-call-1")

        assert tc.id == "mock-call-1"


class TestValidation:
    def test_neither_name_nor_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            tool_call()

    def test_both_name_and_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            tool_call("db_lookup", tool="lookup")

    def test_name_with_server_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            tool_call("db_lookup", server="db")

    def test_server_without_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            tool_call(server="db")
