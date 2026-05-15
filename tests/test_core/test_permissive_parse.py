"""Unit tests for :func:`t0_agents.core.structured.permissive_parse`.

Pure-function tests — no TensorZero, no fakes.  The function never raises:
every test asserts on the ``(value, strategy)`` return tuple.

The deterministic-sweep test at the bottom is the curated alternative to
a hypothesis-based property test: it iterates a fixed cartesian product
of wrappers so every failure case is reproducible by its index rather
than via a shrunk counterexample.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from t0_agents.core.structured import permissive_parse


class TestStrictLayer:
    def test_strict_pass_object(self) -> None:
        assert permissive_parse('{"a": 1}') == ({"a": 1}, "strict")

    def test_strict_pass_with_whitespace(self) -> None:
        assert permissive_parse('  {"a": 1}  ') == ({"a": 1}, "strict")

    def test_top_level_array_strict(self) -> None:
        assert permissive_parse("[1, 2, 3]") == ([1, 2, 3], "strict")


class TestPermissiveLayer:
    def test_prose_prefix(self) -> None:
        value, strategy = permissive_parse('Here is your json: {"a": 1}')
        assert value == {"a": 1}
        assert strategy == "permissive"

    def test_prose_suffix(self) -> None:
        value, strategy = permissive_parse('{"a": 1} and that is the result.')
        assert value == {"a": 1}
        assert strategy == "permissive"

    def test_prose_on_both_sides(self) -> None:
        value, strategy = permissive_parse(
            'Sure! Here it is: {"a": 1, "b": [2, 3]} hope that helps.'
        )
        assert value == {"a": 1, "b": [2, 3]}
        assert strategy == "permissive"

    def test_array_with_prose(self) -> None:
        value, strategy = permissive_parse("Here it is: [1, 2, 3] done.")
        assert value == [1, 2, 3]
        assert strategy == "permissive"

    def test_fenced_code_block_json_lang(self) -> None:
        raw = 'See:\n```json\n{"a": 1}\n```'
        value, strategy = permissive_parse(raw)
        assert value == {"a": 1}
        assert strategy == "permissive"

    def test_fenced_code_block_no_lang(self) -> None:
        raw = '```\n{"a": 1}\n```'
        value, strategy = permissive_parse(raw)
        assert value == {"a": 1}
        assert strategy == "permissive"

    def test_fenced_code_block_python_lang_still_works(self) -> None:
        # The opener ``` is followed by an arbitrary language tag — the
        # extractor skips to the next newline regardless of tag content.
        raw = '```python\n{"a": 1}\n```'
        value, strategy = permissive_parse(raw)
        assert value == {"a": 1}
        assert strategy == "permissive"

    def test_braces_inside_string_literal(self) -> None:
        # The balanced-brace scanner must respect string-literal state so
        # that a ``}`` or ``{`` inside a JSON string does not affect the
        # depth counter.  Without that guard, the scanner would close
        # early on the first ``}`` and produce invalid JSON that
        # json.loads would reject.
        raw = '{"text": "}{nested}{"}'
        value, strategy = permissive_parse(raw)
        assert value == {"text": "}{nested}{"}
        assert strategy == "strict"

    def test_braces_inside_string_with_escapes(self) -> None:
        # Escaped quotes within a JSON string must not exit the
        # string-literal state.
        raw = 'Prefix {"text": "he said \\"} {"} ok'
        value, strategy = permissive_parse(raw)
        assert value == {"text": 'he said "} {'}
        assert strategy == "permissive"

    def test_skips_unbalanced_opener_in_prose(self) -> None:
        # A literal ``{`` in prose with no matching closer must not
        # short-circuit the scan — the scanner should advance to the
        # next occurrence and recover the actual JSON.  Models do this
        # in practice, e.g. when describing options before producing
        # the structured answer.
        raw = (
            'Options include {fast, cheap} (no, not those literally); '
            'final answer: {"speed": "fast"}'
        )
        value, strategy = permissive_parse(raw)
        assert value == {"speed": "fast"}
        assert strategy == "permissive"

    def test_recovers_second_fenced_block_when_first_unparseable(self) -> None:
        # Chain-of-thought models commonly emit a reasoning fence before
        # the JSON fence.  When the first fence is not JSON, every
        # subsequent fence is tried in order until one parses.  This
        # constructs the case where the *outer* balanced-brace scan
        # cannot rescue us either: there are no top-level ``{`` or ``[``
        # openers in the raw text, only inside the fences.
        raw = (
            "```\nreasoning text without any braces here\n```\n"
            '```json\n{"a": 1}\n```'
        )
        value, strategy = permissive_parse(raw)
        assert value == {"a": 1}
        assert strategy == "permissive"


class TestNoneLayer:
    def test_invalid_returns_none(self) -> None:
        assert permissive_parse("totally not json") == (None, "none")

    def test_empty_string_returns_none(self) -> None:
        assert permissive_parse("") == (None, "none")

    def test_unbalanced_returns_none(self) -> None:
        assert permissive_parse("{not balanced") == (None, "none")

    def test_only_opener_no_closer(self) -> None:
        assert permissive_parse("prose then { but no closer ever") == (None, "none")

    def test_fenced_block_with_garbage_content(self) -> None:
        raw = "```json\nnot really json\n```"
        assert permissive_parse(raw) == (None, "none")


class TestDeterministicSweep:
    """Curated alternative to a hypothesis-based property test.

    Iterates a deterministic cartesian product of ``(prefix, obj, suffix)``
    to assert that the balanced-brace layer recovers ``obj`` from any
    reasonable English wrapper.  Every failure case is reproducible by
    its index rather than via a shrunk counterexample.
    """

    PREFIXES: ClassVar[list[str]] = [
        "",
        "Here is the JSON: ",
        "Sure! ",
        "Of course, ",
        "I think the answer is ",
        "Here's what you asked for: ",
        "  ",
    ]
    SUFFIXES: ClassVar[list[str]] = [
        "",
        " — hope that helps.",
        " Done.",
        "\n",
        " (let me know if you need more).",
        "  ",
    ]
    OBJECTS: ClassVar[list[Any]] = [
        {"a": 1},
        {"a": 1, "b": 2, "c": 3},
        {"name": "Ada", "age": 36, "languages": ["English", "Greek"]},
        {"nested": {"inner": True, "values": [1, 2, 3]}},
        {"empty_str": "", "empty_list": [], "null": None},
        {"unicode": "café ☕", "russian": "привет"},
    ]

    def test_sweep_objects_with_prose(self) -> None:
        for obj in self.OBJECTS:
            payload = json.dumps(obj)
            for prefix in self.PREFIXES:
                for suffix in self.SUFFIXES:
                    raw = f"{prefix}{payload}{suffix}"
                    value, strategy = permissive_parse(raw)
                    assert value == obj, (
                        f"permissive_parse failed for "
                        f"prefix={prefix!r} obj={obj!r} suffix={suffix!r}: "
                        f"got {value!r} (strategy={strategy})"
                    )
                    assert strategy in {"strict", "permissive"}, (
                        f"unexpected strategy={strategy!r} for "
                        f"prefix={prefix!r} obj={obj!r} suffix={suffix!r}"
                    )

    @pytest.mark.parametrize(
        ("arr", "wrapper_prefix", "wrapper_suffix"),
        [
            ([1, 2, 3], "Items: ", " ok"),
            (["a", "b"], "Tags = ", ""),
            ([{"k": 1}, {"k": 2}], "", " (two records)"),
            ([], "Empty list: ", " — note it's still valid"),
        ],
    )
    def test_sweep_top_level_arrays_with_prose(
        self,
        arr: list[Any],
        wrapper_prefix: str,
        wrapper_suffix: str,
    ) -> None:
        raw = f"{wrapper_prefix}{json.dumps(arr)}{wrapper_suffix}"
        value, strategy = permissive_parse(raw)
        assert value == arr
        assert strategy in {"strict", "permissive"}
