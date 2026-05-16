"""Unit tests for ``MockGateway.script_json`` / ``canned_json`` and the
JSON-flavoured assertion helpers.

The chat-flavoured surface stays covered by ``test_mock_gateway.py`` and
``test_assertions.py``; this module pins the JSON branch so downstream
consumers can write JSON tests without rolling their own ``FakeJsonT0``
fakes.
"""

from __future__ import annotations

import sys
from typing import Any
from uuid import uuid4

import pytest
from tensorzero.types import (
    JsonInferenceOutput,
    JsonInferenceResponse,
    Usage,
)

from t0_agents.core.types import AgentResult
from t0_agents.testing import (
    MockGateway,
    assert_invalid_json,
    assert_parse_strategy,
    assert_parsed_equals,
    assert_parsed_matches_schema,
    assert_retry_attempts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_result(
    *,
    parsed: Any = None,
    parse_strategy: str = "strict",
    json_parse_attempts: int = 0,
    response: str = '{"hello": "world"}',
) -> AgentResult[Any]:
    """Build an :class:`AgentResult` shaped like a JSON run for assert tests.

    The assert helpers only read four fields — ``parsed``,
    ``parse_strategy``, ``json_parse_attempts``, and (transitively)
    ``response`` for error messages — so hand-building is much faster
    than spinning up a full :class:`~t0_agents.core.agent.Agent` for
    every assertion test.

    Post-M3: ``AgentResult.value`` is the canonical field;
    ``response``/``parsed``/``parse_strategy``/``json_parse_attempts``/
    ``function_kind`` are read-side ``@property`` aliases.  This helper
    builds the value from ``parsed`` (when set) or from ``response``
    (chat fallback), and stamps the v0.12 read-side semantics via
    :class:`OutputDiagnostics`.
    """
    from t0_agents.core.output import OutputDiagnostics

    # JSON shape — ``value`` is the parsed dict; ``response`` lives on
    # diagnostics.raw_text so the v0.12 read-side property returns it.
    return AgentResult(
        value=parsed,
        diagnostics=OutputDiagnostics(
            parse_strategy=parse_strategy,  # type: ignore[arg-type]
            attempts=json_parse_attempts,
            raw_text=response,
        ),
    )


# ---------------------------------------------------------------------------
# MockGateway.script_json / canned_json
# ---------------------------------------------------------------------------


class TestScriptedJsonGateway:
    async def test_script_json_basic(self) -> None:
        """Dict with explicit ``raw`` and ``parsed`` round-trips verbatim."""
        gateway = MockGateway.script_json(
            [
                {
                    "raw": '{"name": "Ada"}',
                    "parsed": {"name": "Ada"},
                }
            ]
        )

        response = await gateway.inference(
            function_name="extract", input={"messages": []}
        )

        assert isinstance(response, JsonInferenceResponse)
        assert response.output.raw == '{"name": "Ada"}'
        assert response.output.parsed == {"name": "Ada"}

    async def test_script_json_missing_fields_defaulted(self) -> None:
        """Missing keys default to sensible values; sticky episode_id holds."""
        gateway = MockGateway.script_json(
            [
                {"raw": '{"a": 1}'},
                {"raw": '{"b": 2}'},
            ]
        )

        first = await gateway.inference(function_name="f")
        second = await gateway.inference(function_name="f")

        # episode_id sticks across turns (matches the real gateway).
        assert isinstance(first, JsonInferenceResponse)
        assert isinstance(second, JsonInferenceResponse)
        assert first.episode_id == second.episode_id
        # inference_ids are unique.
        assert first.inference_id != second.inference_id
        # usage defaults but is present.
        assert isinstance(first.usage, Usage)
        assert first.usage.input_tokens == 0
        # variant_name default.
        assert first.variant_name == "mock_variant"

    async def test_script_json_auto_derives_parsed(self) -> None:
        """When the ``parsed`` key is absent, ``raw`` is best-effort parsed.

        Happy-path: scripting a single ``{"raw": ...}`` yields a
        ``JsonInferenceResponse`` whose ``parsed`` is the JSON object
        decoded from ``raw``.
        """
        gateway = MockGateway.script_json(
            [
                {"raw": '{"derived": true}'},
                {"raw": "not json at all"},
            ]
        )

        ok = await gateway.inference(function_name="f")
        bad = await gateway.inference(function_name="f")

        assert isinstance(ok, JsonInferenceResponse)
        assert ok.output.parsed == {"derived": True}
        assert isinstance(bad, JsonInferenceResponse)
        # Unparseable raw → None (no exception).
        assert bad.output.parsed is None

    async def test_script_json_explicit_parsed_none_wins(self) -> None:
        """Explicit ``parsed: None`` is respected even when ``raw`` parses.

        This is the retry-script semantic the orchestrator relies on: a
        script entry whose raw is structurally valid JSON but whose
        ``parsed`` is ``None`` because TZ returned ``parsed = null``
        (the permissive layer then kicks in upstream).  Auto-derive
        would mask the case.
        """
        gateway = MockGateway.script_json(
            [
                {
                    "raw": '{"valid": "json"}',
                    "parsed": None,
                }
            ]
        )

        response = await gateway.inference(function_name="f")
        assert isinstance(response, JsonInferenceResponse)
        # Explicit None survives.
        assert response.output.parsed is None
        assert response.output.raw == '{"valid": "json"}'

    async def test_canned_json_receives_kwargs(self) -> None:
        """``canned_json`` callback observes the full inference kwargs."""
        captured: dict[str, Any] = {}

        def source(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"raw": '{"ok": true}'}

        gateway = MockGateway.canned_json(source)

        response = await gateway.inference(
            function_name="extract",
            input={"messages": [{"role": "user", "content": "hi"}]},
            output_schema={"type": "object"},
            params={"chat_completion": {"json_mode": "strict"}},
        )

        assert isinstance(response, JsonInferenceResponse)
        # Verify the callback saw every kwarg we passed.
        assert captured["function_name"] == "extract"
        assert captured["output_schema"] == {"type": "object"}
        assert captured["params"] == {
            "chat_completion": {"json_mode": "strict"}
        }
        # And kwargs are also recorded on the gateway for assertion helpers.
        assert gateway.calls[0]["function_name"] == "extract"
        assert gateway.calls[0]["output_schema"] == {"type": "object"}

    async def test_script_json_passes_through_prebuilt(self) -> None:
        """Pre-built :class:`JsonInferenceResponse` entries pass through."""
        prebuilt = JsonInferenceResponse(
            inference_id=uuid4(),
            episode_id=uuid4(),
            variant_name="custom-variant",
            output=JsonInferenceOutput(
                raw='{"x": 1}', parsed={"x": 1}
            ),
            usage=Usage(input_tokens=42, output_tokens=7),
        )

        gateway = MockGateway.script_json([prebuilt])
        response = await gateway.inference(function_name="f")

        assert response is prebuilt
        assert response.usage.input_tokens == 42

    async def test_script_json_raises_when_exhausted(self) -> None:
        """An exhausted JSON script raises ``AssertionError`` (no graceful turn)."""
        gateway = MockGateway.script_json([{"raw": '{"x": 1}'}])
        await gateway.inference(function_name="f")

        with pytest.raises(AssertionError, match="ran out of canned responses"):
            await gateway.inference(function_name="f")


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


class TestJsonAssertHelpers:
    def test_assert_parsed_equals_pass_fail(self) -> None:
        passing = _json_result(parsed={"a": 1})
        assert_parsed_equals(passing, {"a": 1})  # no raise

        failing = _json_result(parsed={"a": 1})
        with pytest.raises(
            AssertionError, match=r"Expected result\.parsed to equal"
        ):
            assert_parsed_equals(failing, {"a": 2})

    def test_assert_parsed_matches_schema_pass(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        passing = _json_result(parsed={"name": "Ada"})
        assert_parsed_matches_schema(passing, schema)  # no raise

        # Negative path: missing required field.
        failing = _json_result(parsed={})
        with pytest.raises(
            AssertionError, match="failed schema validation"
        ):
            assert_parsed_matches_schema(failing, schema)

        # Negative path: parsed is None.
        none_result = _json_result(parsed=None, parse_strategy="none")
        with pytest.raises(AssertionError, match="parsed is None"):
            assert_parsed_matches_schema(none_result, schema)

    def test_assert_parsed_matches_schema_missing_dep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``jsonschema`` isn't importable, the helper raises ``ImportError``.

        Setting ``sys.modules["jsonschema"]`` to ``None`` makes a fresh
        ``import jsonschema`` raise ``ImportError`` — the canonical way
        Python tests simulate a missing optional dependency.
        """
        monkeypatch.setitem(sys.modules, "jsonschema", None)

        with pytest.raises(
            ImportError, match="pip install jsonschema"
        ):
            assert_parsed_matches_schema(
                _json_result(parsed={"x": 1}),
                {"type": "object"},
            )

    def test_assert_invalid_json_pass_fail(self) -> None:
        # Pass: parsed is None AND parse_strategy is "none".
        passing = _json_result(parsed=None, parse_strategy="none")
        assert_invalid_json(passing)

        # Fail: parsed is not None.
        failing_with_parsed = _json_result(
            parsed={"a": 1}, parse_strategy="strict"
        )
        with pytest.raises(
            AssertionError, match=r"Expected result\.parsed to be None"
        ):
            assert_invalid_json(failing_with_parsed)

        # Fail: parsed is None but strategy is not "none" (impossible in
        # real orchestrator code, but the helper should still pin the
        # contract).
        failing_strategy = _json_result(parsed=None, parse_strategy="strict")
        with pytest.raises(AssertionError, match="parse_strategy='none'"):
            assert_invalid_json(failing_strategy)

    def test_assert_parse_strategy_and_retry_attempts(self) -> None:
        result = _json_result(
            parsed={"ok": True},
            parse_strategy="retry",
            json_parse_attempts=2,
        )

        assert_parse_strategy(result, "retry")
        assert_retry_attempts(result, 2)

        with pytest.raises(
            AssertionError, match="Expected parse_strategy='strict'"
        ):
            assert_parse_strategy(result, "strict")

        with pytest.raises(
            AssertionError, match="Expected json_parse_attempts=5"
        ):
            assert_retry_attempts(result, 5)
