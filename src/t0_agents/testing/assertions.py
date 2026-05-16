"""Shared vocabulary for agent tests.

Every assertion helper takes keyword-only arguments so test readers
know exactly what the check is pinning down, and so a new parameter
(e.g. a tighter failure message) can land without breaking callers.

The tool-call assertions accept **either** a :class:`MockGateway`
(walks its :attr:`MockGateway.responses` for
:class:`~tensorzero.ToolCall` blocks) **or** a :class:`MockToolProvider`
(walks its per-call log).  Use the gateway path to assert what the
*model* emitted; use the tool-provider path to assert what actually
executed after guardrails + the loop got a say.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tensorzero import ChatInferenceResponse, ToolCall

from t0_agents.core.structured import ParseStrategy
from t0_agents.core.types import AgentResult
from t0_agents.testing.mocks import MockGateway, MockToolProvider


def _collect_gateway_tool_calls(gateway: MockGateway) -> list[ToolCall]:
    """Return every :class:`ToolCall` the gateway emitted across all turns.

    JSON-flavoured responses on the gateway are skipped — by contract
    JSON functions never emit tool calls (`Agent._run_json` short-circuits
    the tool loop entirely).  Filtering with ``isinstance`` keeps the
    assertion helpers usable when a test mixes chat and JSON responses
    on the same mock.
    """
    calls: list[ToolCall] = []
    for response in gateway.responses:
        if not isinstance(response, ChatInferenceResponse):
            continue
        calls.extend(b for b in response.content if isinstance(b, ToolCall))
    return calls


def assert_tool_called(
    name: str,
    *,
    gateway: MockGateway | None = None,
    tool_provider: MockToolProvider | None = None,
) -> None:
    """Assert that *name* was called at least once.

    Exactly one of *gateway* or *tool_provider* must be supplied:

    * ``gateway=`` walks :attr:`MockGateway.responses` and looks for
      :class:`ToolCall` content blocks whose ``name`` equals *name* —
      i.e. what the model *emitted*, regardless of whether guardrails
      or the loop short-circuited it.
    * ``tool_provider=`` walks :attr:`MockToolProvider.calls` — i.e.
      what actually *executed* after the full pipeline.
    """
    if (gateway is None) == (tool_provider is None):
        raise ValueError(
            "assert_tool_called: pass exactly one of gateway= or tool_provider="
        )
    if tool_provider is not None:
        matched = [c for c in tool_provider.calls if c["name"] == name]
        assert matched, (
            f"Tool {name!r} was not called. "
            f"Observed calls: {[c['name'] for c in tool_provider.calls]!r}"
        )
        return
    assert gateway is not None  # narrow for mypy
    emitted = _collect_gateway_tool_calls(gateway)
    matched_names = [tc.name for tc in emitted if tc.name == name]
    assert matched_names, (
        f"Gateway never emitted a ToolCall for {name!r}. "
        f"Observed: {[tc.name for tc in emitted]!r}"
    )


def assert_tool_not_called(
    name: str,
    *,
    gateway: MockGateway | None = None,
    tool_provider: MockToolProvider | None = None,
) -> None:
    """Assert that *name* was never called.

    Accepts the same exclusive *gateway* / *tool_provider* pair as
    :func:`assert_tool_called`.
    """
    if (gateway is None) == (tool_provider is None):
        raise ValueError(
            "assert_tool_not_called: pass exactly one of "
            "gateway= or tool_provider="
        )
    if tool_provider is not None:
        matched = [c for c in tool_provider.calls if c["name"] == name]
        assert not matched, (
            f"Tool {name!r} should not have been called, "
            f"but was called {len(matched)} time(s) with arguments: "
            f"{[c['arguments'] for c in matched]!r}"
        )
        return
    assert gateway is not None
    emitted = _collect_gateway_tool_calls(gateway)
    matched_emitted = [tc for tc in emitted if tc.name == name]
    assert not matched_emitted, (
        f"Gateway should not have emitted ToolCall for {name!r}, "
        f"but emitted {len(matched_emitted)} time(s)"
    )


def assert_tool_args_equal(
    name: str,
    expected_args: dict[str, Any],
    *,
    tool_provider: MockToolProvider,
) -> None:
    """Assert that the most recent call to *name* used *expected_args*.

    Only supports the tool-provider path — the gateway path would
    compare against the *parsed* ``ToolCall.arguments`` which can be
    ``None`` when the model emits invalid JSON; the tool-provider path
    guarantees the dict the loop built before dispatch.
    """
    matching = [c for c in tool_provider.calls if c["name"] == name]
    assert matching, (
        f"Tool {name!r} was not called; cannot check arguments. "
        f"Observed calls: {[c['name'] for c in tool_provider.calls]!r}"
    )
    actual = matching[-1]["arguments"]
    assert actual == expected_args, (
        f"Tool {name!r} last-call arguments mismatch.\n"
        f"  expected: {expected_args!r}\n"
        f"  actual:   {actual!r}"
    )


def assert_max_turns(result: AgentResult[Any], *, at_most: int) -> None:
    """Assert that *result* completed in at most *at_most* turns."""
    assert result.turns <= at_most, (
        f"Agent used {result.turns} turn(s); expected at most {at_most}"
    )


def assert_response_contains(result: AgentResult[Any], text: str) -> None:
    """Case-insensitive substring check on ``result.response``."""
    assert text.lower() in result.response.lower(), (
        f"Expected response to contain {text!r}; "
        f"got: {result.response!r}"
    )


# ---------------------------------------------------------------------------
# JSON-flavoured assertions
# ---------------------------------------------------------------------------


def assert_parsed_equals(result: AgentResult[Any], expected: Any) -> None:
    """Assert that ``result.parsed`` equals *expected*.

    Use this in place of ``assert result.parsed == ...`` when you want a
    failure message that includes ``parse_strategy`` (so you can tell at
    a glance whether the discrepancy is a wrong shape or a parse miss).
    """
    assert result.parsed == expected, (
        f"Expected result.parsed to equal {expected!r}; "
        f"got: {result.parsed!r} "
        f"(parse_strategy={result.parse_strategy!r})"
    )


def assert_parsed_matches_schema(
    result: AgentResult[Any], schema: Mapping[str, Any]
) -> None:
    """Validate ``result.parsed`` against a JSON Schema.

    The schema is checked with the ``jsonschema`` package, which is **not**
    a hard dependency of ``t0-agents`` — install it explicitly with
    ``pip install jsonschema`` if you want this helper.  When ``jsonschema``
    is missing the helper raises :class:`ImportError` with an install
    hint; when ``result.parsed`` is ``None`` it raises
    :class:`AssertionError`; when validation fails it re-raises the
    schema error as an :class:`AssertionError`.

    The TensorZero gateway may already validate ``output.parsed`` against
    the function's static schema, but per-call ``output_schema`` overrides
    and the permissive-parsing layer mean ``result.parsed`` can sometimes
    surface a structure that wouldn't pass the original schema — that's
    exactly the case this helper catches.
    """
    try:
        import jsonschema as _jsonschema  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "assert_parsed_matches_schema requires the jsonschema package "
            "(not a hard dependency of t0-agents). "
            "Install with `pip install jsonschema`."
        ) from exc

    assert result.parsed is not None, (
        f"Expected result.parsed to match the schema, "
        f"but parsed is None (parse_strategy={result.parse_strategy!r})"
    )
    try:
        _jsonschema.validate(instance=result.parsed, schema=dict(schema))
    except _jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise AssertionError(
            f"result.parsed failed schema validation at {path}: "
            f"{exc.message} (parsed={result.parsed!r})"
        ) from exc


def assert_invalid_json(result: AgentResult[Any]) -> None:
    """Assert that the JSON run could not produce a parseable result.

    Pins both ``result.parsed is None`` and
    ``result.parse_strategy == "none"`` — the terminal state where every
    layer (strict TZ parse, permissive recovery, retries) failed.
    """
    assert result.parsed is None, (
        f"Expected result.parsed to be None (invalid JSON path); "
        f"got: {result.parsed!r} "
        f"(parse_strategy={result.parse_strategy!r})"
    )
    assert result.parse_strategy == "none", (
        f"Expected parse_strategy='none' (every recovery layer exhausted); "
        f"got: {result.parse_strategy!r}"
    )


def assert_parse_strategy(
    result: AgentResult[Any], strategy: ParseStrategy
) -> None:
    """Assert that ``result.parse_strategy`` equals *strategy*.

    *strategy* is one of ``"strict"``, ``"permissive"``, ``"retry"``,
    ``"none"`` — see :class:`AgentResult` for semantics.
    """
    assert result.parse_strategy == strategy, (
        f"Expected parse_strategy={strategy!r}; "
        f"got: {result.parse_strategy!r}"
    )


def assert_retry_attempts(result: AgentResult[Any], expected: int) -> None:
    """Assert that ``result.json_parse_attempts`` equals *expected*.

    The count is the number of **extra** inferences after the first, so
    ``0`` means "first attempt succeeded" (via TZ or the permissive
    layer), ``1`` means "one retry was needed", etc.
    """
    assert result.json_parse_attempts == expected, (
        f"Expected json_parse_attempts={expected}; "
        f"got: {result.json_parse_attempts}"
    )


__all__ = [
    "assert_invalid_json",
    "assert_max_turns",
    "assert_parse_strategy",
    "assert_parsed_equals",
    "assert_parsed_matches_schema",
    "assert_response_contains",
    "assert_retry_attempts",
    "assert_tool_args_equal",
    "assert_tool_called",
    "assert_tool_not_called",
]
