# Phase H Implementation Log

Phase: H (Agent Testing Framework)
Milestone: M8 — Testable Agents
Date: 2026-04-16

## Divergences from Spec

### D1: `_make_call_tool_result` helper never existed — use real `fastmcp.CallToolResult`

**Spec (§4.1 `MockToolProvider.call_tool`)** references an undefined
helper `_make_call_tool_result(result)` to convert a handler's return
value into the object the agent loop expects.

**Actual:** no such helper exists in the codebase.
`fastmcp.Client.call_tool` returns a `fastmcp.client.client.CallToolResult`
dataclass (verified at `.venv/.../fastmcp/client/client.py:113-121`)
with fields `content`, `structured_content`, `meta`, `data`, `is_error`.
`src/t0_agents/mcp/loop.py:122` passes this object directly to
`build_tool_result` in `src/t0_agents/mcp/adapter.py:84`, which
serialises it via `serialize_tool_result` (handling `.is_error`,
`.data` (str / JSON / None), and `.content[*].text` fallback).

**Reason:** The spec was written before Phase A's adapter shipped.
Constructing a real `CallToolResult` inline is simpler than fabricating
an anonymous helper — the mock's output becomes byte-compatible with
the production serializer, which means every edge case the serializer
handles (error payloads, dict→JSON, string passthrough) is exercised
correctly by the mock.

**Impact on the implementation:**
`src/t0_agents/testing/mocks.py:MockToolProvider.call_tool` builds
`CallToolResult(content=[], structured_content=None, meta=None,
data=result, is_error=False)` and returns it.  `MockToolProvider`
remains a runtime-checkable `ToolProvider` match and stays compatible
with any future adapter changes without the mock having to track them.

### D2: Skipped the "pytest-only runtime warning" from §10 risks

**Spec (§10 Risks & mitigations)** suggests a "runtime warning if
`MockGateway` is constructed outside of pytest" to mitigate the risk
of developers shipping mocks to production by accident.

**Actual:** not implemented.  Compensated for with an explicit
module-level docstring on `src/t0_agents/testing/mocks.py` plus an
inline reminder on `MockGateway`'s class docstring: "*Test-only.*
Never use in production — every `inference(...)` returns `self._source`
output, not a real provider call."  Imports from `t0_agents.testing`
also carry the same warning in the package `__init__.py` module
docstring.

**Reason:** A runtime warning that fires unless pytest is imported
into the current process is hostile to legitimate exploratory use
(Jupyter notebooks, REPL debugging, quick scripts).  The test-only
intent is obvious from both the module path (`t0_agents.testing`) and
the class name (`Mock*`).  The risk the warning mitigates — a
developer copy-pasting a `MockGateway` into production — is better
handled by code review than by a noisy runtime check.  The docstrings
are load-bearing here; they appear in every IDE hover and `help()`
call, which is exactly where a developer about to use the class would
look.

### D3: `MockToolProvider.list_tools` yields real `mcp.types.Tool` instances

**Spec (§4.1 `MockToolProvider.list_tools`)** sketches a fabricated
anonymous class via `type("MCPTool", (), {...})()`.

**Actual:** yields real `mcp.types.Tool` instances (a Pydantic model,
validated in `.venv/.../mcp/types.py:1315-1340`).

**Reason:** `mcp.types.Tool` is directly importable, validates its
inputs (rejects schemas without `type`, for example), and matches the
exact shape `t0_agents.mcp.adapter.convert_tools` reads.  The anonymous
class in the spec was a shortcut that would have silently accepted
malformed tools — using the real class catches those at test
construction time.  Same contract, cheaper maintenance: if
`mcp.types.Tool` ever gains a new required field, the mock fails at
construction and points straight at the drift instead of surfacing a
confusing downstream error.

### D4: Added `MockGateway.responses` list (parallel to `.calls`)

**Spec (§4.1)** defines `self.calls: list[dict]` for recording
inference kwargs but does not expose the responses the gateway
returned.

**Actual:** `MockGateway` records both `.calls` (kwargs passed into
`inference`) and `.responses` (the `ChatInferenceResponse` it returned
for each call), as parallel lists with identical indices.

**Reason:** The spec's §4.3 `assert_tool_called(gateway=...)` branch
is implementable only if the gateway retains the responses it emitted
— otherwise there is no way to detect a `ToolCall` the model asked
for.  Adding `.responses` is the minimal fix; it also makes recording
mode easier to reason about (the capture is just `zip(calls,
responses)`).

**Impact:** `assertions.py::assert_tool_called` and
`assert_tool_not_called` now have working `gateway=` paths.  Tests
that only care about "what actually executed" still use the
`tool_provider=` path; tests that want "what the model emitted"
(including calls the loop may have filtered) use `gateway=`.
Documented the distinction in `docs/testing.md`.

## Open questions resolved during the phase

T0 Q7 (embedded gateway) was already source-validated before Phase H
started and drove the phase's dual-path design (MockGateway +
EmbeddedGatewayHelper).  No additional open questions surfaced during
implementation; `02-open-questions.md` is unchanged.

## Contract amendments

None.  Every decision in the phase sits inside the latitude
`00-integration-contract.md` already grants:

- Testing submodule stays in the base package (contract §3.2
  constraint: "testing utilities should be importable without
  installing extras") — no new extra added.
- No modification to `mcp/`, `core/`, or any prior-phase public
  surface (contract §6.1–§6.3).
- Async-first preserved (contract §2.6).

## Tests

`tests/test_testing/` ships seven test files (56 tests total) plus a
minimal `tests/configs/minimal.toml` config for the embedded tests:

- `test_mock_gateway.py` — script / canned / recording modes, episode-id
  stability, kwargs recording, feedback path.
- `test_mock_tool_provider.py` — sync + async handlers, unknown-tool
  `KeyError`, timeout acceptance, serializer compatibility.
- `test_assertions.py` — every helper's pass + fail branches, both
  `gateway=` and `tool_provider=` paths on the exclusive helpers.
- `test_scenarios.py` — each scenario drives a real `Agent` end-to-end
  with a matching `MockToolProvider`; terminal states (tool called /
  max turns exceeded) asserted.
- `test_mock_shape.py` — **contract test** against
  `inspect.signature(AsyncTensorZeroGateway.inference)`,
  `isinstance(..., ToolProvider)`, and
  `isinstance(CallToolResult, …)`.  Guards against future drift.
- `test_streaming_mocks.py` — `fake_event_stream` round-trip,
  `MockStreamingGateway` chunk shape, `stream=False` rejection.
- `test_embedded.py` — behind `@pytest.mark.embedded` + skipif on
  `RUN_EMBEDDED_TESTS != "1"`.

All offline tests (`-m "not integration and not embedded"`) green:
455 passed (56 new + 399 pre-existing), 44 deselected.

Example `examples/15_testing/test_agent_unit.py` runs in 0.7 s on the
author's laptop — well under the 5 s acceptance criterion §9.2.

## M8 release-notes snippet

> **`t0-agents 0.9.0` — Testable Agents.**
> Ship faster with `from t0_agents.testing import MockGateway,
> MockToolProvider, EmbeddedGatewayHelper, assert_tool_called`.  Unit
> tests run in milliseconds against a fully mocked TensorZero + MCP
> surface; integration tests run against real TensorZero inference
> without Docker via the embedded-gateway helper.  Three canned
> scenarios (`single_tool_happy`, `tool_error_recovery`,
> `max_turns_exhausted`) and five assertion helpers cover the
> overwhelming majority of agent-test shapes.  See the new
> [`docs/testing.md`](../../testing.md) and
> [`examples/15_testing/`](../../../examples/15_testing/) for a full
> walkthrough.

## Milestone gate (§3.1 of `01-implementation-plan.md`)

- [x] Example lands in `examples/15_testing/` and runs end-to-end
      (unit path: 4 tests passing in 0.7 s; embedded path: opt-in via
      `RUN_EMBEDDED_TESTS=1`).
- [x] Public doc page lands at
      [`docs/testing.md`](../../testing.md).
- [x] `CHANGELOG.md` entry in the established format under
      `[0.9.0] — Unreleased`.
- [x] Developer-guide §4 updated with the `t0_agents.testing` symbol
      table.
- [x] `CONTRIBUTING.md` created (spec §7 required "`CONTRIBUTING.md`
      updates"; file did not previously exist).

Phase H closes Milestone M8, which closes the Agent-abstraction
roadmap.  The whole `from t0_agents import …` public surface is
feature-complete against the eight phases A–H.  A `1.0.0` bump is a
reasonable next step after a soak period; shipping at `0.9.0` here
preserves the option.
