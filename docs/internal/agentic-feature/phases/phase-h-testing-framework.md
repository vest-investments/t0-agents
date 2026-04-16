# Phase H — Agent Testing Framework

> Status: **ready to start**. Phase deps: A–G complete. Open question
> resolved: T0 Q7 — `AsyncTensorZeroGateway.build_embedded(...)` IS
> available in v2026.4.0 (`tensorzero.pyi:1017`); all backend-DB URLs
> are `Optional`. **Plan revised**: ship BOTH `MockGateway` (deterministic
> unit tests) AND `EmbeddedGatewayHelper` (real T0 inference in CI
> without Docker).
> Delivers: **Milestone M8 — Testable Agents**.

**Related**

- [`../00-integration-contract.md`](../00-integration-contract.md)
- [`../02-open-questions.md`](../02-open-questions.md) — T0 Q7 (corrected)
- [`../problemas.md`](../problemas.md) — Problem 7
- All earlier phases.

---

## 1. Goal

Give developers building on `t0-agents` a first-class testing
framework with two complementary tools:

- **`MockGateway`** — pure-Python mock of `AsyncTensorZeroGateway`.
  Deterministic, zero-infra. Best for unit tests asserting exact LLM
  behaviour, tool-usage assertions, error injection.
- **`EmbeddedGatewayHelper`** — convenience fixture around
  `AsyncTensorZeroGateway.build_embedded(...)`. Real T0 inference
  against any provider (whose API key is in env), no ClickHouse, no
  Docker. Best for integration tests that exercise actual gateway
  behaviour but don't need observability writes.

After Phase H:

```python
# Unit test — deterministic
from t0_agents.testing import MockGateway, MockToolProvider, assert_tool_called
from tensorzero import Text, ToolCall

async def test_research_agent_uses_search():
    gateway = MockGateway.script([
        # Turn 1: model decides to call search_papers
        [ToolCall(name="search_papers", arguments={"q": "RLHF"}, id="c1", raw_name="search_papers", raw_arguments='{"q":"RLHF"}')],
        # Turn 2: model produces final text
        [Text(text="Found 3 papers on RLHF.")],
    ])
    tools = MockToolProvider({
        "search_papers": lambda args: "Paper A; Paper B; Paper C",
    })
    agent = Agent(AgentConfig(name="researcher", function_name="research"),
                  t0=gateway, tool_provider=tools)
    result = await agent.run("What is RLHF?")
    assert_tool_called(result, "search_papers")
    assert "RLHF" in result.response

# Integration test — real T0, no Docker
from t0_agents.testing import EmbeddedGatewayHelper

@pytest.fixture
async def t0_embedded():
    async with EmbeddedGatewayHelper.from_config("tests/configs/test.toml") as t0:
        yield t0

async def test_real_agent_against_provider(t0_embedded):
    agent = Agent(AgentConfig(...), t0=t0_embedded, tool_provider=...)
    result = await agent.run("Hello")
    assert result.response
```

---

## 2. Scope

### 2.1 In scope

- New submodule `src/t0_agents/testing/` with:
  - `__init__.py` — public exports.
  - `mocks.py` — `MockGateway`, `MockChatResponse`, `MockToolProvider`.
  - `embedded.py` — `EmbeddedGatewayHelper` wrapping `build_embedded`.
  - `assertions.py` — `assert_tool_called`, `assert_tool_not_called`,
    `assert_tool_args_equal`, `assert_max_turns`,
    `assert_response_contains`.
  - `fixtures.py` — pytest fixtures: `mock_t0`, `mock_tool_provider`,
    `embedded_t0`, `agent_test_factory`.
  - `scenarios.py` — canned scenarios: `single_tool_happy`,
    `tool_error_recovery`, `max_turns_exhausted`.
  - `streaming_mocks.py` — chunk fakes for testing `Agent.stream()`.
- Tests under `tests/test_testing/` (yes, tests for the testing
  framework — meta but important).
- Example `examples/15_testing/` showing both modes.
- Public doc page `docs/testing.md`.
- `CONTRIBUTING.md` updates documenting both test paths.

### 2.2 Out of scope

- Snapshot-testing infrastructure beyond `inline-snapshot` (already
  in dev deps). Don't ship our own.
- Performance / load testing helpers. Different problem domain.
- A test runner that discovers agents automatically. Use pytest's
  normal discovery; agents are just Python objects.

### 2.3 Non-goals

- Do not ship a "production fake" gateway. `MockGateway` is for tests
  only; `EmbeddedGatewayHelper` is documented as a test helper, not
  a production deployment path.
- Do not duplicate `dryrun=True` semantics. The two are complementary:
  - `dryrun=True` against an HTTP gateway: skip ClickHouse writes
    but keep real provider calls.
  - `MockGateway`: skip provider calls.
  - `EmbeddedGatewayHelper` without backend URLs: real provider, no
    observability — closest to `dryrun=True` over HTTP.

---

## 3. Constraints (from the contract)

1. `mcp/` and earlier-phase tests stay green. Adding `testing/` does
   not modify any production module.
2. `testing/` is in the base package (no extra) — testing utilities
   should be importable without installing extras. Internal use of
   pytest helpers is allowed; pytest itself is a dev dep.
3. `MockGateway` satisfies the same Protocol shape that `Agent`
   expects from `AsyncTensorZeroGateway`. It is duck-typed.
4. `EmbeddedGatewayHelper` returns the real
   `AsyncTensorZeroGateway` instance — no shim.

---

## 4. Module additions

### 4.1 `src/t0_agents/testing/mocks.py` (new)

```python
from collections import deque
from typing import Any, Callable

from tensorzero.types import (
    ChatInferenceResponse, ContentBlock, Usage,
)


class MockChatResponse:
    """Builder for a single ChatInferenceResponse-shaped object."""

    def __init__(self, content: list[ContentBlock], **overrides):
        self.content = content
        self.overrides = overrides

    def to_response(self, *, inference_id="mock-inference", episode_id="mock-episode"):
        return ChatInferenceResponse(
            inference_id=inference_id,
            episode_id=episode_id,
            variant_name="mock_variant",
            content=self.content,
            usage=self.overrides.get("usage", Usage(input_tokens=0, output_tokens=0)),
            finish_reason=self.overrides.get("finish_reason"),
        )


class MockGateway:
    """A duck-typed AsyncTensorZeroGateway for unit tests.

    Three construction patterns:

    1. `MockGateway.script([turn1_content, turn2_content, ...])`
       → returns content lists in order, one per inference call.

    2. `MockGateway.canned(callable)`
       → callable receives (function_name, input, **kwargs) and returns
         a content list. Useful for behaviour-driven tests.

    3. `MockGateway.recording(real_gateway)`
       → records every inference + its response; replay later. Helps
         convert ad-hoc integration tests into deterministic ones.
    """

    def __init__(self, response_source):
        self._source = response_source
        self.calls: list[dict] = []

    @classmethod
    def script(cls, turns: list[list[ContentBlock]]) -> "MockGateway":
        queue = deque(turns)
        def source(**kwargs):
            return queue.popleft() if queue else [Text(text="")]
        return cls(source)

    @classmethod
    def canned(cls, fn: Callable[..., list[ContentBlock]]) -> "MockGateway":
        return cls(fn)

    async def inference(self, **kwargs):
        self.calls.append(kwargs)
        content = self._source(**kwargs)
        if isinstance(content, list):
            return MockChatResponse(content).to_response(
                inference_id=f"mock-inference-{len(self.calls)}",
                episode_id=kwargs.get("episode_id") or "mock-episode-1",
            )
        # If the source returned a fully-built response, pass it through
        return content

    async def feedback(self, **kwargs):
        self.calls.append({"feedback": kwargs})
        return type("FeedbackResponse", (), {"feedback_id": f"mock-fb-{len(self.calls)}"})()


class MockToolProvider:
    """A dict-backed ToolProvider for unit tests.

    Keys are tool names; values are sync or async callables that take
    arguments dict and return any value (str, dict, list, …) — the
    framework's serializer turns them into the ToolResult string.
    """

    def __init__(self, tools: dict[str, Callable]):
        self._tools = tools
        self.calls: list[dict] = []

    async def list_tools(self):
        return [
            type("MCPTool", (), {
                "name": n,
                "description": "",
                "inputSchema": {"type": "object", "properties": {}},
            })()
            for n in self._tools
        ]

    async def call_tool(self, name, arguments, *, timeout=None):
        self.calls.append({"name": name, "arguments": arguments})
        handler = self._tools.get(name)
        if handler is None:
            raise KeyError(f"Mock tool {name!r} not registered")
        result = handler(arguments) if not asyncio.iscoroutinefunction(handler) else await handler(arguments)
        return _make_call_tool_result(result)
```

### 4.2 `src/t0_agents/testing/embedded.py` (new)

```python
import os
from contextlib import asynccontextmanager

from tensorzero import AsyncTensorZeroGateway


class EmbeddedGatewayHelper:
    """Convenience wrapper around AsyncTensorZeroGateway.build_embedded
    for integration tests that don't need observability.

    Exposes:
        - `from_config(config_file_path)` async context manager
        - `with_clickhouse(...)`, `with_postgres(...)` opt-in for
          tests that DO need observability (e.g. assertions on
          ClickHouse rows).

    Source-validated: build_embedded accepts all backend URLs as
    Optional (tensorzero/tensorzero.pyi:1017-1035). Running with no
    backend URL gives degraded mode — feedback IDs are ephemeral,
    similar to dryrun=True.
    """

    @classmethod
    @asynccontextmanager
    async def from_config(cls, config_file: str):
        gw = await AsyncTensorZeroGateway.build_embedded(
            config_file=config_file,
        )
        async with gw:
            yield gw

    @classmethod
    @asynccontextmanager
    async def with_clickhouse(cls, config_file: str, clickhouse_url: str):
        gw = await AsyncTensorZeroGateway.build_embedded(
            config_file=config_file,
            clickhouse_url=clickhouse_url,
        )
        async with gw:
            yield gw
```

### 4.3 `src/t0_agents/testing/assertions.py` (new)

```python
from t0_agents.core.types import AgentResult


def assert_tool_called(
    *,
    gateway: "MockGateway" = None,
    tool_provider: "MockToolProvider" = None,
    name: str,
):
    """Either gateway (matched by inspecting recorded inference responses)
    or tool_provider (matched by recorded calls) must show the tool was
    called.
    """
    if tool_provider is not None:
        called = any(c["name"] == name for c in tool_provider.calls)
        assert called, f"Tool {name!r} was not called. Calls: {[c['name'] for c in tool_provider.calls]}"
        return
    raise ValueError("Provide gateway= or tool_provider=")


def assert_tool_not_called(*, tool_provider, name):
    called = any(c["name"] == name for c in tool_provider.calls)
    assert not called, f"Tool {name!r} should not have been called."


def assert_tool_args_equal(*, tool_provider, name, expected_args):
    matching = [c for c in tool_provider.calls if c["name"] == name]
    assert matching, f"Tool {name!r} was not called."
    assert matching[-1]["arguments"] == expected_args


def assert_max_turns(result: AgentResult, *, at_most: int):
    assert result.turns <= at_most


def assert_response_contains(result: AgentResult, *, text: str):
    assert text.lower() in result.response.lower()
```

### 4.4 `src/t0_agents/testing/fixtures.py` (new)

```python
import pytest

from t0_agents.testing.mocks import MockGateway, MockToolProvider
from t0_agents.testing.embedded import EmbeddedGatewayHelper


@pytest.fixture
def mock_t0():
    return MockGateway.script([])


@pytest.fixture
def mock_tool_provider():
    return MockToolProvider({})


@pytest.fixture
async def embedded_t0(request):
    config_file = getattr(request, "param", "tests/configs/test.toml")
    async with EmbeddedGatewayHelper.from_config(config_file) as gw:
        yield gw
```

### 4.5 `src/t0_agents/testing/scenarios.py` (new)

```python
from tensorzero.types import Text, ToolCall


def single_tool_happy(tool_name: str = "do_thing", final_text: str = "done") -> MockGateway:
    return MockGateway.script([
        [ToolCall(name=tool_name, arguments={}, id="c1",
                  raw_name=tool_name, raw_arguments="{}")],
        [Text(text=final_text)],
    ])


def tool_error_recovery(tool_name: str = "flaky") -> MockGateway:
    """First the model calls a tool that errors; then the model recovers
    by responding with text instead of retrying.
    """
    return MockGateway.script([
        [ToolCall(name=tool_name, arguments={}, id="c1",
                  raw_name=tool_name, raw_arguments="{}")],
        [Text(text="The tool failed; I'll answer from memory.")],
    ])


def max_turns_exhausted(tool_name: str = "loop") -> MockGateway:
    """The model never stops calling the same tool — used to test
    MaxTurnsExceededError.
    """
    def source(**kwargs):
        return [ToolCall(name=tool_name, arguments={}, id="c-loop",
                         raw_name=tool_name, raw_arguments="{}")]
    return MockGateway.canned(source)
```

### 4.6 `src/t0_agents/testing/__init__.py`

```python
from t0_agents.testing.assertions import (
    assert_max_turns,
    assert_response_contains,
    assert_tool_args_equal,
    assert_tool_called,
    assert_tool_not_called,
)
from t0_agents.testing.embedded import EmbeddedGatewayHelper
from t0_agents.testing.fixtures import embedded_t0, mock_t0, mock_tool_provider
from t0_agents.testing.mocks import (
    MockChatResponse, MockGateway, MockToolProvider,
)
from t0_agents.testing import scenarios

__all__ = [
    "EmbeddedGatewayHelper",
    "MockChatResponse",
    "MockGateway",
    "MockToolProvider",
    "assert_max_turns",
    "assert_response_contains",
    "assert_tool_args_equal",
    "assert_tool_called",
    "assert_tool_not_called",
    "embedded_t0",
    "mock_t0",
    "mock_tool_provider",
    "scenarios",
]
```

---

## 5. Tests

Location: `tests/test_testing/`.

### 5.1 Unit (offline)

- [ ] `test_mock_gateway_script_mode` — script returns turns in order;
      raises after exhaustion.
- [ ] `test_mock_gateway_canned_mode` — callable receives kwargs.
- [ ] `test_mock_gateway_recording_mode` — drives a `MockGateway` to
      record then replay.
- [ ] `test_mock_tool_provider_dispatches` — sync + async handlers.
- [ ] `test_mock_tool_provider_unknown_tool` — `KeyError` with helpful
      message.
- [ ] `test_assert_tool_called_pass_and_fail`.
- [ ] `test_assertions_for_other_helpers`.
- [ ] `test_scenarios_single_tool_happy_drives_agent` — orchestrate
      `Agent` with the scenario's `MockGateway` + a matching
      `MockToolProvider`; assert end state.

### 5.2 Embedded gateway tests (require provider API key in env)

Marked `@pytest.mark.embedded` (a new pytest marker — registered in
`pyproject.toml`):

- [ ] `test_embedded_gateway_starts` — `EmbeddedGatewayHelper.from_config`
      yields a usable gateway; `.inference()` succeeds.
- [ ] `test_embedded_no_backend_dryrun_like` — feedback returns IDs
      that are ephemeral (passes if no backend; documented behaviour).

CI skips `embedded` markers unless `RUN_EMBEDDED_TESTS=1` is set, to
keep contributor laptops fast.

### 5.3 Snapshots

- [ ] One snapshot test for `MockChatResponse.to_response` shape, using
      `inline-snapshot` (already a dev dep).

### 5.4 Regression

- [ ] All earlier phases' tests pass; nothing in `mcp/` changes.

---

## 6. Example

Location: `examples/15_testing/`.

- `agent.py` — small agent using a tiny FastMCP server.
- `test_agent_unit.py` — uses `MockGateway` + `MockToolProvider` +
  scenarios + assertions. CI-friendly, no Docker, no API keys.
- `test_agent_embedded.py` — uses `EmbeddedGatewayHelper` (skipped
  unless `RUN_EMBEDDED_TESTS=1`). Real Groq inference.
- `pytest.ini` — registers the `embedded` marker.
- `README.md` — when to use which mode; how to set up the embedded
  test path.

---

## 7. Documentation

- New: `docs/testing.md`.
- Sections:
  - "Two test modes" — when to pick mocks vs embedded.
  - "Writing unit tests with `MockGateway`" — script vs canned vs
    recording.
  - "Writing assertions" — every helper's signature.
  - "Reusable scenarios" — `single_tool_happy`, etc.
  - "Embedded gateway tests" — how to bring your own provider API
    key, how to opt in via env flag.
  - "What about `dryrun=True`?" — comparison table.
- Update `developer-guide.md` §4 (public API) — `t0_agents.testing`
  surface.
- Update `CONTRIBUTING.md` — explain when to write unit-only vs
  embedded vs Docker-integration tests.
- `CHANGELOG.md` — `0.9.0` (or `1.0.0` if Phase H concludes the
  Agent abstraction work).

---

## 8. Checklist

### 8.1 Code

- [ ] `src/t0_agents/testing/__init__.py`.
- [ ] `src/t0_agents/testing/mocks.py`.
- [ ] `src/t0_agents/testing/embedded.py`.
- [ ] `src/t0_agents/testing/assertions.py`.
- [ ] `src/t0_agents/testing/fixtures.py`.
- [ ] `src/t0_agents/testing/scenarios.py`.
- [ ] `src/t0_agents/testing/streaming_mocks.py`.
- [ ] `pyproject.toml` — register `embedded` marker.
- [ ] `uv run ruff check src/`, `uv run mypy src/` green.

### 8.2 Tests

- [ ] All §5.1 unit tests.
- [ ] §5.2 embedded tests (run locally with `RUN_EMBEDDED_TESTS=1`).
- [ ] Snapshot test green.

### 8.3 Example

- [ ] `examples/15_testing/` complete with both unit and embedded test
      files.

### 8.4 Docs

- [ ] `docs/testing.md`.
- [ ] `developer-guide.md` updates.
- [ ] `CONTRIBUTING.md` updates.
- [ ] `CHANGELOG.md`.

### 8.5 Log

- [ ] `phase-h-implementation-log.md` with the T0 Q7 correction
      narrative (build_embedded was actually available — file a doc
      PR upstream as a follow-up), and the M8 release-notes snippet.

---

## 9. Acceptance criteria (Milestone M8)

1. `from t0_agents.testing import MockGateway, MockToolProvider,
   EmbeddedGatewayHelper, assert_tool_called` works.
2. `examples/15_testing/test_agent_unit.py` runs in `<5s` with
   `pytest`, no external services.
3. `examples/15_testing/test_agent_embedded.py` runs successfully when
   `RUN_EMBEDDED_TESTS=1` and a provider key is set, against
   `build_embedded`, with no Docker required.
4. The roadmap's earlier phases A–G have green test suites with the
   same assertion helpers (refactor opt-in, not required).
5. `docs/testing.md` published; CONTRIBUTING.md guides new contributors
   through both modes.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| `MockGateway`'s shape drifts from real `AsyncTensorZeroGateway` | medium | One contract-test in `tests/test_testing/test_mock_shape.py` that imports both and asserts attribute compatibility. |
| Developers ship `MockGateway` to production by accident | low | Module-level docstring + a runtime warning if `MockGateway` is constructed outside of pytest. |
| `build_embedded` requires a working tensorzero.toml — friction for trivial tests | medium | Provide `tests/configs/minimal.toml` template; ship a fixture that uses it by default. |
| Embedded tests are flaky due to provider rate limits | medium | Embedded tests are opt-in (env flag); document the trade-off. |

---

## 11. Estimated surface

- **New files:** 7 src + 8 test + 5 example + 1 doc.
- **Modified files:** 3 (`pyproject.toml`, `CONTRIBUTING.md`,
  `CHANGELOG.md`).
- **Net new public symbols:** ~12 (`MockGateway`, `MockChatResponse`,
  `MockToolProvider`, `EmbeddedGatewayHelper`, the assertion helpers,
  `scenarios.*`, fixtures).
- **New deps:** 0.
