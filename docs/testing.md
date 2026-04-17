# Testing agents

`t0-agents` ships a first-class testing submodule,
`t0_agents.testing`, that removes the boilerplate of writing tests
against an agent without locking you into a single style. There are
two complementary paths:

- **`MockGateway` + `MockToolProvider`** — deterministic, no
  infrastructure, ideal for unit tests that pin down exactly what the
  agent did on a given turn.
- **`EmbeddedGatewayHelper`** — real TensorZero inference against any
  provider whose API key is in your environment, no Docker required.
  Ideal for integration tests that need realistic model behaviour
  without spinning up containers.

Everything in this document lives under `t0_agents.testing` and is
importable from the base distribution — no extra needed.

## Two test modes

| Question you're answering | Prefer |
| --- | --- |
| Did the agent call tool X with arguments Y? | mocks |
| Does the agent hit `max_turns` on a loop? | mocks |
| Does the agent recover gracefully from a raising tool? | mocks |
| Does the real model understand this prompt? | embedded |
| Does our `tensorzero.toml` variant parse? | embedded |
| Does this provider key actually work? | embedded |

If you have to pick one, **start with mocks**. They fail fast, they're
deterministic, and most bugs land there. Use embedded sparingly and
only when the question is about real-model behaviour.

## Unit tests with `MockGateway`

`MockGateway` duck-types `AsyncTensorZeroGateway`. It supports three
construction modes:

### Script mode — one content list per turn

```python
from t0_agents.testing import MockGateway
from tensorzero import Text, ToolCall

gateway = MockGateway.script([
    # Turn 1: model decides to call a tool
    [ToolCall(
        id="c1", raw_name="search", raw_arguments='{"q": "RLHF"}',
        name="search", arguments={"q": "RLHF"},
    )],
    # Turn 2: model produces final text
    [Text(text="Found 3 papers.")],
])
```

When the script is exhausted, the gateway returns an empty
`Text` turn so the loop terminates cleanly — no hung tests.

### Canned mode — branch on the call

```python
def source(**kwargs):
    if "RLHF" in str(kwargs["input"]):
        return [Text(text="matched RLHF")]
    return [Text(text="no match")]

gateway = MockGateway.canned(source)
```

Ideal when a test wants to branch on what the loop asked for.

### Recording mode — capture real, replay deterministic

```python
recorder = MockGateway.recording(real_gateway)
# ... drive real_gateway through an Agent as usual ...
deterministic = recorder.replay()
# Subsequent tests use `deterministic` — no provider calls.
```

Handy for converting a flaky integration test into a fast unit test
without hand-writing the expected content blocks.

### What's recorded

Every `MockGateway` records two parallel lists:

- `gateway.calls` — the kwargs passed to each `inference(...)` call.
- `gateway.responses` — the `ChatInferenceResponse` the gateway
  returned for each entry.

That pair lets tests inspect both what the *loop asked for* (tool
list, tags, dryrun flag) and what the *model decided* (tool calls,
final text).

## Tool providers with `MockToolProvider`

`MockToolProvider` implements the `ToolProvider` Protocol. Each
registered handler receives the tool arguments as a dict and returns
any JSON-serialisable payload — strings pass through, everything else
is `json.dumps`-ed by the real serializer, so the loop sees the same
bytes a real tool would produce.

```python
from t0_agents.testing import MockToolProvider

tools = MockToolProvider({
    "search_papers": lambda args: ["Paper A", "Paper B"],
    "summarise":     lambda args: f"summary of {args['id']}",
})

# Async handlers are also accepted:
async def fetch(args):
    return await http_client.get(args["url"])

tools = MockToolProvider({"fetch": fetch})
```

Unknown tool names raise `KeyError` immediately. Schema overrides are
supported via `schemas={"tool": {...}}` on the constructor.

## Assertions

Five helpers, all keyword-only, all with clear failure messages:

```python
from t0_agents.testing import (
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_args_equal,
    assert_max_turns,
    assert_response_contains,
)

assert_tool_called("search", tool_provider=tools)
assert_tool_called("search", gateway=gateway)        # or this

assert_tool_not_called("dangerous", tool_provider=tools)

assert_tool_args_equal("search", {"q": "RLHF"}, tool_provider=tools)

assert_max_turns(result, at_most=3)
assert_response_contains(result, "42")
```

`assert_tool_called` and `assert_tool_not_called` accept **either**
`gateway=` (inspects what the model *emitted*) or `tool_provider=`
(inspects what actually *executed* after guardrails and the loop).

## Reusable scenarios

Common loop shapes live in `t0_agents.testing.scenarios`:

```python
from t0_agents.testing import scenarios, MockToolProvider

# Model calls `lookup` once, then answers.
gateway = scenarios.single_tool_happy("lookup", final_text="done")
tools = MockToolProvider({"lookup": lambda args: "42"})

# Model calls `flaky`; tool raises; model answers anyway.
gateway = scenarios.tool_error_recovery("flaky")
tools = MockToolProvider({"flaky": broken_handler})

# Model calls the same tool every turn — exercises MaxTurnsExceededError.
gateway = scenarios.max_turns_exhausted("loop")
```

## Embedded gateway tests

For the questions mocks can't answer, run against real TensorZero
inference — no Docker:

```python
from t0_agents.testing import EmbeddedGatewayHelper

async def test_real_prompt_parses():
    async with EmbeddedGatewayHelper.from_config("tests/configs/minimal.toml") as gw:
        agent = Agent(..., t0=gw, tool_provider=...)
        result = await agent.run("...")
        assert result.response
```

Three entry points:

- `EmbeddedGatewayHelper.from_config(path)` — no backend DB; feedback
  IDs are ephemeral (mirrors `dryrun=True`).
- `EmbeddedGatewayHelper.with_clickhouse(path, url)` — plus ClickHouse
  for tests that assert on inference rows.
- `EmbeddedGatewayHelper.with_postgres(path, url)` — plus Postgres
  for configuration / experimentation assertions.

All three return an `async with` context manager whose value is the
real `AsyncTensorZeroGateway` — no shim.

### Opt in via a pytest marker

Register a custom marker (`pyproject.toml` or the example's
`pytest.ini` does this for you) and skip the marker's tests unless an
env flag is set — keeps contributor CI fast:

```python
import os, pytest

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDED_TESTS") != "1",
        reason="Set RUN_EMBEDDED_TESTS=1 (plus a provider API key) to run",
    ),
]
```

Run locally:

```bash
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest -m embedded
```

## `MockGateway` vs `dryrun=True`

All three of these skip producing real observability rows, but they're
not interchangeable:

| | Real provider call? | Deterministic? | Needs config.toml? |
| --- | --- | --- | --- |
| `MockGateway` | no | yes | no |
| `dryrun=True` over HTTP gateway | yes | no | yes (on the gateway) |
| `EmbeddedGatewayHelper` without backend URLs | yes | no | yes (passed to helper) |

Rule of thumb:

- Use `MockGateway` to pin down *exact* behaviour.
- Use `dryrun=True` during development when you want real model
  reasoning without ClickHouse noise.
- Use `EmbeddedGatewayHelper` in CI for "does the real surface still
  work?" smoke tests.

## Fixtures

Pytest fixtures for the two mocks and the embedded helper:

```python
# conftest.py
pytest_plugins = ["t0_agents.testing.fixtures"]
```

After that:

```python
async def test_thing(mock_t0, mock_tool_provider):
    # mock_t0 is a fresh MockGateway.script([])
    # mock_tool_provider is a fresh MockToolProvider({})
    ...

async def test_real(embedded_t0):
    # Gateway built from tests/configs/minimal.toml; override via
    # @pytest.mark.parametrize("embedded_t0", ["my.toml"], indirect=True)
    ...
```

The `agent_test_factory` fixture builds an `Agent` against the mock
pair with sensible defaults — useful when a test doesn't care about
config boilerplate.

## See also

- Runnable example: [`examples/15_testing/`](../examples/15_testing/README.md).
- Contributing guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- Agent reference: [`docs/agent.md`](agent.md).
