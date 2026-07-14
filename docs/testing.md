# Testing

You have built, composed, served, and driven agents interactively. This chapter is about keeping them from regressing. `t0_agents.testing` removes the boilerplate of writing tests for an `Agent` without locking you into a single style: a deterministic mock path for unit tests (no Docker, no API keys) and an embedded-gateway path for integration tests against real TensorZero inference.

A first-class test-authoring surface for agents built on `t0-agents`.

`t0_agents.testing` removes the boilerplate of writing tests for an
`Agent` without locking you into a single style. Three complementary
paths:

- **`MockGateway` + `MockToolProvider`** — deterministic, no
  infrastructure, ideal for unit tests that pin down *exactly* what
  the agent did on a given turn.
- **`ScriptedAgent`** — a whole-`Agent` fake whose `invoke()` replays
  a fixed event script + result with no gateway at all. Ideal when the
  unit under test consumes `invoke() -> Run` (a glow app, a `serve`
  route, a TUI) and you want to script the *events*, not the gateway.
- **`EmbeddedGatewayHelper`** — real TensorZero inference against any
  provider whose API key is in your environment, no Docker required.
  Ideal for integration tests that need realistic model behaviour
  without spinning up containers.

Everything in this document lives under `t0_agents.testing` and is
importable from the base distribution — no extra needed.

```python
from t0_agents.testing import (
    MockGateway, MockChatResponse, MockToolProvider, RecordingGateway,
    MockStreamingGateway, fake_event_stream, ScriptedAgent,
    EmbeddedGatewayHelper,
    assert_tool_called, assert_tool_not_called, assert_tool_args_equal,
    assert_max_turns, assert_response_contains,
    scenarios, streaming_mocks,
    mock_t0, mock_tool_provider, embedded_t0, agent_test_factory,
    AgentFactory, ToolHandler,
)
```

**Test-only.** Every symbol in this module bypasses real inference or
tool dispatch by design. Never import `t0_agents.testing` from a
production code path — the module path and the `Mock*` prefixes make
the intent obvious, and the mocks' docstrings restate the constraint
everywhere it's relevant.

---

## Installation

The whole module ships in the base distribution:

```bash
pip install t0-agents
# or
uv pip install t0-agents
```

No new dependency, no new extra — the contract that governs this
library requires testing utilities to be importable without installing
extras, so consumers can add tests on day one. Pytest is a dev
dependency of *your* project; `t0_agents.testing.fixtures` uses it but
does not re-declare it.

Optional prerequisites for the embedded path:

- A provider API key in the environment that matches the `tensorzero.toml`
  you hand to `EmbeddedGatewayHelper` (Groq's `GROQ_API_KEY` is the
  canonical default across the library's examples).
- `RUN_EMBEDDED_TESTS=1` as the opt-in flag for CI runs that want to
  execute the embedded suite. Matches the `embedded` pytest marker
  registered in `pyproject.toml`.

---

## Quickstart — unit test with mocks (no Docker, no API keys) — Easy

Script the turns, register a mock tool, assert on what the loop
actually executed. Runs in milliseconds.

```python
from typing import cast

from tensorzero import AsyncTensorZeroGateway, Text, ToolCall

from t0_agents import Agent, AgentConfig
from t0_agents.core.types import ToolProvider
from t0_agents.testing import (
    MockGateway,
    MockToolProvider,
    assert_response_contains,
    assert_tool_args_equal,
    assert_tool_called,
)


async def test_agent_uses_search_and_summarises() -> None:
    gateway = MockGateway.script(
        [
            # Turn 1: model decides to call search_notes
            [ToolCall(
                id="c1",
                raw_name="search_notes",
                raw_arguments='{"query": "groceries"}',
                name="search_notes",
                arguments={"query": "groceries"},
            )],
            # Turn 2: model produces final text from the tool result
            [Text(text="Found 2 notes: milk, eggs.")],
        ]
    )
    tools = MockToolProvider({
        "search_notes": lambda args: ["milk", "eggs"],
    })

    agent = Agent(
        AgentConfig(name="assistant", function_name="assistant"),
        t0=cast(AsyncTensorZeroGateway, gateway),
        tool_provider=cast(ToolProvider, tools),
    )
    result = await agent.invoke("Find notes about groceries")

    assert_tool_called("search_notes", tool_provider=tools)
    assert_tool_args_equal(
        "search_notes", {"query": "groceries"}, tool_provider=tools,
    )
    assert_response_contains(result, "milk")
```

A runnable version of this exact test — with three tools, all five
assertion helpers, and an error-recovery scenario — lives at
[`examples/agent/11_testing/test_agent_unit.py`](../examples/agent/11_testing/test_agent_unit.py).

---

## Quickstart — integration test against real inference (API key required) — Advanced

When a test has to answer a question mocks can't answer — "does this
prompt actually work?" / "does my `tensorzero.toml` parse?" / "does
this provider key have access to this model?" — swap `MockGateway` for
`EmbeddedGatewayHelper` and keep everything else the same.

```python
import os

import pytest
from fastmcp import Client, FastMCP

from t0_agents import Agent, AgentConfig
from t0_agents.testing import EmbeddedGatewayHelper


pytestmark = [
    pytest.mark.embedded,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDED_TESTS") != "1",
        reason="Set RUN_EMBEDDED_TESTS=1 (plus a provider API key) to run.",
    ),
]


async def test_real_prompt_answers_question() -> None:
    mcp = FastMCP("demo")
    async with (
        EmbeddedGatewayHelper.from_config("tests/configs/minimal.toml") as t0,
        Client(mcp) as client,
    ):
        agent = Agent(
            AgentConfig(name="assistant", function_name="assistant"),
            t0=t0,
            tool_provider=client,
        )
        result = await agent.invoke("What is 2 + 2?")
        assert result.response
```

Run it once via the marker:

```bash
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest -m embedded
```

A runnable version lives at
[`examples/agent/11_testing/test_agent_embedded.py`](../examples/agent/11_testing/test_agent_embedded.py).

---

## Concepts

### Two complementary surfaces

| Question you're answering | Prefer |
|---|---|
| Did the agent call tool X with arguments Y? | mocks |
| Does the agent hit `max_turns` on a loop? | mocks |
| Does the agent recover gracefully from a raising tool? | mocks |
| Does the real model understand this prompt? | embedded |
| Does our `tensorzero.toml` variant actually parse? | embedded |
| Does this provider key actually work? | embedded |

If you have to pick one, **start with mocks.** They fail fast, they're
deterministic, and most bugs land there. Use embedded sparingly —
only when the question is about real-model behaviour.

### Lifecycle of a mocked turn

```
agent.invoke(message)
   │
   ├─ 1. Infer           MockGateway.inference(**kwargs)
   │                         ├─ deep-copy kwargs  ────►  gateway.calls
   │                         ├─ source(**kwargs)
   │                         └─ record + return ChatInferenceResponse
   │                                                ─►  gateway.responses
   │
   ├─ 2. Tools           MockToolProvider.call_tool(name, arguments)
   │                         ├─ record ──►  tool_provider.calls
   │                         ├─ dispatch to handler (sync or async)
   │                         └─ wrap return in fastmcp.CallToolResult
   │                            (pass-through if handler pre-built one)
   │
   ├─ 3. Serialise       serialize_tool_result(CallToolResult)
   │                     (the production serializer — same path a
   │                      real tool would take)
   │
   └─ 4. Next turn       Back to step 1 with the updated message list
                         until the model produces text-only content.
```

Two invariants anchor the whole design:

1. **Duck-typing, not subclassing.** `MockGateway` and
   `MockToolProvider` match the real types structurally: they expose
   `inference` and `list_tools` / `call_tool` with compatible
   signatures, and the agent loop sees no difference. Subclassing
   would have coupled the mocks to the real SDK's class layout; the
   structural path lets the mocks stay thin.
2. **Real `fastmcp.CallToolResult`, always.** `MockToolProvider` wraps
   handler return values in the same dataclass a real FastMCP client
   returns, and the production
   [`serialize_tool_result`](../src/t0_agents/mcp/adapter.py) runs
   over it unchanged. That means every edge case the serializer
   handles — error payloads, dict→JSON, string passthrough — is
   exercised by the mock exactly the way a real tool would exercise it.

A shape-contract test at
[`tests/test_testing/test_mock_shape.py`](../tests/test_testing/test_mock_shape.py)
asserts `MockGateway` accepts every kwarg the agent loop passes,
`MockToolProvider` satisfies the runtime-checkable `ToolProvider`
Protocol, and `MockToolProvider.call_tool` returns a real
`fastmcp.CallToolResult`. If you extend the real surface — a new
inference kwarg, a new ToolProvider method — the shape-contract test
fails fast and points at the mock.

### Why the embedded path exists

TensorZero's Python SDK ships
[`AsyncTensorZeroGateway.build_embedded`](https://www.tensorzero.com/docs/gateway/deployment/embedded/):
a classmethod that builds an in-process gateway with optional backend
URLs. Call it with just a config file and you get real provider
inference, no observability backend, no HTTP listener — exactly the
sweet spot for tests that want the model's behaviour without the
ClickHouse overhead. `EmbeddedGatewayHelper` is the async-context
wrapper that keeps the build / teardown dance out of your tests.

---

## `MockGateway`

`MockGateway` duck-types
[`AsyncTensorZeroGateway`](https://www.tensorzero.com/docs/gateway/).
Three construction modes, pick the one that matches the question:

### Easy — script mode

One content list per turn, popped left-to-right.

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
`Text` turn so the loop terminates cleanly — no hung tests. Pass a
pre-chosen `episode_id=` (`UUID`) if you want assertions that span
`episode_id` equality across turns; otherwise a stable UUID is
generated on the first turn and reused for the rest of the run.

### Intermediate — canned mode

Supply a callable that receives the inference kwargs and returns
either a content list or a fully built `ChatInferenceResponse`.

```python
def source(**kwargs):
    if "RLHF" in str(kwargs["input"]):
        return [Text(text="matched RLHF")]
    return [Text(text="no match")]

gateway = MockGateway.canned(source)
```

Ideal when a test wants to branch on what the loop asked for — for
example, simulating a model that only emits a tool call when the
system prompt changes, or one whose response depends on a tag.

### Advanced — recording + replay

Wrap a real gateway in a `RecordingGateway`, drive it through an
`Agent` as usual, then snap a deterministic `MockGateway` from the
capture.

```python
from t0_agents.testing import MockGateway

async with await AsyncTensorZeroGateway.build_http(...) as real:
    recorder = MockGateway.recording(real)
    agent = Agent(config, t0=recorder, tool_provider=client)
    await agent.invoke("…")

# After the run — build a deterministic MockGateway from what was captured.
deterministic = recorder.replay()
# Save / load as you see fit; the replayed gateway carries no network
# dependency and can be reused across many tests.
```

`MockGateway.recording(real_gateway)` returns a `RecordingGateway`
(a gateway-shaped object in its own right, forwarding every call to
`real_gateway` and capturing the `(kwargs, response)` pair). It is
exported as a top-level symbol; use it directly if you prefer
composing a recorder around a gateway you already own.

### What's recorded

Every `MockGateway` and every `RecordingGateway` exposes three
parallel lists:

| Field | Contents |
|---|---|
| `.calls` | `dict` of the kwargs the agent loop passed to `inference` — `function_name`, `input`, `additional_tools`, `tool_choice`, `tags`, `namespace`, `dryrun`, `episode_id`. |
| `.responses` | The `ChatInferenceResponse` the gateway returned for the matching call. Inspect `.content` to see what the *model* emitted (tool calls, text). |
| `.feedback_calls` | `dict` of the kwargs passed to `feedback(...)`. Separate from `.calls` so inference- and feedback-side assertions don't interfere. |

Two tests can ask complementary questions:

```python
# What did the loop ask for?
assert gateway.calls[0]["tags"] == {"env": "test"}

# What did the model decide?
from tensorzero import ToolCall
first_turn = gateway.responses[0].content
assert any(isinstance(b, ToolCall) for b in first_turn)
```

**Snapshot semantics.** `kwargs` are deep-copied at call time before
being appended to `.calls`. Agent loops reuse the same
`input["messages"]` list across turns and mutate it in place with
`messages.append(...)` after each inference — without the deep copy,
every `gateway.calls[i]["input"]["messages"]` entry would end up
pointing at the same mutating list, and multi-turn assertions like
`len(gateway.calls[0]["input"]["messages"]) == 1` would always see
the final state. With the deep copy, each entry is a true snapshot of
what the loop passed on that turn.

### Testing JSON agents

`MockGateway` also covers the JSON dispatch path used by agents whose
`AgentConfig.function_kind == "json"` — see
[Structured outputs](structured-outputs.md#testing-your-json-agent) for
the canonical reference.  The script-mode pattern mirrors `script()`:

```python
from t0_agents.testing import MockGateway

gateway = MockGateway.script_json([
    # parsed auto-derived from raw when the key is absent.
    {"raw": '{"name": "Ada", "email": "ada@example.com"}'},
])
```

Dict entries accept `raw`, `parsed`, `episode_id`, `inference_id`,
`usage`, `variant_name`, and `finish_reason` (all optional except
`raw`).  Pre-built `JsonInferenceResponse` instances pass through
unchanged.  When `parsed` is omitted, the gateway runs a best-effort
`json.loads(raw)` and falls back to `None` on failure; passing
`parsed=None` explicitly preserves the "TZ returned the raw but
couldn't parse" semantic used by the retry tests.

`canned_json(fn)` is the callable-driven sibling — `fn` receives the
inference kwargs and returns either a dict or a pre-built
`JsonInferenceResponse`.  Useful when the response must branch on the
caller's per-call `output_schema` / `json_mode` overrides.



Five JSON-aware assertion helpers complement the chat ones — see the
[Assertions](#assertions) section below.

---

## `MockToolProvider`

`MockToolProvider` implements the `ToolProvider` Protocol. Each
registered handler receives the tool arguments as a dict and returns
any JSON-serialisable payload — strings pass through, everything else
is `json.dumps`-ed by the production serializer, so the loop sees the
same bytes a real tool would produce.

### Dict-backed handlers

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

Unknown tool names raise `KeyError` immediately with a message that
lists the registered tools — so the test surfaces the
misconfiguration instead of silently succeeding.

### Custom schemas

By default, every tool is listed with a no-argument schema
(`{"type": "object", "properties": {}}`). Pass `schemas=` when the
agent's prompt logic depends on the tool's JSON schema:

```python
tools = MockToolProvider(
    {"lookup": lambda args: args["id"]},
    schemas={
        "lookup": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
            "required": ["id"],
        },
    },
)
```

### Simulating tool errors — Advanced

For the fast path, raise from the handler — the agent loop catches
the exception and hands the LLM an `"ERROR: …"` string on the next
turn:

```python
def broken(_args):
    raise RuntimeError("disk is full")

tools = MockToolProvider({"save_note": broken})
```

When a test needs to exercise the production serializer's explicit
`is_error=True` branch — for example, to pin down the exact bytes the
LLM sees when the tool reports an error status rather than raising —
return a prebuilt `fastmcp.client.client.CallToolResult`. The
serializer reads the error message from `content[*].text`, so the
realistic shape is a `mcp.types.TextContent` block:

```python
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

def save_note(_args):
    return CallToolResult(
        content=[TextContent(type="text", text="disk is full")],
        structured_content=None,
        meta=None,
        data=None,
        is_error=True,
    )

tools = MockToolProvider({"save_note": save_note})
```

`MockToolProvider` detects a prebuilt `CallToolResult` and passes it
through verbatim — wrapping it again would nest the object inside a
new result's `.data` field and the serializer would `json.dumps` the
dataclass instead of surfacing the error string.  The agent loop
then sees a `ToolResult(result="ERROR: disk is full")` on the next
turn, ready for the model to react to.  See
[`examples/agent/11_testing/test_agent_unit.py::test_agent_surfaces_structured_tool_error`](../examples/agent/11_testing/test_agent_unit.py)
for a runnable end-to-end test of this pattern.

---

## Assertions

Ten helpers in two families — five chat-flavoured ones for tool-call
and response shape, five JSON-flavoured ones for the `parsed` /
`parse_strategy` / `json_parse_attempts` fields the structured-outputs
surface exposes.  The tool-name arguments are positional; the source
selector (`gateway=` / `tool_provider=`) is keyword-only.

### Chat assertions

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

`assert_tool_called` and `assert_tool_not_called` accept **exactly
one** of `gateway=` or `tool_provider=` — passing both or neither
raises `ValueError`. The two paths answer different questions:

- **`gateway=`** walks `MockGateway.responses` for `ToolCall` content
  blocks — i.e. what the *model emitted*, regardless of whether
  guardrails or the loop short-circuited the call.
- **`tool_provider=`** walks `MockToolProvider.calls` — i.e. what
  actually *executed* after the full pipeline ran.

`assert_tool_args_equal` only accepts the `tool_provider=` path: the
gateway-side `ToolCall.arguments` is the *parsed* dict which can be
`None` when the model emits invalid JSON, so the tool-provider's
pre-dispatch snapshot is the authoritative record of what the tool
actually saw.

All chat helpers raise `AssertionError` on failure with messages that
name the observed state (the list of tools called, the last-call
arguments, the observed turn count, the actual response text).

### JSON assertions

```python
from t0_agents.testing import (
    assert_parsed_equals,
    assert_parsed_matches_schema,
    assert_invalid_json,
    assert_parse_strategy,
    assert_retry_attempts,
)

assert_parsed_equals(result, {"name": "Ada"})
assert_parse_strategy(result, "retry")        # "strict" / "permissive" / "retry" / "none"
assert_retry_attempts(result, 1)
assert_invalid_json(result)                   # parsed is None AND parse_strategy == "none"
assert_parsed_matches_schema(result, {"type": "object", "required": ["name"]})
```

`assert_parsed_matches_schema` soft-imports `jsonschema` inside the
helper — it is **not** a hard dependency of t0-agents.  When the
package is missing the helper raises `ImportError` with the install
hint `pip install jsonschema`.  All five helpers raise `AssertionError`
on failure with messages that include the observed `parse_strategy`
so a passing test that suddenly degrades from `"strict"` to `"retry"`
surfaces clearly.

See [Structured outputs — Testing your JSON agent](structured-outputs.md#testing-your-json-agent)
for the full worked example.

---

## Reusable scenarios

`t0_agents.testing.scenarios` ships three canned `MockGateway`
factories for the common loop shapes:

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

Each factory returns a fresh `MockGateway`, so reusing one across
multiple tests in the same file is safe. Pair each scenario with a
matching `MockToolProvider` — the scenario picks the tool name; the
provider picks the behaviour.

---

## Streaming mocks

`t0_agents.testing.streaming_mocks` covers the two streaming shapes
you'll want to test without spinning up a real gateway.

### `fake_event_stream` — event iterators

Wrap any iterable of `AgentEvent` into an `AsyncIterator[AgentEvent]`:

```python
from t0_agents import TextDeltaEvent, DoneEvent, AgentResult
from t0_agents.testing import fake_event_stream

events = [
    TextDeltaEvent(id="t1", delta="Hello"),
    TextDeltaEvent(id="t1", delta=" world"),
    DoneEvent(result=AgentResult(
        response="Hello world", episode_id="ep", inference_ids=[], turns=1,
    )),
]

async for event in fake_event_stream(events):
    ...  # drive a TUI renderer, an SSE route, a WS bridge
```

Perfect for testing anything that consumes `Run.events()` — it lets
the consumer handle each event type without the overhead of a real
gateway and a real `StreamingAgentLoop`.

### `MockStreamingGateway` — stream-capable gateway

A tiny `AsyncTensorZeroGateway` duck for driving
`Run.events()` through a full turn loop:

```python
from t0_agents.testing import MockStreamingGateway
from tensorzero import Text

gateway = MockStreamingGateway([
    [Text(text="Hello world")],
])
```

Each turn emits a single `ChatChunk` carrying the chunk counterparts
of the scripted blocks, followed by a terminal `Usage`-only chunk.
Enough to exercise `Run.events()`'s turn loop and tool-dispatch
path deterministically.

Calls with `stream=False` are rejected with `ValueError` so
misconfigurations surface immediately — use `MockGateway` for
non-streaming tests. On script exhaustion, `MockStreamingGateway`
emits `[Text(text="")]` — the same empty-turn contract `MockGateway`
uses — so assistant-history assertions see the same shape across
streaming and non-streaming modes and a switch between them never
leaks into a `content=[]` vs `content=[Text(text="")]` surprise.

`MockStreamingGateway` does not reproduce TensorZero's real
delta-level streaming (one chunk per content block, not one per
text fragment). Tests that need delta-level assertions should drive
`t0_agents.stream.assembler.TurnStreamAssembler` directly with
hand-written chunks instead.

---

## `ScriptedAgent` — a whole-agent fake

`MockGateway` fakes the *gateway* a real `Agent` talks to.
`ScriptedAgent` fakes the *`Agent` itself*: its `invoke()` replays a
fixed event script and result with no gateway — no Docker, no API keys,
not even a `MockGateway`.

```python
from t0_agents.stream.events import TextDeltaEvent, TextEndEvent
from t0_agents.testing import ScriptedAgent

agent = ScriptedAgent(
    result="It's 22C in Tokyo.",
    events=[
        TextDeltaEvent(id="t1", delta="It's 22C in Tokyo."),
        TextEndEvent(id="t1"),
    ],
)

run = agent.invoke("weather in Tokyo?")
assert (await run).response == "It's 22C in Tokyo."
async for event in run.events():
    ...  # the scripted events, then a synthesised terminal DoneEvent
```

- **`events=`** are the non-terminal events to replay; the single
  terminal `DoneEvent` is synthesised from **`result=`** (pass an
  `AgentResult` to control `episode_id` / `turns`, or any value to
  wrap).
- **`raises=`** makes the run fail: the exception propagates out of
  both `result()` and `events()` exactly as a real fatal error would
  (no `ErrorEvent` — that event is for recoverable, inline anomalies).

### Swapping a glow agent by name — `app.agent_overrides`

`ScriptedAgent` pairs with glow's `App.agent_overrides`, the
agent-level mirror of FastAPI's `dependency_overrides`: register a fake
by name and it replaces the real agent wherever glow resolves it.
Because a fully-overridden `App` never builds a real `Agent`, it needs
no gateway — construct it with `App()`:

```python
from t0_agents.glow import App, llm
from t0_agents.testing import ScriptedAgent

app = App()  # no t0 — the only agent is overridden below

@app.agent("forecaster")
async def forecaster(query: str) -> str:
    """Forecast the weather."""
    return llm()

app.agent_overrides["forecaster"] = ScriptedAgent(result="sunny")

assert await forecaster("weather?") == "sunny"
```

The swapped agent's events flow through `FluentRun.events()` and its
selectors unchanged, so a test can assert on the exact event stream a
glow consumer would see. An un-overridden agent on a gateway-less
`App` raises `ConfigurationError` — the fake path is explicit, never
silent.

---

## Embedded gateway

For the questions mocks can't answer, run against real TensorZero
inference — no Docker:

```python
from t0_agents.testing import EmbeddedGatewayHelper

async def test_real_prompt_parses():
    async with EmbeddedGatewayHelper.from_config("tests/configs/minimal.toml") as gw:
        agent = Agent(..., t0=gw, tool_provider=...)
        result = await agent.invoke("...")
        assert result.response
```

Three entry points:

- `EmbeddedGatewayHelper.from_config(config_file)` — no backend DB;
  feedback IDs are ephemeral (mirrors `dryrun=True`).
- `EmbeddedGatewayHelper.with_clickhouse(config_file, clickhouse_url)` —
  plus ClickHouse for tests that assert on inference rows.
- `EmbeddedGatewayHelper.with_postgres(config_file, postgres_url)` —
  plus Postgres for configuration / experimentation assertions.

All three are `@staticmethod @asynccontextmanager` — you `async with`
them and the yielded value is the real `AsyncTensorZeroGateway`, no
shim. The helper owns the build and teardown; the test owns
everything else.

### Opting in via a pytest marker

Register the `embedded` marker in `pyproject.toml` (the repo's
top-level `pyproject.toml` already does this) and skip the marker's
tests unless `RUN_EMBEDDED_TESTS=1` is set — keeps contributor CI
fast:

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

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full three-marker
strategy (`integration` vs `embedded` vs default) the library uses
for its own suite.

---

## `MockGateway` vs `dryrun=True`

All three of these skip producing real observability rows, but
they're not interchangeable:

| | Real provider call? | Deterministic? | Needs `tensorzero.toml`? |
|---|---|---|---|
| `MockGateway` | no | yes | no |
| `dryrun=True` over an HTTP gateway | yes | no | yes (on the gateway) |
| `EmbeddedGatewayHelper` (no backend URLs) | yes | no | yes (passed to helper) |

Rule of thumb:

- Use `MockGateway` to pin down *exact* behaviour.
- Use `dryrun=True` during development when you want real model
  reasoning without ClickHouse noise.
- Use `EmbeddedGatewayHelper` in CI for "does the real surface still
  work?" smoke tests.

---

## Fixtures

Four fixtures ship under `t0_agents.testing.fixtures`. Register the
whole module as a pytest plugin from your `conftest.py`:

```python
# conftest.py
pytest_plugins = ["t0_agents.testing.fixtures"]
```

After that:

| Fixture | Default | Shape |
|---|---|---|
| `mock_t0` | `MockGateway.script([])` | A fresh `MockGateway`. Tests typically reassign before passing it to an `Agent`, or use `scenarios.*` instead. |
| `mock_tool_provider` | `MockToolProvider({})` | A fresh `MockToolProvider`. Tests register tools inline. |
| `embedded_t0` | `tests/configs/minimal.toml` | An async `AsyncTensorZeroGateway` built via `EmbeddedGatewayHelper.from_config`. Parametrize with `@pytest.mark.parametrize("embedded_t0", ["my.toml"], indirect=True)` to override the config path. |
| `agent_test_factory` | — | Zero-config factory that builds an `Agent` against `mock_t0` + `mock_tool_provider`. Defaults: `name="test"`, `function_name="test-function"`, `max_turns=5`. Override either dependency by passing `t0=` / `tool_provider=` explicitly; override any `AgentConfig` field by passing it as a keyword. |

```python
from t0_agents.testing import MockGateway, scenarios

async def test_thing(mock_t0, mock_tool_provider):
    # mock_t0 is a fresh MockGateway.script([])
    # mock_tool_provider is a fresh MockToolProvider({})
    ...

async def test_happy_path(agent_test_factory):
    # factory signature: factory(*, t0=None, tool_provider=None, **config_overrides)
    agent = agent_test_factory(
        t0=scenarios.single_tool_happy("lookup"),
        max_turns=10,
    )
    ...

async def test_real(embedded_t0):
    # Gateway built from tests/configs/minimal.toml; override via indirect parametrize.
    ...
```

The factory's return type is `AgentFactory = Callable[..., Agent]`,
exported alongside the fixtures in case your own helper wants to
accept a factory-shaped dependency.

---

## API reference

All symbols below live in `t0_agents.testing` and are listed in
`__all__`. Signatures are taken from the module itself and verified
against the shape-contract test — when the real surface extends, this
table is what the test protects.

### Mocks

| Symbol | Signature | Purpose |
|---|---|---|
| `MockGateway` | `MockGateway(source, *, episode_id: UUID \| None = None)` | Duck-typed `AsyncTensorZeroGateway` for unit tests. |
| `MockGateway.script` | `classmethod script(turns: list[list[ContentBlock]], *, episode_id: UUID \| None = None) -> MockGateway` | Queue-driven responses, one content list per turn. Empty `Text` on exhaustion. |
| `MockGateway.canned` | `classmethod canned(fn: Callable[..., list[ContentBlock] \| ChatInferenceResponse], *, episode_id: UUID \| None = None) -> MockGateway` | Callable-driven responses; receives full inference kwargs. |
| `MockGateway.recording` | `classmethod recording(real_gateway) -> RecordingGateway` | Wrap a real gateway to capture `(kwargs, response)` pairs. |
| `MockGateway.inference` | `async def inference(**kwargs) -> ChatInferenceResponse` | Records kwargs (deep-copied) on `.calls`, appends response to `.responses`. |
| `MockGateway.feedback` | `async def feedback(**kwargs) -> FeedbackResponse` | Records kwargs on `.feedback_calls`; returns a response with an ephemeral `feedback_id`. |
| `MockGateway.calls` | `list[dict[str, Any]]` | Deep-copied snapshots of inference kwargs. |
| `MockGateway.responses` | `list[ChatInferenceResponse]` | Responses emitted for each entry in `.calls`. |
| `MockGateway.feedback_calls` | `list[dict[str, Any]]` | Kwargs passed to `.feedback(...)`. |
| `RecordingGateway` | `RecordingGateway(real_gateway)` | Gateway wrapper that forwards to `real_gateway` and captures the exchange. |
| `RecordingGateway.captured` | `property -> list[_RecordedInference]` | Read-only view of the `(kwargs, response)` pairs. |
| `RecordingGateway.replay` | `replay() -> MockGateway` | Build a deterministic `MockGateway` from the capture. |
| `MockChatResponse` | `MockChatResponse(content, *, usage=None, variant_name="mock_variant", finish_reason=None)` | `@dataclass(slots=True)` builder for a `ChatInferenceResponse`. |
| `MockChatResponse.to_response` | `to_response(*, inference_id: UUID \| None = None, episode_id: UUID \| None = None) -> ChatInferenceResponse` | Materialise a response; generates stable UUIDs if omitted. |
| `MockToolProvider` | `MockToolProvider(tools: dict[str, ToolHandler], *, schemas: dict[str, dict] \| None = None)` | Dict-backed `ToolProvider` Protocol implementation. |
| `MockToolProvider.list_tools` | `async def list_tools() -> list[mcp.types.Tool]` | Yields real `mcp.types.Tool` instances. |
| `MockToolProvider.call_tool` | `async def call_tool(name: str, arguments: dict, *, timeout: float \| None = None) -> CallToolResult` | Dispatches to the handler; passes prebuilt `CallToolResult` through unchanged. |
| `MockToolProvider.calls` | `list[dict[str, Any]]` | Per-dispatch log of `{"name": ..., "arguments": ...}`. |
| `ToolHandler` | `TypeAlias = Callable[[dict[str, Any]], Any] \| Callable[[dict[str, Any]], Awaitable[Any]]` | Sync or async handler return type. |
| `ScriptedAgent` | `ScriptedAgent(*, result: Any = None, events: Sequence[AgentEvent] \| None = None, raises: BaseException \| None = None)` | Whole-`Agent` fake; `invoke()` replays a scripted `Run` (events + terminal, or `raises=`) with no gateway. |

### Assertions

| Symbol | Signature |
|---|---|
| `assert_tool_called` | `assert_tool_called(name: str, *, gateway: MockGateway \| None = None, tool_provider: MockToolProvider \| None = None) -> None` |
| `assert_tool_not_called` | `assert_tool_not_called(name: str, *, gateway: MockGateway \| None = None, tool_provider: MockToolProvider \| None = None) -> None` |
| `assert_tool_args_equal` | `assert_tool_args_equal(name: str, expected_args: dict, *, tool_provider: MockToolProvider) -> None` |
| `assert_max_turns` | `assert_max_turns(result: AgentResult, *, at_most: int) -> None` |
| `assert_response_contains` | `assert_response_contains(result: AgentResult, text: str) -> None` |

### Scenarios — `t0_agents.testing.scenarios`

| Factory | Signature |
|---|---|
| `single_tool_happy` | `single_tool_happy(tool_name: str = "do_thing", *, final_text: str = "done") -> MockGateway` |
| `tool_error_recovery` | `tool_error_recovery(tool_name: str = "flaky", *, final_text: str = "I couldn't run the tool; answering from memory.") -> MockGateway` |
| `max_turns_exhausted` | `max_turns_exhausted(tool_name: str = "loop") -> MockGateway` |

### Embedded

| Symbol | Signature |
|---|---|
| `EmbeddedGatewayHelper.from_config` | `@staticmethod @asynccontextmanager async def from_config(config_file: str) -> AsyncIterator[AsyncTensorZeroGateway]` |
| `EmbeddedGatewayHelper.with_clickhouse` | `@staticmethod @asynccontextmanager async def with_clickhouse(config_file: str, clickhouse_url: str) -> AsyncIterator[AsyncTensorZeroGateway]` |
| `EmbeddedGatewayHelper.with_postgres` | `@staticmethod @asynccontextmanager async def with_postgres(config_file: str, postgres_url: str) -> AsyncIterator[AsyncTensorZeroGateway]` |

### Streaming mocks — `t0_agents.testing.streaming_mocks`

| Symbol | Signature |
|---|---|
| `fake_event_stream` | `async def fake_event_stream(events: Iterable[AgentEvent]) -> AsyncIterator[AgentEvent]` |
| `MockStreamingGateway` | `MockStreamingGateway(turns: list[list[ContentBlock]], *, episode_id: UUID \| None = None)` |
| `MockStreamingGateway.inference` | `async def inference(*, stream: bool = False, **kwargs) -> AsyncIterator[ChatChunk]` (raises `ValueError` on `stream=False`) |

### Fixtures

All four are `@pytest.fixture`, importable directly from
`t0_agents.testing` or via `pytest_plugins = ["t0_agents.testing.fixtures"]`.

| Fixture | Default value |
|---|---|
| `mock_t0` | `MockGateway.script([])` |
| `mock_tool_provider` | `MockToolProvider({})` |
| `embedded_t0` | `AsyncTensorZeroGateway` from `tests/configs/minimal.toml` (override via indirect parametrize) |
| `agent_test_factory` | Callable `factory(*, t0=None, tool_provider=None, **config_overrides) -> Agent`, `AgentFactory` type alias exported. |

---

## CI / test reproducibility

### Marker strategy

Two opt-in markers are registered in `pyproject.toml`:

- **`integration`** — requires `docker compose up -d` and a real
  gateway. Deselect with `-m "not integration"`.
- **`embedded`** — requires `RUN_EMBEDDED_TESTS=1` and a provider
  API key. Deselect with `-m "not embedded"`.

Tests without a marker are the default gate — offline, deterministic,
always runnable. This is the contract the library's own suite
follows, and it's the one the `embedded`-marker snippet in the
[Opting in via a pytest marker](#opting-in-via-a-pytest-marker)
section above assumes.

### Fast-CI path

The default-gate suite runs in a few seconds end-to-end:

```bash
uv run pytest -m "not integration and not embedded"
```

This is the line contributor PRs must keep green. The library's own
CI uses the same invocation.

### Shape-contract guard

[`tests/test_testing/test_mock_shape.py`](../tests/test_testing/test_mock_shape.py)
locks `MockGateway`, `MockToolProvider`, and the assertions against
drift from the real `AsyncTensorZeroGateway.inference` signature and
the `ToolProvider` / `CallToolResult` types. When you add a kwarg to
the agent loop, add a method to `ToolProvider`, or upgrade the
TensorZero SDK to a version that changes the inference signature,
this test fails first and points at the mock to update — before any
downstream test has a chance to start producing confusing failures.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full three-mode
story on when each marker is appropriate, plus the exact commands
the library's own gate pipelines run.

---

## Observability

Testing is largely the absence of observability writes — the mocks
fabricate IDs so tests can assert on them without talking to a
backend. The specific behaviours:

- **`MockGateway`** — `inference_id` and `episode_id` on every
  response are freshly-generated `UUID`s; the episode id is sticky
  across turns within the same gateway instance, matching the real
  gateway's contract. `feedback()` returns a response with an
  ephemeral `feedback_id`.
- **`EmbeddedGatewayHelper.from_config`** — runs the gateway in
  degraded mode: real provider calls, but no ClickHouse / Postgres /
  Valkey — so the IDs and usage numbers the gateway returns reflect
  the real provider while feedback rows are ephemeral (mirroring
  `dryrun=True`).
- **`EmbeddedGatewayHelper.with_clickhouse`** and **`with_postgres`**
  — opt-in paths for tests that need to assert on the row the
  gateway wrote. Stand up the backend yourself (a local Docker
  ClickHouse, a test Postgres schema) and pass the URL.

For the end-to-end observability story — ClickHouse tables, feedback
metrics, per-episode analytics — see
[Observability](observability.md).

---

## External references

- TensorZero —
  [Embedded gateway](https://www.tensorzero.com/docs/gateway/deployment/embedded/)
  · [`AsyncTensorZeroGateway`](https://www.tensorzero.com/docs/gateway/)
  — the real class `MockGateway` duck-types and
  `EmbeddedGatewayHelper` wraps.
- FastMCP —
  [`CallToolResult`](https://gofastmcp.com/python-sdk/fastmcp-client-client) —
  the return type `MockToolProvider.call_tool` produces, verbatim.
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html) —
  the plugin mechanism `pytest_plugins = ["t0_agents.testing.fixtures"]`
  wires into.
- [pytest markers](https://docs.pytest.org/en/stable/how-to/mark.html)
  — how `integration` and `embedded` are registered and selected.

---

## Closing the tour

This chapter completes the progressive tour. Starting from a single `Agent` primitive, you have added tools (the `mcp` adapter and loop), interactive input (elicitation), persistence (memory), policy (guardrails), live narration (streaming), multi-agent orchestration (composition), HTTP transport (serving), and terminal access (CLI) — and you now have the tools to keep every layer under automated test.

For deeper reference material that spans the tour — the full public API, the exception taxonomy, episode-level observability, production deployment trade-offs, multi-server composition — see the **Appendices** on the [index](index.md):

- Runnable example: [`examples/agent/11_testing/`](../examples/agent/11_testing/README.md).
- Contributing guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- Agent reference: [`agent.md`](agent.md).
- Error handling (what assertion helpers fire alongside): [`error-handling.md`](error-handling.md).
- Observability (the end-to-end picture mocks opt out of): [`observability.md`](observability.md).
- Production notes (the line between "real inference in tests" and "inference in production"): [`production.md`](production.md).
- API reference (cross-module index): [`api-reference.md`](api-reference.md).
