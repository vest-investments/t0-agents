# API Reference

Complete API reference for the `t0-agents` library. All public symbols are
importable from the top-level package:

```python
from t0_agents import run_agent, AgentResult, MaxTurnsExceededError, ToolChoice
```

Lower-level helpers are available from `t0_agents.mcp`:

```python
from t0_agents.mcp import (
    convert_tools,
    serialize_tool_result,
    build_tool_result,
    execute_tool_calls,
    execute_tool_calls_parallel,
    has_tool_calls,
)
```

> **Glow surface** — the decorator API (13 public symbols, behind the
> `[glow]` extra):
>
> ```python
> from t0_agents.glow import (
>     App,              # FastAPI-style decorator container
>     Depends,          # injection sentinel (tool values, sub-agents)
>     FluentRun,        # awaitable handle, .events()/.json()/.text()
>     MCPServer,        # typed MCP server metaclass + Annotated marker
>     Selector,         # generic typed event filter
>     Thought,          # selector for ThoughtEvent
>     ToolCall,         # selector for ToolCallEvent
>     Text,             # selector for TextEndEvent
>     ToolResult,       # selector for ToolResultEvent
>     TurnEnd,          # selector for TurnEndEvent
>     Done,             # selector for DoneEvent
>     Error,            # selector for ErrorEvent
>     regex_guardrail,  # re-export from t0_agents.guardrails.regex
> )
> ```
>
> See [`docs/glow/index.md`](glow/index.md) for the full guide.
> Install with `pip install 't0-agents[glow]'` — the extra pulls
> `pydantic>=2.0`.  Every `@app.agent` wrapper exposes `_agent` (the
> v1.0 `Agent` the decorator built); the drop-down recipes live at
> [`docs/glow/escape-hatch.md`](glow/escape-hatch.md).

> **v1.0 output modes** — 11 new public symbols:
>
> ```python
> from t0_agents import (
>     OutputMode,         # Protocol[T_co]
>     ChatOutput,         # T=str, default — tool loop
>     JsonOutput,         # T=dict — single-shot JSON
>     StructuredOutput,   # T=M (Pydantic BaseModel subclass)
>     ToolCallOutput,     # T=list[ToolCall] — router pattern
>     StreamingOutput,    # placeholder; wraps base mode
>     OutputDiagnostics,  # parse_strategy / attempts / raw_text
>     Recovery,           # retry / accept / fail outcomes
>     ParseFailure,       # failure description handed to recover()
>     Attempt,            # one past inference attempt
>     InferenceRequest,   # mutable request descriptor
> )
> ```
>
> See `docs/output-modes.md` for the full guide. The
> `[pydantic]` extra (`pip install 't0-agents[pydantic]'`) is required
> for `StructuredOutput`. The 5 v0.12 JSON-only kwargs on
> `AgentConfig` (`function_kind`, `json_mode`, `output_schema`,
> `permissive_json_parsing`, `max_json_retries`) were **removed** in
> v1.0 — passing any raises `TypeError`. The read-side
> `AgentResult.{response, parsed, parse_strategy, json_parse_attempts,
> function_kind}` properties remain.

---

## `run_agent`

Main entry point. Runs the full agent loop: discover tools, call TensorZero
inference, execute tool calls, feed results back, repeat until the LLM
responds without tool calls or `max_turns` is reached.

### Signature

```python
async def run_agent(
    t0: AsyncTensorZeroGateway,
    tool_provider: Client[Any] | None = None,
    user_message: str = "",
    function_name: str = "",
    *,
    episode_id: str | None = None,
    max_turns: int = 20,
    tool_timeout: float | None = 30.0,
    parallel_tools: bool = False,
    dryrun: bool = False,
    tool_choice: ToolChoice | None = None,
    system_arguments: dict[str, Any] | None = None,
    system: str | None = None,
    prior_messages: list[Message] | None = None,
) -> AgentResult:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `t0` | `AsyncTensorZeroGateway` | *(required)* | Connected TensorZero gateway instance (already inside `async with`). |
| `tool_provider` | `Client[Any] \| None` | `None` | Optional connected FastMCP client (already inside `async with`). When `None` (or omitted), no tools are advertised, `list_tools()` is never called, and the loop runs a single inference and returns the text response — the "toolless chat" path. When provided, must remain connected for the full loop. If the LLM emits a tool call while `tool_provider` is `None`, the loop raises `T0AgentsFatalError` rather than stringifying the policy violation into a tool error. |
| `user_message` | `str` | `""` | The user's query as a plain string. Non-empty in practice; defaults to `""` so `tool_provider` can default to `None` ahead of it in the positional signature (Python forbids a required positional after a defaulted one). |
| `function_name` | `str` | `""` | Name of the chat function defined in `tensorzero.toml`. Defaulted to `""` for the same positional-signature reason. |
| `episode_id` | `str \| None` | `None` | Existing episode ID to continue. `None` starts a fresh episode; TensorZero allocates one and the loop captures it. |
| `max_turns` | `int` | `20` | Maximum inference turns before aborting with `MaxTurnsExceededError`. |
| `tool_timeout` | `float` | `30.0` | Per-tool-call timeout in seconds, forwarded to `Client.call_tool(timeout=...)`. Covers the **entire** `call_tool()` duration including any [elicitation](elicitation.md) wait time. When using elicitation-enabled tools, set to `None` (no timeout) or a generous value (e.g., `300.0`). Only applies to foreground execution. |
| `parallel_tools` | `bool` | `False` | If `True`, dispatches tool calls concurrently via `asyncio.gather`. Otherwise sequential. |
| `dryrun` | `bool` | `False` | Forwarded to every `t0.inference(dryrun=True)` call. Prevents writes to ClickHouse. |
| `tool_choice` | `ToolChoice \| None` | `None` | Tool-choice directive forwarded to every inference call. See `ToolChoice` below. |
| `system_arguments` | `dict[str, Any] \| None` | `None` | Forwarded to TensorZero as `input.system` for the system-template render. When the function declares a system schema, the dict must match the schema directly — its keys are the schema-described properties (no `arguments` wrapper). Used to fill MiniJinja variables declared in the function's system template (e.g. `{{ max_turns }}`). When `Agent.invoke` / `Run.events` is the caller, this is shallow-merged with the memory strategy's `memory_context` injection — caller keys win on collision. See [Memory](memory.md#combining-memory-with-caller-supplied-template-variables). |
| `system` | `str \| None` | `None` | Raw system prompt forwarded to TensorZero as `input.system` (string form). Mutually exclusive with `system_arguments`. Only valid when the TensorZero function does **not** declare `schemas.system.path` — the gateway returns HTTP 400 otherwise. Escape hatch for prompt-as-a-service flows where the prompt body itself is runtime data; prefer `system_arguments` (with a configured template) for production code so ClickHouse captures structured prompt variables instead of opaque strings. See [Runtime system prompt: `system` vs `system_arguments`](#runtime-system-prompt-system-vs-system_arguments). |
| `prior_messages` | `list[Message] \| None` | `None` | TensorZero-native message history to seed the loop with. Used by `Agent.invoke` when `memory_strategy="messages"` is in effect (see [Memory](memory.md)). When `None` or empty, the loop starts with a single `Message(role="user", content=user_message)`, byte-identical to behaviour pre-flag. |

### Returns

`AgentResult` with the final text response, episode ID, per-turn inference
IDs, turn count, and response source.

### Raises

- `MaxTurnsExceededError` -- Loop ran `max_turns` times without a
  tool-call-free response.
- `T0AgentsFatalError` -- The LLM emitted a `ToolCall` while
  `tool_provider` was `None`. The optional contract requires the
  caller to wire a tool provider whenever tool dispatch is expected;
  a hallucinated tool call against a toolless configuration is
  surfaced as a fatal policy violation rather than stringified into
  a `ToolResult` error so the run aborts cleanly.
- `TensorZeroError` -- Non-recoverable inference error from TensorZero
  (propagated). Recoverable errors (matching `RECOVERABLE_INFERENCE_PATTERNS`)
  trigger an automatic retry with a corrective user message.
- `ValueError` -- Both `system` and `system_arguments` were supplied
  (they're mutually exclusive — both target `input.system`), or
  `user_message` / `function_name` was left at the empty-string
  default.

### Behavior notes

- `dryrun=True` makes inference IDs ephemeral. They are never stored in
  ClickHouse, so any subsequent `t0.feedback()` call targeting those IDs
  will fail with HTTP 400.
- When the LLM returns only `Thought` blocks (no `Text`), the thought
  content is used as a fallback and `response_source` is set to `"thought"`.
- Recoverable `TensorZeroError` (status 400, text matching `"not in
  request.tools"`) injects a corrective user message listing available tools
  and retries the loop turn.
- Tool results from all tool calls in a single turn are bundled into one user
  message (not one message per result).
- When elicitation-enabled tools are in use (tools that call `ctx.elicit()`
  server-side), `tool_timeout` must account for human response time. The
  default `30.0` seconds may not be sufficient. Use `tool_timeout=None` to
  disable or `tool_timeout=300.0` for a generous budget. See
  [Elicitation](elicitation.md) for details.

### Examples

**Easy** -- minimal call:

```python
async with AsyncTensorZeroGateway("http://localhost:3000") as t0:
    async with Client(mcp_server) as client:
        result = await run_agent(t0, client, "Add 2 + 3", "my_chat")
        print(result.response)
```

**Intermediate** -- with `max_turns` and `tool_timeout`:

```python
result = await run_agent(
    t0, client, "Search and summarize results", "my_chat",
    max_turns=5,
    tool_timeout=60.0,
)
print(f"Completed in {result.turns} turns")
```

**Advanced** -- with `parallel_tools`, `tool_choice`, and episode continuation:

```python
result1 = await run_agent(
    t0, client, "Look up the weather in Berlin", "my_chat",
    parallel_tools=True,
    tool_choice="required",
)
result2 = await run_agent(
    t0, client, "Now compare it to Munich", "my_chat",
    episode_id=result1.episode_id,
    tool_choice={"specific": "get_weather"},
)
```

### Runtime system prompt: `system` vs `system_arguments`

Both kwargs target TensorZero's `input.system` field, but they encode
different intents and the gateway accepts different shapes depending
on the function's configuration.

| Function in `tensorzero.toml`      | `system_arguments={"k": "v"}`         | `system="<string>"`              |
|------------------------------------|---------------------------------------|----------------------------------|
| No `schemas.system.path` declared  | Passes through as `input.system={...}` | Passes through as `input.system="…"` |
| `schemas.system.path` declared     | Renders the configured template       | **HTTP 400 from the gateway**    |

**Default to `system_arguments`.** It keeps the prompt template in
`tensorzero.toml`, where you can iterate on wording or A/B-test
variants without redeploying — and ClickHouse stores the structured
*arguments* rather than an opaque rendered string, which keeps the
captured inferences usable for fine-tuning later (TensorZero's
[prompt-template rationale](https://www.tensorzero.com/docs/gateway/create-a-prompt-template.md)
spells this out).

**Reach for `system`** only when the prompt body is itself runtime
data — for example a prompt-as-a-service product where each request
ships the user's own prompt. In that case there's nothing to template
against, and the structured-dataset benefit doesn't apply because the
prompt *is* the user input.

The two kwargs are mutually exclusive — passing both raises
`ValueError`. The check is the same on every public surface
(`run_agent`, `Agent.invoke`, `Run.events`, `run_structured_inference`,
`StreamingAgentLoop.stream`).

When using `Agent.invoke` / `Run.events`, `system` is **also**
incompatible with `memory_strategy="system_template"` (the strategy
injects `memory_context` into `input.system` via the template, which
requires a dict). Use `memory_strategy="prepended_message"` (the
default) or `memory_strategy="messages"` if you need raw `system`
alongside memory.

```python
# Recommended: prompt structure lives in tensorzero.toml; only the
# template variables flow at runtime.
result = await agent.invoke(
    "where is the treasure?",
    system_arguments={"persona": "pirate"},
)

# Escape hatch: the prompt body itself is request data.  Only valid
# when the TensorZero function has no system schema declared.
result = await agent.invoke(
    "where is the treasure?",
    system="You are a pirate. Reply in exaggerated pirate slang.",
)
```

See [`examples/agent/02_agent_class/`](../examples/agent/02_agent_class/)
for the runnable PaaS-style demo.

---

## `convert_tools`

Converts FastMCP tool definitions to TensorZero's `additional_tools` format.

### Signature

```python
def convert_tools(mcp_tools: list[MCPTool]) -> list[dict[str, Any]]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mcp_tools` | `list[MCPTool]` | *(required)* | Tool definitions from `client.list_tools()`. |

### Returns

`list[dict[str, Any]]` -- each dict has keys `"name"` (str), `"description"`
(str), and `"parameters"` (JSON Schema dict).

### Behavior notes

- `description` is coerced: `None` and empty strings become `""`. This
  handles `@mcp.tool` functions that have no docstring and no explicit
  `description=` parameter.
- `inputSchema` from the MCP tool is passed through as-is to `"parameters"`.

### Example

```python
mcp_tools = await client.list_tools()
t0_tools = convert_tools(mcp_tools)
# [{"name": "add", "description": "Add two numbers", "parameters": {...}}]
```

---

## `serialize_tool_result`

Serializes a FastMCP `CallToolResult` to the `str` that TensorZero expects
for `ToolResult.result`.

### Signature

```python
def serialize_tool_result(result: CallToolResult) -> str:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `CallToolResult` | *(required)* | Result from `client.call_tool()`. |

### Returns

`str` -- serialized result.

### Behavior notes

Four branches, evaluated in order:

1. **`is_error=True`** -- returns `"ERROR: "` + joined `.content[*].text`.
   Falls back to `"ERROR: Unknown tool error"` if no textual block is present.
2. **`.data` is `str`** -- passthrough, returned as-is.
3. **`.data` is non-`None`, non-`str`** -- `json.dumps(data, default=str,
   ensure_ascii=False)`. Booleans become JSON-cased (`"true"` / `"false"`).
   `datetime` and `UUID` are stringified via `default=str`. If `json.dumps`
   fails, falls through to branch 4.
4. **`.data` is `None`** (or `json.dumps` failed) -- concatenates `.text`
   from every content block that has it, joined with `"\n"`. Non-textual
   blocks (`ImageContent`, `AudioContent`, `EmbeddedResource`) are skipped.

### Example

```python
mcp_result = await client.call_tool("add", {"a": 1, "b": 2})
text = serialize_tool_result(mcp_result)  # "3" or '{"sum": 3}'
```

---

## `build_tool_result`

Builds a TensorZero `ToolResult` from a `ToolCall` and a FastMCP
`CallToolResult`.

### Signature

```python
def build_tool_result(tool_call: ToolCall, mcp_result: CallToolResult) -> ToolResult:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_call` | `ToolCall` | *(required)* | The tool call from the inference response. |
| `mcp_result` | `CallToolResult` | *(required)* | The result from FastMCP `call_tool`. |

### Returns

`ToolResult` with `name`, `result` (serialized string), and `id`.

### Behavior notes

- `ToolCall.name` is `Optional[str]` (`None` when the LLM hallucinated a
  tool name not in the schema). `ToolResult.name` must be `str`, so the
  function falls back to `tool_call.raw_name` -- the raw string the model
  produced, which is always populated.

### Example

```python
tool_result = build_tool_result(tool_call, mcp_result)
# ToolResult(name="add", result="5", id="call_abc123")
```

---

## `execute_tool_calls`

Executes every `ToolCall` in a list of content blocks sequentially.

### Signature

```python
async def execute_tool_calls(
    tool_provider: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_provider` | `Client[Any]` | *(required)* | Connected FastMCP client. Non-optional — this helper only runs inside the tool-dispatch branch of `run_agent`, after the optional contract has already proven `tool_provider is not None`. |
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response (mix of `Text`, `Thought`, `ToolCall`). |
| `timeout` | `float` | `30.0` | Per-tool-call timeout in seconds. |

### Returns

`list[ToolResult]` -- one per `ToolCall` block, in input order.

### Behavior notes

- Non-`ToolCall` blocks (`Text`, `Thought`) are skipped silently.
- Output order matches the input order of `ToolCall` blocks.
- Uses `_execute_single_tool_call` internally, so the full error contract
  (hallucinated names, invalid JSON, `ToolError`, `McpError`, catch-all)
  applies per call.

### Example

```python
results = await execute_tool_calls(client, response.content, timeout=10.0)
```

---

## `execute_tool_calls_parallel`

Executes `ToolCall` blocks concurrently via `asyncio.gather`.

### Signature

```python
async def execute_tool_calls_parallel(
    tool_provider: Client[Any],
    content_blocks: list[Any],
    timeout: float = 30.0,
) -> list[ToolResult]:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_provider` | `Client[Any]` | *(required)* | Connected FastMCP client. Non-optional — same contract as `execute_tool_calls`. |
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response. |
| `timeout` | `float` | `30.0` | Per-tool-call timeout in seconds. |

### Returns

`list[ToolResult]` -- one per `ToolCall` block, in input order (guaranteed by
`asyncio.gather`).

### Behavior notes

- Same contract as `execute_tool_calls`: skips non-`ToolCall` blocks,
  preserves order, same error handling per call.
- Returns an empty list if no `ToolCall` blocks are present.
- Best suited for I/O-bound tools where concurrent dispatch reduces
  wall-clock time.

### Example

```python
results = await execute_tool_calls_parallel(client, response.content)
```

---

## `has_tool_calls`

Checks whether any content block in a list is a `ToolCall`.

### Signature

```python
def has_tool_calls(content_blocks: list[Any]) -> bool:
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content_blocks` | `list[Any]` | *(required)* | Content blocks from the inference response. |

### Returns

`bool` -- `True` if at least one block is a `ToolCall` instance.

### Behavior notes

- `Thought` blocks are not `ToolCall` instances and return `False`.
- Used by `run_agent` to decide whether to continue the loop (tool execution)
  or return the final response.

### Example

```python
if has_tool_calls(response.content):
    results = await execute_tool_calls(client, response.content)
```

---

## `AgentResult`

Dataclass returned by `run_agent`.

### Definition

```python
@dataclass
class AgentResult:
    response: str
    episode_id: str | None = None
    inference_ids: list[str] = field(default_factory=list)
    turns: int = 0
    response_source: Literal["text", "thought"] = "text"
```

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `response` | `str` | *(required)* | Joined text from the terminating turn's `Text` blocks (or `Thought` blocks as fallback). |
| `episode_id` | `str \| None` | `None` | TensorZero episode ID. Captured from the first inference response. |
| `inference_ids` | `list[str]` | `[]` | One inference ID per turn. Includes `"error-recoverable-turn-N"` placeholders for recovered errors. |
| `turns` | `int` | `0` | Number of inference turns actually used. |
| `response_source` | `Literal["text", "thought"]` | `"text"` | `"text"` if the response came from `Text` blocks, `"thought"` if from `Thought` blocks (fallback). |

### Example

```python
result = await run_agent(t0, client, "Hello", "my_chat")
print(result.response)         # "Hi there!"
print(result.turns)            # 1
print(result.response_source)  # "text"
```

---

## `Run` *(v1.0)*

> See [docs/run.md](run.md) for the canonical narrative — lifecycle,
> two-channel consumption, replay/fork semantics, error model,
> `Run(agent=None)` mode for third-party authors who need a Run
> without an enclosing Agent.  This section is the API-reference
> summary.

First-class invocation handle returned by `Agent.invoke(...)`.  A `Run`
represents one live or completed agent invocation.  It supports two
consumption channels — `await run.result()` for the final
`AgentResult`, and `async for ev in run.events()` for the streaming
event sequence — that work concurrently for chat agents.

```python
from t0_agents import Agent, AgentConfig, Run

agent = Agent(AgentConfig(name="assistant", function_name="agent"), t0=t0, tool_provider=mcp)
run: Run = agent.invoke("hello")
result = await run             # equivalent to await run.result()
async for ev in run.events():  # chat-only; raises on JSON agents
    print(ev)
```

### Public surface

| Member | Purpose |
|--------|---------|
| `await run` / `await run.result()` | Final `AgentResult` (v0.12-identical shape) |
| `async for ev in run.events(replay=False)` | Stream `AgentEvent`s; `replay=True` re-yields buffered history |
| `async for s in run.stream_text()` | Text-delta-only convenience iterator |
| `await run.cancel()` | Cancel the drive; subsequent `result()` raises `RunCancelledError` |
| `await run.feedback(name, value, *, target="episode"\|"inference", inference_id=None)` | Forward outcome feedback to TensorZero |
| `await run.replay(*, agent=None, variant=None, tags=None, episode_id=None)` | Re-run with the same input; new episode by default |
| `await run.fork(extra_message, *, agent=None)` | Continue from this Run's transcript with an extra user turn |
| `run.transcript()` | TZ-shaped `{role, content}` list (the boundary pair) |
| `run.state` | `"pending" \| "running" \| "succeeded" \| "failed" \| "cancelled"` |
| `run.input` | Frozen `RunInput` snapshot of the call |
| `run.episode_id`, `run.inference_ids` | Captured mid-run |

`Run` holds a `weakref.ref` to its `Agent`; if the `Agent` is garbage-
collected before `replay()` or `fork()` runs, those methods raise
`RuntimeError`.  Pass `agent=` to retarget the replay/fork explicitly.

### `RunInput`

Immutable snapshot of one `Agent.invoke(...)` call.

```python
@dataclass(frozen=True, slots=True)
class RunInput:
    user_message: str
    function_name: str
    episode_id: str | None = None
    tags: Mapping[str, str] | None = None
    system: str | None = None
    system_arguments: Mapping[str, Any] | None = None
    prior_messages: tuple[Mapping[str, Any], ...] = ()
```

The :class:`OutputMode` for the invocation is stored separately on
the `Run` itself rather than on `RunInput`, so each Run knows how
to materialise its result without re-deriving from config.

### `RunState`

Type alias for the lifecycle states:

```python
RunState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
```

### `RunCancelledError`

Raised by `await run.result()` after `await run.cancel()`.

### `RunFailedError`

Raised by `await run.result()` when the underlying drive failed
without a captured exception.  The original cause is preserved on
`.cause`.  In the normal failure path `result()` re-raises the
original exception (`MaxTurnsExceededError`, `GuardrailBlockedError`,
…) directly; `RunFailedError` is the defensive fallback for the
"failed without captured cause" edge case.

---

## `Middleware` *(v1.0)*

`Middleware` is the v1.0 extension surface for cross-cutting concerns:
memory hydration, guardrails, cost tracking, sub-agent orchestration,
JSON retry, observability — and any custom layer users add (rate
limiters, PII redactors, recovery policies).  Each middleware wraps
the inner inference loop with `async def __call__(ctx, next_) ->
AgentResult` and is composed by `AgentConfig.__post_init__` into a
chain that wraps a single terminal callable.

### Public surface

| Symbol | Purpose |
|--------|---------|
| `Middleware` | Runtime-checkable Protocol; `name: str` + `async __call__(ctx, next_) -> AgentResult`. |
| `RunContext` | Mutable per-run state threaded through the chain.  Carries config, user_message, episode_id, tags, system, system_arguments, output, t0, tool_provider, on_inference, prior_messages, extras, result, run.  Methods: `chain_on_inference(cb)`, `replace_tool_provider(tp)`. |
| `MemoryMiddleware(provider, strategy)` | Load + inject + save memory records around the run (parity with v0.12 inline behaviour). |
| `GuardrailsMiddleware(guardrails)` | `before_run`/`after_run` hooks, `GuardedToolProvider` wrap, cost-budget tracker on `GuardrailContext`. |
| `CostTrackingMiddleware()` | Always-on `RunUsage` accumulator; populates `Run.cost()` / `Run.usage()`. |
| `SubAgentMiddleware(sub_agents)` | Wraps `tool_provider` with `_OrchestrationToolProvider`; binds `parent_episode_id` ContextVar. |
| `JsonRetryMiddleware()` | Drives the parse-failure retry loop for single-shot output modes by consulting `OutputMode.recover()`; no-op pass-through for chat agents. |
| `ObservabilityMiddleware(tracer=None)` | Wraps the run in `tracer.start_as_current_span(f"agent.invoke/{name}")`.  Opt-in. |

### Default chain

When `AgentConfig.middleware is None`, `__post_init__` materialises:

```
Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal
```

`Memory` is included only when `memory=` is set; `SubAgents` only
when `sub_agents=` is non-empty; `Guardrails` only when `guardrails=`
is non-empty.  `CostTracking` and `JsonRetry` are always present.

`SubAgents` precedes `Guardrails` in chain order so the
`_OrchestrationToolProvider` wrap is INNER and `GuardedToolProvider`
is OUTER — preserving the v0.12 invariant that orchestrator-level
guardrails observe sub-agent delegations as tool calls.

### Overrides

- `middleware=None` (default) — auto-materialise the chain.
- `middleware=[]` — explicit "no middleware".  Disables every default
  layer including `CostTracking`, so `Run.cost()` stays at `0.0`.
- `middleware=[Custom1(), Custom2(), ...]` — replace the default
  chain entirely.

The static helper `AgentConfig.default_middleware_for(memory=, ...)`
returns the same list `__post_init__` would build, so power users
can sandwich the default:

```python
AgentConfig(
    name="...",
    function_name="...",
    middleware=[
        MyOutermost(),
        *AgentConfig.default_middleware_for(memory=m, guardrails=g),
        MyInnermost(),
    ],
)
```

---

## `RunUsage` *(v1.0)*

Frozen dataclass returned by `Run.usage()` after a Run that fired
at least one inference.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Sum of `response.usage.input_tokens` over every inference. |
| `output_tokens` | `int` | Sum of `response.usage.output_tokens`. |
| `cost` | `float` | Sum of `response.usage.cost` (treats `None` as `0.0`). |
| `provider_cache_read_input_tokens` | `int` | Sum of provider cache-read input tokens (`None → 0`). |
| `provider_cache_write_input_tokens` | `int` | Sum of provider cache-write input tokens (`None → 0`). |
| `cost_is_priced` | `bool` | `True` only when a provider reported a price — tells a real `$0.00` apart from missing pricing. |

`Run.cost() -> float` returns the same `cost` field as a convenience
for callers that only need the dollar figure.  Both default to `0.0`
/ `None` when no inference has fired (or when `middleware=[]`
disabled `CostTrackingMiddleware`).

See [Token usage](usage.md) for the footprint-vs-billing distinction,
the per-inference `Usage` value type, and `ConversationUsage`.

---

## Output modes *(v1.0)*

> See [`docs/output-modes.md`](output-modes.md) for the canonical
> narrative — the protocol, the 5 built-ins with construction
> examples, the recovery handshake, custom mode authoring.  This
> section is the API-reference summary.

`OutputMode[T]` is the polymorphic protocol that replaces v0.12's
`function_kind` / `json_mode` / `output_schema` cluster.  Every
single-shot or chat output goes through an `OutputMode` instance;
the `JsonRetryMiddleware` drives recovery for single-shot modes.

```python
from t0_agents import (
    OutputMode, ChatOutput, JsonOutput, StructuredOutput,
    ToolCallOutput, StreamingOutput,
    OutputDiagnostics, Recovery, ParseFailure, Attempt, InferenceRequest,
)
```

### Public protocol

```python
@runtime_checkable
class OutputMode(Protocol[T_co]):
    name: str
    expects_tool_loop: bool
    supports_streaming: bool

    def shape_inference(self, req: InferenceRequest) -> InferenceRequest: ...
    async def materialize(self, response, attempts) -> AgentResult[T_co]: ...
    async def recover(self, failure, attempts) -> Recovery: ...
```

### Built-in modes

| Class | `T` | `expects_tool_loop` | `supports_streaming` | Use when |
|---|---|---|---|---|
| `ChatOutput()` | `str` | `True` | `True` | Default — full tool-call loop. |
| `JsonOutput(schema=, json_mode=, permissive=, max_retries=)` | `dict[str, Any]` | `False` | `False` | Single-shot JSON with retry. |
| `StructuredOutput[M](model)` | `M` (Pydantic) | `False` | `False` | Pydantic-validated extraction; `pip install 't0-agents[pydantic]'`. |
| `ToolCallOutput(tool_choice=)` | `list[ToolCall]` | `False` | `False` | Router pattern — extract tool calls without executing. |
| `StreamingOutput(base=ChatOutput())` | delegates | delegates | `True` | Documentation wrapper; `base=JsonOutput(...)` deferred to v1.1. |

### Helper types

| Type | Purpose |
|---|---|
| `InferenceRequest` | Mutable request descriptor threaded through `shape_inference`. |
| `OutputDiagnostics(parse_strategy, attempts, raw_text, failure_detail)` | Attached to `AgentResult.diagnostics`; backs the v0.12 read-side properties. |
| `Attempt(inference_id, raw_text, parsed, usage)` | One entry in the retry history; accumulated on `ctx.extras["_output_attempts"]`. |
| `ParseFailure(raw_text, inference_id, reason, detail)` | Passed to `OutputMode.recover` on parse failure. |
| `Recovery.retry(hint) \| Recovery.accept(value) \| Recovery.fail()` | The three recovery actions. |

### Per-call narrowing

`Agent.invoke(output=...)` is overloaded so mypy narrows `Run[T]`
when a concrete `OutputMode[T]` is supplied:

```python
run: Run[Customer] = agent.invoke(msg, output=StructuredOutput(Customer))
result: AgentResult[Customer] = await run.result()
print(result.value)   # Customer instance, statically typed
```

Without `output=`, `agent.invoke(...)` returns `Run[Any]` — explicit
and predictable.

### `[pydantic]` extra

`StructuredOutput` lazily imports pydantic at construction.  Install
the extra for the canonical path:

```bash
pip install 't0-agents[pydantic]'
```

`ConfigurationError` surfaces at construction time with an install
hint if pydantic isn't available.

---

## `MaxTurnsExceededError`

Exception raised when the agent loop reaches `max_turns` without the LLM
producing a tool-call-free response.

### Definition

```python
class MaxTurnsExceededError(Exception):
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        super().__init__(f"Agent loop exceeded {max_turns} turns")
```

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `max_turns` | `int` | The limit that was exceeded. |

### Example

```python
try:
    result = await run_agent(t0, client, "Complex task", "my_chat", max_turns=3)
except MaxTurnsExceededError as e:
    print(f"Loop stopped after {e.max_turns} turns")
```

---

## `ToolChoice`

Re-exported from `tensorzero`. Controls how the LLM selects tools during
inference.

### Type

```python
ToolChoice = Literal["auto", "required", "off"] | dict[str, str]
```

### Accepted values

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to call a tool. |
| `"required"` | Model must call at least one tool. |
| `"off"` | Tool calling disabled for this inference. |
| `{"specific": "<name>"}` | Model must call the named tool. |

### Behavior notes

- Passed as `tool_choice` to `run_agent`. `None` (the default) is equivalent
  to letting the model decide freely (TensorZero's default behavior).
- The `tool_choice` directive applies to every inference call in the loop, not
  just the first.

### Example

```python
# Force a specific tool
result = await run_agent(
    t0, client, "Get weather", "my_chat",
    tool_choice={"specific": "get_weather"},
)
```

---

## Reference Elicitation Handlers

Pre-built handlers for use with `Client(mcp, elicitation_handler=...)`. Intended
for **testing and prototyping** — production applications should implement a
handler that presents prompts to the user via their UI.

Available from `t0_agents.mcp`:

```python
from t0_agents.mcp import auto_approve_handler, auto_decline_handler
```

### `auto_approve_handler`

Approves binary approval requests (`response_type is None`) and declines
structured input requests (where data would need to be provided).

#### Signature

```python
async def auto_approve_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
```

#### Behavior

| `response_type` | Action | Content |
|-----------------|--------|---------|
| `None` (approval) | `"accept"` | `{}` |
| Anything else | `"decline"` | — |

#### Example

```python
from fastmcp import Client
from t0_agents.mcp import auto_approve_handler

client = Client(mcp_server, elicitation_handler=auto_approve_handler)
async with client:
    result = await client.call_tool("delete_files", {"files": ["a.txt"]})
```

### `auto_decline_handler`

Declines all elicitation requests unconditionally. Useful for verifying that
tools handle rejection gracefully.

#### Signature

```python
async def auto_decline_handler(
    message: str,
    response_type: type[Any] | None,
    params: Any,
    context: Any,
) -> ElicitResult[Any]:
```

#### Behavior

Always returns `ElicitResult(action="decline")` regardless of `response_type`.

#### Example

```python
from fastmcp import Client
from t0_agents.mcp import auto_decline_handler

client = Client(mcp_server, elicitation_handler=auto_decline_handler)
async with client:
    result = await client.call_tool("delete_files", {"files": ["a.txt"]})
    # Tool receives decline and handles gracefully
```

See [Elicitation](elicitation.md) for the full handler interface, content
format guide, and server-side patterns.
