# Agent

The `Agent` class is the declarative front door of t0-agents. It holds your configuration, your TensorZero gateway, and your tool provider in one object, then exposes a clean `.invoke(message)` surface that returns a `Run[T]` handle (awaitable for the final `AgentResult`, iterable for live `AgentEvent`s). Every higher-level feature in this library — memory, guardrails, streaming, HTTP serving, the TUI, multi-agent composition, testing — plugs in through `Agent` without changing the shape of the class.

> **v1.0 surface — read these four pages first:**
> [Run](run.md) (the invocation handle), [Middleware](middleware.md)
> (composable cross-cutting concerns), [Output modes](output-modes.md)
> (polymorphic output shape), [Composition](composition.md) (nine
> topology builders for multi-agent orchestration).  The legacy
> `agent.invoke()` / `agent.invoke().events()` shims keep working (one
> `DeprecationWarning` per source line); they're removed in v1.1.  The
> v0.12 → v1.0 migration playbook lives in
> [migration-v0.12-to-v1.0.md](migration-v0.12-to-v1.0.md).

```
                        ┌──────────────────────────┐
                        │         Agent            │
                        │  (config, t0, provider)  │
                        └────────────┬─────────────┘
                                     │ .run / .stream
                                     ▼
                        ┌──────────────────────────┐
                        │  t0_agents.mcp.run_agent │
                        │     tool-calling loop    │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌──────────────────┐    ┌────────────────┐
     │  ToolProvider  │    │  MemoryProvider  │    │    Guardrail   │
     │  (chat-mode)   │    │    (optional)    │    │   (optional)   │
     └────────────────┘    └──────────────────┘    └────────────────┘
```

`Agent` delegates to `t0_agents.mcp.run_agent` rather than reimplementing the loop. Everything `run_agent` gives you — recoverable-error handling, parallel tool dispatch, thought-only fallback, error-as-tool-result — carries through untouched.

## Install

```bash
pip install t0-agents
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add t0-agents
```

Requires Python 3.11+. The base install pulls in [`tensorzero`](https://www.tensorzero.com/docs/) and [`fastmcp`](https://gofastmcp.com/) only. Collaborators live behind extras and are documented in their own pages — install only what you use:

| Extra | Unlocks | Read more |
|---|---|---|
| *(none)* | `Agent`, `AgentConfig`, `ToolProvider`, exceptions, `run_agent` | this page |
| `memory-sqlite` | `SqliteMemory` for persistent `MemoryProvider` backends | [memory.md](memory.md) |
| `serve` | FastAPI wrapper (`serve.create_app`) | [serving.md](serving.md), [streaming.md](streaming.md) |
| `cli` | Typer + Rich terminal UI (`t0-agents run`) | [cli.md](cli.md) |

## Quickstart

Start the gateway and set a provider API key first — `Agent` is a client of TensorZero, not a replacement for it.

```bash
export GROQ_API_KEY=...
docker compose up -d
```

Then wire an `Agent` around the running gateway and your own FastMCP server:

```python
import asyncio
from fastmcp import Client
from tensorzero import AsyncTensorZeroGateway
from t0_agents import Agent, AgentConfig

from my_server import mcp  # your FastMCP instance

async def main() -> None:
    config = AgentConfig(
        name="weather-assistant",
        function_name="agent",         # must match your tensorzero.toml
        max_turns=10,
    )

    gateway = AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000")
    mcp_client = Client(mcp)

    async with await gateway as t0, mcp_client:
        agent = Agent(config, t0=t0, tool_provider=mcp_client)
        result = await agent.invoke("What is the weather in Tokyo?")

    print(result.response)
    print(f"{result.turns} turn(s), episode {result.episode_id}")
    print(f"{len(result.inference_ids)} inference id(s)")

asyncio.run(main())
```

What just happened: `Agent.invoke` called `AsyncTensorZeroGateway.inference` with your prompt plus the tool schemas discovered from your FastMCP server, executed any tool calls the LLM produced, fed the results back, and looped until the model returned a tool-call-free response. The caller still owns the lifecycle of both the gateway and the MCP client — `Agent` never opens or closes either.

For the end-to-end runnable project (tensorzero.toml, docker-compose.yml, MCP server, README), see [`examples/02_agent_class/`](../examples/02_agent_class/).

## Concepts

### Composition

`Agent` is a wrapper with opt-in collaborators:

| Collaborator | Lives on | Enabled when | Behaviour when absent |
|---|---|---|---|
| `ToolProvider` | `Agent(tool_provider=...)` | chat-mode agents | required when `function_kind="chat"`; optional (and unused) when `function_kind="json"` |
| `MemoryProvider` | `AgentConfig.memory` | non-`None` | no load/save, no context injection |
| `Guardrail` list | `AgentConfig.guardrails` | non-empty | provider passed through unwrapped, no cost tracking |
| Sub-agents | `AgentConfig.sub_agents` | non-empty | no additional MCP tools registered |
| Streaming | `Run.events()` call | chosen by caller | `.run()` returns a complete `AgentResult` |

With every collaborator absent, `Agent.invoke(message)` is equivalent to calling `run_agent(t0, tool_provider, message, config.function_name, ...)` directly. Each collaborator is a non-invasive addition: you pay for what you use.

### Lifecycle ownership

The caller opens and closes the gateway and the tool provider. `Agent` only holds references:

```python
async with await AsyncTensorZeroGateway.build_http(...) as t0, Client(mcp) as tool_provider:
    agent = Agent(config, t0=t0, tool_provider=tool_provider)
    result = await agent.invoke("…")
# t0 and tool_provider closed here; agent should no longer be used.
```

`Agent` does not install signal handlers, configure logging, spawn threads, or touch the filesystem on your behalf.

JSON-mode agents (`function_kind="json"`) never dispatch tools, so the `tool_provider` kwarg is optional — drop it and let the constructor short-circuit the MCP plumbing entirely. See [structured-outputs.md](structured-outputs.md) for the full pattern.

```python
async with await AsyncTensorZeroGateway.build_http(...) as t0:
    agent = Agent(json_config, t0=t0)  # no tool_provider needed
    result = await agent.invoke("…")
```

### Concurrency

Three places in the stack can run things in parallel: tool calls within one turn, batched tools across turns, and sub-agents under a single orchestrator. Each maps to a single knob; the framework intentionally does not provide a planner, a DAG DSL, or a `parallel(...)` helper on top of `asyncio.gather`. The full mental model with code examples lives in [concurrency.md](concurrency.md); the short version:

| Layer | Trigger | Knob |
|---|---|---|
| Intra-turn | LLM emits several tool calls in one assistant turn | `AgentConfig(parallel_tools=True)` |
| Inter-turn | LLM emits one tool per turn but the calls are independent | A "batch tool" on your MCP server |
| Multi-agent | Orchestrator delegates to several sub-agents in one turn | Same `parallel_tools=True` on the orchestrator |

`parallel_tools` defaults to `False` on purpose: it is unsafe to combine with `require_approval` (it would fan out N approval prompts to one human in the same turn) and it multiplies your peak tool-side QPS. See [Guardrails / `parallel_tools=True`](guardrails.md#parallel_toolstrue) and the [Production checklist](production.md#parallel-vs-sequential-tool-execution) before flipping it.

### Key design decisions

| Decision | Why |
|---|---|
| `Agent` delegates to `run_agent` | Anything the low-level loop guarantees (recoverable-error retries, thought-only fallback, error-as-tool-result) carries through automatically. |
| Opt-in collaborators via `AgentConfig` | Memory, guardrails, streaming, composition each cost you nothing when unused. |
| Structural typing via Protocols | `fastmcp.Client` satisfies `ToolProvider` with no adapter; custom tool sources just match the same two-method shape. |
| Caller owns connections | One less thing `Agent` can leak, and it matches how you'd use `AsyncTensorZeroGateway` or `fastmcp.Client` directly. |
| Config fields mirror TensorZero primitives | `tags`, `namespace`, `tool_choice`, `dryrun`, `parallel_tools` forward verbatim — no parallel abstraction to learn. |
| Async-first | Matches `AsyncTensorZeroGateway` and `fastmcp.Client`. No sync variant ships today. |

## API reference

Everything on this page is importable from the package root:

```python
from t0_agents import (
    Agent, AgentConfig, AgentResult,
    ToolProvider, MemoryProvider, Guardrail, GuardrailLogEntry,
    ConfigurationError, MaxTurnsExceededError, GuardrailBlockedError,
    SubAgentError, T0AgentsError, T0AgentsFatalError,
    ElicitationForwarder, ElicitationRequest,
)
```

### `AgentConfig`

Frozen-shape dataclass (slots, no inheritance). Validation runs in `__post_init__`; any failure raises `ConfigurationError`.

| Field | Type | Default | Role |
|---|---|---|---|
| `name` | `str` | *(required)* | Human-readable agent name (logging, tags). Must be non-empty. |
| `function_name` | `str` | *(required)* | TensorZero function name. Must match an entry in your `tensorzero.toml`. |
| `max_turns` | `int` | `20` | Hard cap on inference turns before `MaxTurnsExceededError`. Must be `> 0`. |
| `tool_timeout` | `float \| None` | `30.0` | Per-tool-call timeout in seconds. `None` resolves to the 30-second default. Must be positive when not `None`. |
| `parallel_tools` | `bool` | `False` | When `True`, tool calls in the same turn dispatch concurrently via `asyncio.gather`. |
| `dryrun` | `bool` | `False` | Forwarded to every `t0.inference(...)` call (skips logging and billing). |
| `tool_choice` | `ToolChoice \| None` | `None` | Forces tool-choice behaviour per TensorZero's [inference API](https://www.tensorzero.com/docs/gateway/api-reference/inference). |
| `tags` | `dict[str, str] \| None` | `None` | Observability tags attached to every inference. See [observability.md](observability.md). |
| `namespace` | `str \| None` | `None` | Experimentation namespace. Unknown namespaces fall back to the default config silently. |
| `memory` | `MemoryProvider \| None` | `None` | Opt-in persistent memory. See [memory.md](memory.md). |
| `memory_strategy` | `Literal["auto", "system_template", "prepended_message"]` | `"auto"` | Controls how loaded memory is injected into the prompt. See [memory.md](memory.md). |
| `guardrails` | `list[Guardrail]` | `[]` | Declarative policy layer. See [guardrails.md](guardrails.md). |
| `sub_agents` | `list[Agent]` | `[]` | Exposes nested agents as MCP tools on an orchestrator. See [multi-agent.md](multi-agent.md). |
| `function_kind` | `Literal["chat", "json"]` | `"chat"` | Declares the TensorZero function type. `"chat"` (default) is byte-identical to v0.9.0; `"json"` marks a single-shot JSON function and routes `Agent.invoke` through `run_structured_inference`. See [structured-outputs.md](structured-outputs.md). |
| `output_schema` | `Mapping[str, Any] \| Path \| None` | `None` | Optional dynamic schema for JSON functions. `Path` is read as JSON at construction time (relative paths resolve against `cwd`). Forbidden when `function_kind == "chat"`. |
| `json_mode` | `Literal["off", "on", "strict", "tool"] \| None` | `None` | TensorZero `json_mode` override. Required when `function_kind == "json"`; forbidden when `function_kind == "chat"`. |
| `permissive_json_parsing` | `bool` | `True` | When `True`, JSON runs whose `output.parsed` came back `None` fall through an in-Python recovery layer that scans the raw text for balanced `{}` / `[]` spans and fenced code blocks. `False` disables the in-Python recovery layer for byte-perfect TensorZero-pure behaviour. See [structured-outputs.md](structured-outputs.md). |
| `max_json_retries` | `int` | `2` | Bounds the number of extra inferences the JSON path may emit when both TZ's parser and the permissive layer fail. `0` disables retries; negative values raise `ConfigurationError`. Per-call `Agent.invoke(max_json_retries=…)` overrides this for one call. See [structured-outputs.md](structured-outputs.md). |

### `Agent`

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider | None = None,
    ) -> None: ...

    @property
    def config(self) -> AgentConfig: ...

    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        json_mode: Literal["off", "on", "strict", "tool"] | None = None,
        max_json_retries: int | None = None,
    ) -> AgentResult: ...

    async def stream(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
        tags: dict[str, str] | None = None,
        system_arguments: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def warmup(self) -> None: ...
```

| Member | What it does |
|---|---|
| `__init__` | Validates `tool_provider` against the `ToolProvider` protocol when supplied and (when configured) `config.memory` against `MemoryProvider`. Rejects chat-mode agents constructed without a `tool_provider` (JSON-mode agents may omit it — they never dispatch tools). Registers any `config.sub_agents` on an in-process FastMCP server. Never opens connections. |
| `config` | Read-only property returning the `AgentConfig` passed to the constructor. |
| `run` | One-shot run. Loads memory, wraps the tool provider with any guardrails, calls `run_agent`, persists memory, fires `after_run` hooks, and returns an `AgentResult`. When `config.function_kind == "json"`, the tool loop is replaced by a bounded JSON inference orchestration (`run_structured_inference` called up to `max_json_retries + 1` times, with the permissive layer and a corrective hint between attempts) — `before_run` / `after_run` guardrails still fire, but `before_tool` / `after_tool` never do. Optional `tags` shallow-merge on top of `config.tags` (per-call values win on key collisions). Optional `system_arguments` is forwarded to TensorZero as `input.system` for the system-template render (the dict matches the function's system schema directly — no `arguments` wrapper); shallow-merged with the memory strategy's contribution, with caller keys winning on collision. Optional `output_schema` / `json_mode` override the config-level values for one call; both are forbidden on chat agents and raise `ConfigurationError` if supplied there. Optional `max_json_retries` overrides `config.max_json_retries` for one call (JSON agents only; harmless on chat agents). |
| `stream` | Streaming analogue. Yields `AgentEvent`s in real time and ends with a terminal `DoneEvent(result=AgentResult(...))`. Same `tags` / `system_arguments` per-call kwargs as `run`. See [streaming.md](streaming.md). |
| `warmup` | Optional lifespan-phase readiness hook called by `t0_agents.serve.create_app` at startup. Default implementation is a no-op; override in a subclass to front-load probes. See [serving.md](serving.md). |

### `AgentResult`

| Field | Type | Description |
|---|---|---|
| `response` | `str` | Final text from the terminating turn. Joined from `Text` blocks, or `Thought` blocks if no text was produced. For JSON runs (`function_kind == "json"`) it is the raw JSON string the model produced. |
| `episode_id` | `str \| None` | TensorZero episode id. Captured from the first inference response. `None` only if no inference ran. |
| `inference_ids` | `list[str]` | One inference id per turn, in order. Pass these to `t0.feedback(...)` to attribute outcomes to specific turns. JSON runs accumulate one id per retry attempt: `1 + json_parse_attempts` entries total. |
| `turns` | `int` | Number of inference turns the loop executed. `1` means a single inference with no tool calls. JSON runs use `1 + json_parse_attempts` to reflect each retry as a fresh turn. |
| `response_source` | `Literal["text", "thought"]` | `"text"` when the response came from `Text` blocks; `"thought"` when it fell back to reasoning blocks. JSON runs stay `"text"` — they emit no thought blocks. |
| `guardrail_log` | `list[GuardrailLogEntry]` | Per-hook audit record (empty when no guardrails are configured). See [guardrails.md](guardrails.md). |
| `function_kind` | `Literal["chat", "json"]` | Echoes `AgentConfig.function_kind`. `"chat"` for the tool-loop path, `"json"` for single-shot JSON runs. |
| `parsed` | `Any \| None` | The JSON object recovered for a JSON run. `None` for chat runs and for JSON runs where every recovery layer failed (`parse_strategy == "none"`). See [structured-outputs.md](structured-outputs.md). |
| `parse_strategy` | `Literal["strict", "permissive", "retry", "none"]` | For JSON runs, reports which layer produced `parsed`: `"strict"` (TensorZero's parser), `"permissive"` (the in-Python recovery layer), `"retry"` (a re-emitted inference succeeded), or `"none"` (every attempt and every layer failed; `parsed` is `None`). Defaults to `"strict"` for chat runs (semantically meaningless there). See [structured-outputs.md](structured-outputs.md). |
| `json_parse_attempts` | `int` | For JSON runs, the number of *extra* inferences after the first.  `0` on the happy path, `max_json_retries` at most. `0` for chat runs. |

### Collaborator Protocols

All three live in `t0_agents.core.types` and are `@runtime_checkable`.

```python
class ToolProvider(Protocol):
    async def list_tools(self) -> list[Any]: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any: ...
```

`ToolProvider` is the structural slice of `fastmcp.Client` the agent loop actually uses: tool discovery via `list_tools()` and execution via `call_tool()`. `fastmcp.Client` satisfies it by construction — no adapter, no import of anything extra. Implement it yourself when you want to feed tools from a non-MCP source, or pair it with a test double from [testing.md](testing.md) (the packaged `MockToolProvider` is a working reference implementation).

`MemoryProvider` (four async methods: `load`, `save`, `search`) and `Guardrail` (four async hooks: `before_run`, `before_tool`, `after_tool`, `after_run`) are documented in their own pages: [memory.md](memory.md) and [guardrails.md](guardrails.md). The Protocols live in `core/` so every submodule that consumes them imports from one place; you never need a specific extra installed just to type-check against them.

### `ElicitationForwarder` and `ElicitationRequest`

```python
class ElicitationForwarder:
    def __init__(
        self,
        *,
        on_request: Callable[[RequestEnvelope], Awaitable[None]] | None = None,
        default_timeout: float = 300.0,
    ) -> None: ...

    async def handler(
        self,
        message: str,
        response_type: type[Any] | None,
        params: Any,
        context: Any,
    ) -> ElicitResult[Any]: ...

    async def resolve(self, elicitation_id: str, payload: dict[str, Any]) -> None: ...

    async def cancel_all(self, *, action: Literal["accept", "decline", "cancel"] = "decline") -> None: ...
```

Bridges a FastMCP elicitation handler (which must return inline) to an out-of-process resolver (typically a WebSocket frame from a browser). Constructing it without `on_request` yields a handler that always declines — the safe default for contexts where no return channel is wired. See [serving.md](serving.md) and [streaming.md](streaming.md) for the full wiring recipe.

## Runnable examples by difficulty

### Easy — one-liner

The quickstart above (and [`examples/02_agent_class/`](../examples/02_agent_class/)) is the canonical one-liner: a single `AgentConfig`, a FastMCP server with `get_weather` / `add`, one `agent.invoke(message)` call.

### Intermediate — non-trivial config knobs

Combine observability tags, an experimentation namespace, parallel tool dispatch, and a narrower turn budget on the same agent:

```python
config = AgentConfig(
    name="analytics-assistant",
    function_name="agent",
    max_turns=5,
    parallel_tools=True,                           # dispatch same-turn tool calls concurrently
    tags={"deployment": "analytics", "owner": "platform-team"},
    namespace="analytics",                          # selects the analytics experimentation config
    tool_choice={"specific": "search_papers"},      # force the first turn to use a tool
)
agent = Agent(config, t0=t0, tool_provider=mcp_client)

# Per-call override: merge a request-scoped tag on top of the configured ones.
result = await agent.invoke(
    "Find papers on retrieval-augmented generation.",
    tags={"request_id": "req-42"},
)
```

The `tags` passed to `.run()` shallow-merge on top of `config.tags`; per-call keys win on collisions and the configured deployment label is preserved.

### Advanced — custom `ToolProvider`

`ToolProvider` is structural — anything with the two async methods works. The most complete reference implementation that ships with this library is `t0_agents.testing.MockToolProvider`, documented in [testing.md](testing.md) and used in [`examples/11_testing/`](../examples/11_testing/). Use it as the template when you want to:

- feed tools into an `Agent` from a source that isn't a FastMCP server,
- write fast, deterministic unit tests without Docker, or
- inject fault-injection logic between the agent loop and your real backend.

For real MCP deployments, `fastmcp.Client` satisfies the Protocol out of the box and is the expected production path.

## Error handling

| Exception | Inherits from | Raised when |
|---|---|---|
| `T0AgentsError` | `Exception` | Base class for every exception this library raises. Catch this to catch anything originating inside t0-agents. |
| `T0AgentsFatalError` | `T0AgentsError` | Marker base for exceptions that must abort a run rather than be stringified into a tool error. |
| `ConfigurationError` | `T0AgentsError` | `AgentConfig.__post_init__` or `Agent.__init__` rejects an invalid value (empty `name`/`function_name`, non-positive `max_turns`, non-positive non-`None` `tool_timeout`, invalid `memory_strategy`, a `tool_provider` that does not match `ToolProvider`, a `config.memory` that does not match `MemoryProvider`, a non-`Agent` entry in `sub_agents`, or a sub-agent name collision). |
| `MaxTurnsExceededError` | `T0AgentsError` | The loop ran for `config.max_turns` turns without producing a tool-call-free response. Carries `.max_turns` on the instance. |
| `GuardrailBlockedError` | `T0AgentsFatalError` | A guardrail hard-aborted the run. Propagates past the loop's broad exception handler so it is never stringified into a tool result. See [guardrails.md](guardrails.md). |
| `SubAgentError` | `T0AgentsError` | A delegated sub-agent raised a non-fatal exception. The original exception is preserved as `__cause__`. See [multi-agent.md](multi-agent.md). |

Tool-call errors from inside the MCP server (wrong name, bad arguments, `ToolError`, `McpError`, unexpected exceptions) are never raised — they are stringified into a `ToolResult("ERROR: …")` payload so the LLM sees the failure and can retry on the next turn. That contract lives in `mcp/loop.py` and is preserved through `Agent.invoke`. Cross-cutting retry, logging, and feedback patterns are documented in [error-handling.md](error-handling.md).

## Production considerations

- **Lifecycle.** `Agent` holds references only. Keep the `async with` around your gateway and tool provider alive for the full span of `.run()` / `.stream()` calls.
- **Per-session MCP clients with elicitation.** A single `fastmcp.Client` accepts concurrent `call_tool` invocations safely, but its `elicitation_handler` is a process-wide singleton. If any tool may call `ctx.elicit(...)`, every concurrent caller needs its own `Client` (and therefore its own `Agent`). The HTTP/WebSocket serving layer enforces this automatically; see [serving.md](serving.md).
- **Warmup.** `t0_agents.serve.create_app` calls `await agent.warmup()` during FastAPI startup. Override it in a subclass to front-load probes (pinging an auxiliary backend, warming caches). The default implementation is intentionally empty because the gateway's own healthcheck and the first `/chat` surface any real failure.
- **Concurrency.** Sequential `.run()` / `.stream()` calls on the same `Agent` are safe. Concurrent calls are safe only when no tool uses elicitation (see above) and when the underlying `t0` and tool provider are themselves safe to share.
- **Dry runs.** `AgentConfig(dryrun=True)` forwards `dryrun=True` to every `t0.inference(...)` call — useful for contract tests that still want real inference shapes without writing to ClickHouse or billing the provider.

Deployment topologies (single-process, per-surface servers, sticky-session load balancing) are covered in [production.md](production.md).

## Observability

- **`episode_id`** correlates all turns of a single run in TensorZero's ClickHouse tables. Pass it to a follow-up `Agent.invoke(..., episode_id=result.episode_id)` to continue the episode.
- **`inference_ids`** gives you one id per turn — the granularity feedback calls operate on. Call `t0.feedback(metric_name=..., inference_id=id, value=...)` directly to attach outcome signals.
- **`tags`** attaches free-form metadata to every inference. Set them on `AgentConfig.tags` for deployment-wide values and on `Agent.invoke(tags=...)` / `Run.events(tags=...)` for per-call overrides; per-call keys win on collisions and configured tags are preserved otherwise.
- **`namespace`** selects an experimentation block in `tensorzero.toml`. TensorZero also auto-stores it as the `tensorzero::namespace` tag, so you don't need to duplicate it in `tags` for cost accounting.
- **`AgentResult.guardrail_log`** records every guardrail hook activation for audit and manual `t0.feedback(...)` emission. Empty when no guardrails are configured.

Cross-cutting observability patterns (dashboards, ClickHouse queries, PromQL) live in [observability.md](observability.md).

---

Internally every `Agent.invoke()` / `Run.events()` call delegates to the `run_agent` loop in `t0_agents.mcp`. The next chapter opens that loop and walks through it step by step — tool discovery, inference, execution, error mapping — and names the design decisions the `Agent` class inherits. See [Concepts](concepts.md).

## External references

- [TensorZero gateway API reference](https://www.tensorzero.com/docs/gateway/api-reference) — `t0.inference(...)` accepts the configuration fields this module forwards verbatim (`tags`, `namespace`, `dryrun`, `parallel_tools`, `tool_choice`, `additional_tools`) and returns the `ChatInferenceResponse` shape `AgentResult` is assembled from.
- [TensorZero episodes guide](https://www.tensorzero.com/docs/gateway/guides/episodes) — the lifecycle model behind `AgentResult.episode_id`. There is no explicit close API; continuity is maintained by passing the id back on the next run.
- [FastMCP `Client`](https://gofastmcp.com/python-sdk/fastmcp-client-client.md) — the class that satisfies `ToolProvider` structurally. `list_tools`, `call_tool`, and `elicitation_handler` are the exact surfaces `Agent` relies on.
- [FastMCP elicitation](https://gofastmcp.com/clients/elicitation.md) — the human-in-the-loop primitive `ElicitationForwarder` bridges to a remote channel.
