# t0-agents

<p align="center">
  <img src="docs/assets/hero.png" alt="t0-agents" width="420">
</p>

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).
Build LLM agents that discover and execute tools through
[MCP](https://modelcontextprotocol.io/) (Model Context Protocol), while
TensorZero handles inference routing, episode tracking, and
ClickHouse-backed observability for you.  Two surfaces, one engine:
`t0_agents.glow` for the four-line decorator path, the canonical
`Agent` surface for the moments you need every escape hatch.

> **Status**: Beta, built and used in production by Vest. Requires `tensorzero >= 2026.6.0` and `fastmcp >= 3.2.3`.

## Four lines, one agent.

```python
from t0_agents.glow import App
app = App(t0=t0)

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""

print(await tutor("How do I say hello in formal Japanese?"))
```

The docstring is the system prompt.  The return annotation picks the
[`OutputMode`](docs/output-modes.md).  `await tutor(...)` returns the
unwrapped string directly.  Install with the `[glow]` extra
([Install](#install)).  Full tour at
[`docs/glow/`](docs/glow/index.md).

## When you need the controls — drop down to Agent.

```python
from t0_agents import Agent, AgentConfig

# Construct once, run many.
agent = Agent(AgentConfig(
    name="assistant",
    function_name="agent",
    middleware=[TimingMiddleware(), MyGuardrails()],   # Starlette-style chain
), t0=t0, tool_provider=client)

# invoke(...) returns a Run[T] — the first-class invocation handle.
run = agent.invoke("Plan my trip to Lisbon under $1500.")

# Iterate live AgentEvent stream:
async for event in run.events():
    ...

# Cooperative cancellation:
await run.cancel()

# Await the terminal AgentResult:
result = await run                      # episode_id, inference_ids, cost
forked = await run.fork("Make it Madrid instead.")

# Polymorphic outputs — typed end-to-end.
from t0_agents import StructuredOutput
from pydantic import BaseModel

class Customer(BaseModel):
    name: str
    email: str

extractor = Agent(AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=StructuredOutput(Customer),
), t0=t0)
parsed = (await extractor.invoke("Sarah <sarah@example.com>")).value
parsed.email                            # statically typed Customer
```

And toolless chat works the same way — drop `tool_provider`,
get a pure-conversation `Agent` with zero MCP plumbing:

```python
chat = Agent(AgentConfig(name="tutor", function_name="agent"), t0=t0)
result = await chat.invoke("Explain monads in two sentences.")
print(result.response)
```

No `Client`, no FastMCP server, no `list_tools()` roundtrip per turn —
the loop runs one inference and returns the text response.

Glow compiles down to exactly this `Agent`.  Every glow wrapper
exposes `_agent` — the same v1.0 surface — for the moments the
decorator is in your way.  See
[`docs/glow/escape-hatch.md`](docs/glow/escape-hatch.md).

Each axis pairs with one chapter of the docs:
[Run](docs/run.md) · [Middleware](docs/middleware.md) ·
[Output Modes](docs/output-modes.md).  Multi-agent topologies are
built directly from these primitives plus plain `asyncio` — see the
runnable patterns under
[`examples/agent/12_concurrency/`](examples/agent/12_concurrency/)
and [`examples/glow/`](examples/glow/), documented in
[Multi-Agent](docs/multi-agent.md).

## Why t0-agents

Most agentic frameworks try to be everything at once.  They reimplement
inference routing, prompt versioning, tool discovery, observability and
human-in-the-loop — and as a result they reimplement most of those
things badly.

t0-agents takes the opposite bet.  Two upstream projects already solved
the hard parts:

- **TensorZero** owns inference: gateway routing, provider fallback,
  prompt versions, episode IDs, ClickHouse analytics, A/B testing.
- **FastMCP** (Model Context Protocol) owns tools: dynamic discovery,
  schema evolution, sticky-session HTTP transport, the elicitation
  primitive for human-in-the-loop.

t0-agents is the thin glue between them.  The `mcp/` submodule fits in
~100 lines of production code; everything above it — `Agent`,
`Run[T]`, `Middleware`, `OutputMode[T]`, memory, guardrails,
streaming, multi-agent, HTTP serving, the test harness, and the
`glow` decorator skin — is opt-in.

## Install

You pay only for what you use.  Base = adapter + loop + `Agent` +
`InMemoryMemory` + guardrails + streaming + composition + testing.
Everything that pulls a heavier third-party dependency lives behind an
optional extra.  Install from PyPI:

```bash
# Base install (no Docker, no FastAPI, no SQLite).
pip install t0-agents

# With the extras you need:
pip install "t0-agents[glow,memory-sqlite,serve,pydantic]"
```

| Extra | Adds |
| --- | --- |
| `glow` | pydantic + the `@app.agent` decorator surface (the four-line path) |
| `memory-sqlite` | aiosqlite for cross-process memory and usage persistence |
| `serve` | FastAPI + Uvicorn + Pydantic + sse-starlette for HTTP serving |
| `pydantic` | Pydantic for `StructuredOutput[M]` |

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add "t0-agents[glow]"
```

Or straight from a release tag:

```bash
pip install "t0-agents[glow] @ git+https://github.com/vest-investments/t0-agents.git@v2.14.1"
```

## Documentation

The public docs are written as a **progressive tour**: each chapter
picks up the tension the previous one left open.  Start at
[the index](docs/index.md) for the full Reading path and Appendices.

**glow — the four-line path** (start here)
- [Getting Started](docs/glow/getting-started.md) — toolless chat, structured outputs by return type, per-call overrides, multi-agent patterns.
- [Streaming and events](docs/glow/streaming-and-events.md) — `FluentRun.events(*selectors)` and the seven typed selectors.
- [Depends and tools](docs/glow/depends-and-tools.md) — value-style injection and sub-agent injection via `Depends(...)`.
- [Typed MCP and guardrails](docs/glow/typed-mcp-and-guardrails.md) — class-based `MCPServer` and `Annotated[T, guardrail(...)]` metadata.
- [Escape hatch](docs/glow/escape-hatch.md) — `wrapper._agent` IS the v1.0 `Agent`; lifespan + drop-down recipes.
- [Known limitations](docs/glow/limitations.md) — what glow does not do yet.

**Agent — the canonical, fully-typed surface**
- [Getting Started](docs/getting-started.md) — install and run your first agent.
- [Agent](docs/agent.md) — the declarative `Agent` class and its collaborator Protocols.
- [Run](docs/run.md) — `agent.invoke(msg)` returns `Run[T]`: await, iterate events, cancel, replay, fork.
- [Middleware](docs/middleware.md) — composable cross-cutting concerns (memory, guardrails, cost, retry…).
- [Output Modes](docs/output-modes.md) — chat strings, JSON dicts, Pydantic models, tool-call routers, streaming.
- [Concepts](docs/concepts.md) — pop the hood: what `agent.invoke()` actually does, turn by turn.
- [Elicitation](docs/elicitation.md) — tools that pause to ask humans questions.
- [Memory](docs/memory.md) — remember across turns and across processes.
- [Guardrails](docs/guardrails.md) — constrain what the agent may do.
- [Streaming](docs/streaming.md) — narrate the loop live with typed events, SSE, and WebSocket.
- [Multi-Agent Composition](docs/multi-agent.md) — declarative `sub_agents=[...]`, the LLM-driven path.
- [Serving](docs/serving.md) — `create_app(agent)` and the FastAPI surface.
- [Testing](docs/testing.md) — keep every layer above from regressing, including in-memory MCP mounts.

**Appendices**
- [API Reference](docs/api-reference.md) · [Migration v0.12 → v1.0](docs/migration-v0.12-to-v1.0.md) · [Error Handling](docs/error-handling.md) · [Observability](docs/observability.md) · [Production](docs/production.md) · [Concurrency](docs/concurrency.md) · [Multi-Server](docs/multi-server.md) · [Structured outputs (reference)](docs/structured-outputs.md)

**Guides** — production patterns that compose t0-agents with adjacent tooling:
- [Elicitation in your FastAPI / SSE backend](docs/guides/elicitation-in-api.md)
- [Elicitation in your FastMCP server](docs/guides/elicitation-in-mcp-server.md)

Runnable examples for every chapter live under [`examples/`](examples/) —
[`examples/glow/`](examples/glow/) is the 8-example progressive tour
on top of the decorator surface; [`examples/agent/`](examples/agent/)
has 15 area-by-area examples on top of the canonical `Agent` surface.

## Features

Every layer is opt-in.  The base install gets you the core stack;
heavier extras live behind named installs.

| Capability | What it gives you |
| --- | --- |
| **MCP adapter & loop** | Convert FastMCP tool definitions to TensorZero's inference format, orchestrate the turn-based inference-execute loop, serialize tool results back.  Low-level `run_agent` for full control. |
| **`Run[T]` handles** | `agent.invoke(msg)` returns a `Run` you can `await`, iterate with `.events()`, cancel, replay, fork, and attach feedback to. |
| **Middleware chain** | Memory, guardrails, cost tracking, usage ledger, sub-agent orchestration, JSON retry — all composable middlewares; add your own without forking `core/agent.py`. |
| **`OutputMode[T]` polymorphism** | `ChatOutput`, `JsonOutput`, `StructuredOutput[M]` (Pydantic), `ToolCallOutput`, `StreamingOutput`.  Custom modes (YAML, Instructor, SQLAlchemy) plug in via the same protocol. |
| **glow decorator surface** | FastAPI-style `@app.agent` over the v1.0 primitives.  Docstring = system prompt, return annotation = `OutputMode`, `Depends(...)` for tools and sub-agents, `Annotated[T, guardrail(...)]` for policy as type metadata. |
| **Cross-run memory** | `InMemoryMemory`, `SqliteMemory`, `TensorZeroMemory`, or your own `MemoryProvider`.  Keyed by TensorZero's `episode_id`. |
| **Elicitation forwarding** | `ElicitationForwarder` bridges MCP `ctx.elicit()` requests to your transport (FastAPI/SSE, WebSocket, queue). Pause a tool to ask a human, resume the loop with the response. See [`docs/elicitation.md`](docs/elicitation.md) and [`docs/guides/elicitation-in-api.md`](docs/guides/elicitation-in-api.md). |
| **Policy guardrails** | `require_approval`, `max_cost`, `max_tool_calls`, `block_tools` out of the box; custom `Guardrail` Protocol for domain rules.  Every decision is captured on `AgentResult.guardrail_log` for audit. |
| **Multi-agent composition** | Register sub-agents on `AgentConfig.sub_agents` and they become `{name}_ask` tools on the orchestrator's MCP server.  Nested orchestrators, `parent_episode_id` correlation. |
| **HTTP serving** | `create_app(agent)` wraps any agent in a FastAPI app with `/chat`, `/feedback`, `/health`, `/info` — plus `/chat/stream` (SSE) and `/chat/ws` (WebSocket) when `streaming=True`. |
| **First-class testing** | `MockGateway` + `MockToolProvider` for deterministic unit tests, `glow_tools` + `ScriptedElicitation` to drive in-memory MCP mounts and `ctx.elicit` round-trips through the real FastMCP boundary, `EmbeddedGatewayHelper` for real-inference integration tests without Docker, ten `assert_*` helpers. |
| **Observability** | Every run returns the TensorZero `episode_id`, per-turn `inference_ids`, the guardrail log, aggregated cost / usage, and a `response_source` signal. |

## Status & Limitations

t0-agents is **Beta**: the API may still change between minor
versions, and every change is recorded in [CHANGELOG.md](CHANGELOG.md).
Pin a version (`t0-agents==X.Y.Z`) or a release tag when consuming
from another project; see [Install](#install).

Known caveats today:

- **`t0_agents.glow`** ships behind the `[glow]` extra.  What it does
  not do yet is listed in
  [`docs/glow/limitations.md`](docs/glow/limitations.md).
- **JSON output modes do not stream.** `JsonOutput` and
  `StructuredOutput` are single-shot by design; constructing
  `StreamingOutput(base=JsonOutput(...))` raises `ConfigurationError`.
- **Async-only.** The agent loop, `Run`, streaming, serving, and
  testing surfaces are all `async`-first; there is no sync facade.
  Bridge with `asyncio.run(...)` at your entry point if you need one.
- **Multi-agent narrative is moving toward `asyncio`.** The
  declarative `AgentConfig.sub_agents` path plus the runnable patterns
  under [`examples/agent/12_concurrency/`](examples/agent/12_concurrency/)
  are the recommended way to wire orchestrators today.

## Requirements

- Python 3.11, 3.12, or 3.13
- Docker + Docker Compose (for the TensorZero gateway, required by every example beyond the low-level adapter layer)
- An LLM provider API key (Groq, OpenAI, Anthropic, …) — the default `docker-compose.yml` expects `GROQ_API_KEY` in `.env`

## Development

```bash
uv sync --all-extras                        # install dev dependencies and every extra
bash ci/checks.sh                           # the gate every pull request runs
uv run pytest -m "not integration and not embedded"   # offline unit tests
uv run pytest -m integration                # requires docker compose up
uv run pytest -m embedded                   # requires RUN_EMBEDDED_TESTS=1 + provider key
uv run ruff check src/ tests/               # lint
uv run mypy src/                            # type check (strict mode)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full pull-request workflow and the test-mode cheat-sheet.

## License

MIT.  See [LICENSE](LICENSE).
