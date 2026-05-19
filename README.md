# t0-agents

<p align="center">
  <img src="docs/assets/hero.png" alt="t0-agents" width="420">
</p>

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).
Build LLM agents that discover and execute tools through
[MCP](https://modelcontextprotocol.io/) (Model Context Protocol), while
TensorZero handles inference routing, episode tracking, and
ClickHouse-backed observability for you.

> **Status**: Alpha. APIs are stabilising. Pinned against `tensorzero >= 2026.4.0` and `fastmcp >= 3.2.3`.

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
streaming, multi-agent, HTTP serving, the terminal TUI, and the test
harness — is opt-in.

## The three v1.0 abstractions

Every agentic feature in t0-agents sits on top of one of these three
primitives, and you can adopt them piece by piece:

```python
from t0_agents import Agent, AgentConfig

agent = Agent(AgentConfig(name="assistant", function_name="agent"),
              t0=t0, tool_provider=client)

# 1. Run[T] — the first-class invocation handle.
run = agent.invoke("Plan my trip to Lisbon under $1500.")
result = await run                    # final AgentResult, awaitable
async for event in run.events():      # live AgentEvent stream
    ...
await run.cancel()                    # cooperative cancellation
forked = await run.fork("Make it Madrid instead.")

# 2. Middleware — composable cross-cutting concerns.
class TimingMiddleware:
    name = "timing"
    async def __call__(self, ctx, next_):
        start = time.perf_counter()
        result = await next_(ctx)
        print(f"{ctx.config.name} took {time.perf_counter() - start:.2f}s")
        return result

timed = Agent(AgentConfig(
    name="assistant",
    function_name="agent",
    middleware=[TimingMiddleware()],     # plugs into the chain
), t0=t0, tool_provider=client)
await timed.invoke("Plan my trip to Lisbon under $1500.")

# 3. OutputMode[T] — polymorphic outputs, typed end-to-end.
from t0_agents import JsonOutput, StructuredOutput
from pydantic import BaseModel

class Customer(BaseModel):
    name: str
    email: str

extractor = Agent(AgentConfig(
    name="extractor",
    function_name="extract_customer",
    output=StructuredOutput(Customer),     # T = Customer
), t0=t0)
result = await extractor.invoke("Sarah <sarah@example.com>")
result.value.email                          # statically typed Customer

```

Each pairs with one chapter of the docs: see
[Run](docs/run.md) · [Middleware](docs/middleware.md) ·
[Output Modes](docs/output-modes.md).

Multi-agent topologies are built directly from these primitives plus
plain `asyncio`.  Three canonical patterns — Sequential, Parallel,
and Orchestrator — live as runnable examples under
[`examples/12_concurrency/`](examples/12_concurrency/) (legacy Agent
API) and [`examples/glow/`](examples/glow/) (decorator API).  See
[Multi-Agent](docs/multi-agent.md) for the wiring details.

## Features

Every layer is opt-in.  The base install gets you the core stack;
heavier extras live behind named installs.

| Capability | What it gives you |
| --- | --- |
| **MCP adapter & loop** | Convert FastMCP tool definitions to TensorZero's inference format, orchestrate the turn-based inference-execute loop, serialize tool results back.  Low-level `run_agent` for full control. |
| **`Run[T]` handles** | `agent.invoke(msg)` returns a `Run` you can `await`, iterate with `.events()`, cancel, replay, fork, and attach feedback to. |
| **Middleware chain** | Memory, guardrails, cost tracking, sub-agent orchestration, JSON retry — all composable middlewares; add your own without forking `core/agent.py`. |
| **`OutputMode[T]` polymorphism** | `ChatOutput`, `JsonOutput`, `StructuredOutput[M]` (Pydantic), `ToolCallOutput`, `StreamingOutput`.  Custom modes (YAML, Instructor, SQLAlchemy) plug in via the same protocol. |
| **Cross-run memory** | `InMemoryMemory`, `SqliteMemory`, `TensorZeroMemory`, or your own `MemoryProvider`.  Keyed by TensorZero's `episode_id`. |
| **Policy guardrails** | `require_approval`, `max_cost`, `max_tool_calls`, `block_tools` out of the box; custom `Guardrail` Protocol for domain rules.  Every decision is captured on `AgentResult.guardrail_log` for audit. |
| **Multi-agent composition** | Register sub-agents on `AgentConfig.sub_agents` and they become `{name}_ask` tools on the orchestrator's MCP server.  Nested orchestrators, `parent_episode_id` correlation. |
| **HTTP serving** | `create_app(agent)` wraps any agent in a FastAPI app with `/chat`, `/feedback`, `/health`, `/info` — plus `/chat/stream` (SSE) and `/chat/ws` (WebSocket) when `streaming=True`. |
| **Terminal CLI** | `t0-agents run` (interactive streaming TUI), `t0-agents serve` (dev HTTP server), `t0-agents info` (config introspection). |
| **First-class testing** | `MockGateway` + `MockToolProvider` for deterministic unit tests, `EmbeddedGatewayHelper` for real-inference integration tests without Docker, ten `assert_*` helpers. |
| **Observability** | Every run returns the TensorZero `episode_id`, per-turn `inference_ids`, the guardrail log, aggregated cost / usage, and a `response_source` signal. |

## Install

You pay only for what you use.  Base = adapter + loop + `Agent` +
`InMemoryMemory` + guardrails + streaming + composition + testing.
Everything that pulls a heavier third-party dependency lives behind an
optional extra.

```bash
# Base install (no Docker, no FastAPI, no SQLite).
pip install t0-agents

# Adds aiosqlite for cross-process memory persistence.
pip install 't0-agents[memory-sqlite]'

# Adds FastAPI + Uvicorn + Pydantic + sse-starlette for HTTP serving.
pip install 't0-agents[serve]'

# Adds Typer + Rich on top of [serve]; registers the `t0-agents` script.
pip install 't0-agents[cli]'

# Adds Pydantic for StructuredOutput[M].
pip install 't0-agents[pydantic]'

# Combine extras as needed:
pip install 't0-agents[memory-sqlite,serve,pydantic]'
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add t0-agents                                 # base
uv add 't0-agents[memory-sqlite,cli,pydantic]'   # with extras
```

## Quickstart

```python
import asyncio

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

mcp = FastMCP("demo")

@mcp.tool
def get_specialty_today() -> str:
    """Today's specialty coffee at the shop."""
    return "Ethiopian Yirgacheffe pour-over"

@mcp.tool
def estimate_wait_minutes() -> int:
    """How many minutes the queue is right now."""
    return 7

async def main() -> None:
    gateway = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )
    async with gateway as t0, Client(mcp) as client:
        agent = Agent(
            AgentConfig(name="barista", function_name="agent"),
            t0=t0,
            tool_provider=client,
        )
        # `invoke` returns a Run handle — awaitable for the final
        # AgentResult; iterate `run.events()` to watch the loop live;
        # `run.cancel()` stops it cooperatively.
        result = await agent.invoke(
            "What should I order, and how long will I wait?"
        )
        print(result.response)
        print(f"episode_id={result.episode_id} turns={result.turns}")

asyncio.run(main())
```

The agent figures out it needs both tools, picks the order, and
synthesises an answer like *"The Ethiopian Yirgacheffe pour-over is on
today, and the queue is about a 7-minute wait."*

Prerequisites: a local TensorZero gateway (`docker compose up -d` with
a provider API key in `.env`).  The step-by-step setup lives in
[Getting Started](docs/getting-started.md).

Prefer a no-Docker first try?  The `t0_agents.testing` submodule ships
a `MockGateway` you can drive the exact same `Agent` against — see
[Testing](docs/testing.md).

## Documentation

The public docs are written as a **progressive tour**: each chapter
picks up the tension the previous one left open.  Start at
[the index](docs/index.md) for the full Reading path and Appendices.

**Reading path**
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
- [CLI](docs/cli.md) — the `t0-agents` terminal experience.
- [Testing](docs/testing.md) — keep every layer above from regressing.

**Appendices**
- [API Reference](docs/api-reference.md) · [Migration v0.12 → v1.0](docs/migration-v0.12-to-v1.0.md) · [Error Handling](docs/error-handling.md) · [Observability](docs/observability.md) · [Production](docs/production.md) · [Concurrency](docs/concurrency.md) · [Multi-Server](docs/multi-server.md) · [Structured outputs (reference)](docs/structured-outputs.md)

**Guides** — production patterns that compose t0-agents with adjacent tooling:
- [Elicitation in your FastAPI / SSE backend](docs/guides/elicitation-in-api.md)
- [Elicitation in your FastMCP server](docs/guides/elicitation-in-mcp-server.md)

Runnable examples for every chapter live under [`examples/`](examples/).

## Experimental: `t0_agents.glow`

A Pythonic skin over the v1.0 primitives — `Agent`, `Run[T]`,
`OutputMode[T]` — lives on the
[`experimental/glow`](https://github.com/sproutfi/t0-agents/tree/experimental/glow)
branch.  Where the base API takes ~25 lines to wire an agent, glow takes
four: a function with a docstring (the system prompt) and a return-type
annotation (the `OutputMode`), decorated with `@app.agent`.  Adds
FastAPI-style `Depends` injection for tools and sub-agents and typed
MCP declarations via class-based or `Annotated[...]` metadata.

The tutorial walks every public symbol — see
[`docs/glow/getting-started.md`](docs/glow/getting-started.md) on the
experimental branch.  Public API is **unstable** and may change before
v1.1; eight runnable examples ship under
[`examples/glow/`](examples/glow/).  Install with the extra
(`pip install 't0-agents[glow]'`).

## Requirements

- Python 3.11, 3.12, or 3.13
- Docker + Docker Compose (for the TensorZero gateway, required by every example beyond the low-level adapter layer)
- An LLM provider API key (Groq, OpenAI, Anthropic, …) — the default `docker-compose.yml` expects `GROQ_API_KEY` in `.env`

## Development

```bash
uv sync                                     # install dev dependencies
uv run pytest -m "not integration and not embedded"   # offline unit tests
uv run pytest -m integration                # requires docker compose up
uv run pytest -m embedded                   # requires RUN_EMBEDDED_TESTS=1 + provider key
uv run ruff check src/ tests/               # lint
uv run mypy src/                            # type check (strict mode)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full pull-request workflow and the test-mode cheat-sheet.

## License

MIT.  See [pyproject.toml](pyproject.toml) and [CHANGELOG.md](CHANGELOG.md).
