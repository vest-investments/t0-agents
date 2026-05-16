# t0-agents

<p align="center">
  <img src="docs/assets/hero.png" alt="t0-agents" width="420">
</p>

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).

Build LLM agents that discover and execute tools through [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) while inheriting TensorZero's structured inference, episode tracking, and ClickHouse-backed observability. Every layer of the agent stack — memory, guardrails, streaming, multi-agent composition, HTTP serving, a terminal TUI, and a test harness — is opt-in so you only install what you need.

> **Status**: Alpha. APIs are stabilising. Pinned against `tensorzero >= 2026.4.0` and `fastmcp >= 3.2.3`.

## Features

| Capability | What it gives you |
| --- | --- |
| **MCP adapter & loop** | Convert FastMCP tool definitions to TensorZero's inference format, orchestrate the turn-based inference-execute loop, serialize tool results back. Low-level `run_agent` function for full control. |
| **Declarative `Agent`** | One-liner `Agent(config, t0=..., tool_provider=...)` for chat-mode agents — drop the `tool_provider` for JSON-mode single-shot agents. `AgentConfig` plus structurally-typed Protocols for tools, memory, and guardrails. No inheritance required. |
| **Cross-run memory** | `InMemoryMemory` for ephemeral state, `SqliteMemory` for persistence, or plug in Redis / a vector store by implementing the three-method `MemoryProvider` Protocol. Keyed by TensorZero's `episode_id`. |
| **Policy guardrails** | `require_approval`, `max_cost`, `max_tool_calls`, `block_tools` out of the box; custom `Guardrail` Protocol for domain rules. Every decision is captured on `AgentResult.guardrail_log` for audit. |
| **Real-time streaming** | `Agent.stream()` yields a typed `AgentEvent` iterator — text deltas, tool calls, thoughts, elicitation prompts, completion. Consumable directly, over Server-Sent Events, or over WebSocket. |
| **Multi-agent composition** | Register sub-agents on `AgentConfig.sub_agents` and they become `{name}_ask` tools on the orchestrator's MCP server. Nested orchestrators, `parent_episode_id` correlation, hybrid-fatal error model. |
| **HTTP serving** | `create_app(agent)` wraps any agent in a FastAPI application with `/chat`, `/feedback`, `/health`, `/info` — and `/chat/stream` (SSE) + `/chat/ws` (WebSocket) when `streaming=True`. Per-session `Client` model for safe live elicitation. |
| **Terminal CLI** | `t0-agents run` (interactive streaming TUI), `t0-agents serve` (dev HTTP server), `t0-agents info` (config introspection) — all driven by a two-line agent factory. |
| **First-class testing** | `MockGateway` + `MockToolProvider` for deterministic unit tests, `EmbeddedGatewayHelper` for real-inference integration tests without Docker, five `assert_*` helpers, ready-made pytest fixtures. |
| **Observability** | Every run returns the TensorZero `episode_id`, per-turn `inference_ids`, the guardrail log, and a `response_source` signal. Submit outcomes via `t0.feedback(...)`; filter ClickHouse by deployment tag. |

## Install

`t0-agents` is designed so you pay only for what you use. The base install covers the whole *core* stack (adapter, loop, `Agent`, `InMemoryMemory`, guardrails, streaming, composition, testing). Everything that pulls in a heavier third-party dependency — SQLite, FastAPI, Uvicorn, Typer, Rich — lives behind an optional extra.

```bash
# Base — everything except the SQLite memory backend, HTTP serving, and CLI.
pip install t0-agents

# Base + persistent memory (adds aiosqlite).
pip install 't0-agents[memory-sqlite]'

# Base + HTTP serving (adds FastAPI, Uvicorn, Pydantic, sse-starlette).
pip install 't0-agents[serve]'

# Base + terminal CLI (adds Typer and Rich on top of the serve extras; registers the `t0-agents` console script).
pip install 't0-agents[cli]'

# Combine extras as needed:
pip install 't0-agents[memory-sqlite,serve]'
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add t0-agents                          # base
uv add 't0-agents[memory-sqlite,cli]'     # with extras
```

### What each extra adds

| Extra | Dependencies | Unlocks |
| --- | --- | --- |
| *(none)* | `fastmcp`, `tensorzero` | MCP adapter & loop, `Agent`, `InMemoryMemory`, guardrails, streaming primitives, multi-agent composition, and the `t0_agents.testing` submodule. |
| `memory-sqlite` | `aiosqlite` | `t0_agents.memory.SqliteMemory` for cross-process memory persistence. |
| `serve` | `fastapi`, `uvicorn[standard]`, `pydantic`, `sse-starlette` | `t0_agents.serve.create_app` + the `/chat`, `/feedback`, `/health`, `/info` endpoints (and `/chat/stream` + `/chat/ws` when `streaming=True`). |
| `cli` | *(everything in `serve`)* + `typer`, `rich` | `t0-agents` console script (`run`, `serve`, `info`) and the Rich-based TUI renderer. |

## Quickstart

```python
import asyncio

from fastmcp import Client, FastMCP
from tensorzero import AsyncTensorZeroGateway

from t0_agents import Agent, AgentConfig

mcp = FastMCP("demo")

@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

async def main() -> None:
    gateway = AsyncTensorZeroGateway.build_http(
        gateway_url="http://localhost:3000",
    )
    async with gateway as t0, Client(mcp) as client:
        agent = Agent(
            AgentConfig(name="assistant", function_name="agent"),
            t0=t0,
            tool_provider=client,
        )
        result = await agent.run("What is 3 + 7?")
        print(result.response)     # e.g. "The answer is 10."
        print(result.episode_id)   # UUID — use it for follow-up feedback
        print(result.turns)        # inference turns consumed

asyncio.run(main())
```

Prerequisites: a local TensorZero gateway (`docker compose up -d` with a provider API key in `.env`). The step-by-step setup lives in [Getting Started](docs/getting-started.md).

Prefer a no-Docker first-try? The `t0_agents.testing` submodule ships a `MockGateway` you can drive the exact same `Agent` against — see [Testing](docs/testing.md).

## Documentation

The public documentation is written as a **progressive tour**: each chapter picks up the tension the previous one leaves open. Start at [the index](docs/index.md) for the full Reading path and the Appendices.

**Reading path**
- [Getting Started](docs/getting-started.md) — install and run your first agent.
- [Agent](docs/agent.md) — the declarative `Agent` class and its collaborator Protocols.
- [Concepts](docs/concepts.md) — the `run_agent` loop, turn by turn.
- [Elicitation](docs/elicitation.md) — tools that pause to ask humans questions.
- [Memory](docs/memory.md) — remember across turns and across processes.
- [Guardrails](docs/guardrails.md) — constrain what the agent may do.
- [Streaming](docs/streaming.md) — watch the loop run in real time.
- [Multi-Agent Composition](docs/multi-agent.md) — orchestrators and sub-agents.
- [Serving](docs/serving.md) — expose an agent over HTTP.
- [CLI](docs/cli.md) — interact with any agent from the terminal.
- [Testing](docs/testing.md) — keep the whole stack from regressing.

**Appendices**
- [API Reference](docs/api-reference.md) · [Error Handling](docs/error-handling.md) · [Observability](docs/observability.md) · [Production](docs/production.md) · [Multi-Server Composition](docs/multi-server.md)

Runnable examples for every chapter live under [`examples/`](examples/).

## Requirements

- Python 3.11, 3.12, or 3.13
- Docker + Docker Compose (for the TensorZero gateway, required by every example beyond the adapter layer)
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

MIT. See [pyproject.toml](pyproject.toml) and [CHANGELOG.md](CHANGELOG.md).
