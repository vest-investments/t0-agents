# Example 09 — Agent with memory

Demonstrates the `memory/` submodule across four difficulty tiers.
Each tier is a standalone script.

## Scripts

| Script | Tier | Needs Docker? | What it shows |
|---|---|---|---|
| [`easy.py`](easy.py) | **Easy** | No | Store primitives on their own: `InMemoryMemory` + `MemoryRecord` + `render_memory_summary` + `search`. No Agent, no TensorZero gateway. |
| [`main.py`](main.py) | **Intermediate** | Yes | Full `Agent` loop with cross-session memory. Runs the same agent twice on the same `episode_id`, once with `InMemoryMemory` and once with `SqliteMemory` (including a simulated process restart). |
| [`tensorzero_backend.py`](tensorzero_backend.py) | **Intermediate** | Yes | `TensorZeroMemory` + `memory_strategy="messages"`: a two-turn conversation that recalls a fact across turns by reading TZ's own observability log — no Postgres, no SQLite, no in-process state. The native messages strategy threads `input.messages` through verbatim, so the user prompt never gets inflated with a rendered summary. |
| [`advanced.py`](advanced.py) | **Advanced** | No | Custom `MemoryProvider` from scratch: per-tenant namespacing + per-episode LRU cap. Standalone, asserts Protocol conformance via `isinstance`. |

Tool-visible memory (an MCP server exposing `remember()` / `recall()`
tools the LLM calls directly) is demonstrated in the adjacent
[`server.py`](server.py), which `main.py` imports.

## Running each tier

```bash
# Easy — no extra setup
python easy.py

# Intermediate (InMemory + SQLite) — boot the gateway and install the SQLite extra
docker compose up -d --env-file .env          # from repo root, requires GROQ_API_KEY
pip install 't0-agents[memory-sqlite]'
python main.py

# Intermediate (TensorZeroMemory) — only the gateway is required
docker compose up -d --env-file .env
python tensorzero_backend.py

# Advanced — no extra setup
python advanced.py
```

## Backends and strategies

| Backend | Persistence | Use case |
|---|---|---|
| `InMemoryMemory` | Process lifetime | Tests, prototyping |
| `SqliteMemory` | Disk (`.db` file) | Single-node production, local dev |
| `TensorZeroMemory` | Gateway-side ClickHouse (TZ owns the writes) | Multi-process or multi-host deployments where TZ is already in the path; avoids running a parallel store |

| Strategy | How it works | When to use |
|---|---|---|
| `"auto"` (default) | Currently resolves to `"prepended_message"` | Most cases |
| `"prepended_message"` | Prepends a text summary to the user message | Works with any TensorZero function or variant |
| `"system_template"` | Passes memory via `input.system.arguments.memory_context` | Your variant declares a MiniJinja system template with a `memory_context` variable |
| `"messages"` | Threads the prior conversation through `input.messages` verbatim | Pair with `TensorZeroMemory`; cleanest multi-turn path; preserves tool calls across turns |

See [`docs/memory.md`](../../docs/memory.md) for the full reference:
concepts, API signatures, error handling, production considerations,
and more custom-backend patterns (Redis sketch, RAG via a vector
store).
