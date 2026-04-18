# Example 09 — Agent with memory

Demonstrates the `memory/` submodule across three difficulty tiers.
Each tier is a standalone script.

## Scripts

| Script | Tier | Needs Docker? | What it shows |
|---|---|---|---|
| [`easy.py`](easy.py) | **Easy** | No | Store primitives on their own: `InMemoryMemory` + `MemoryRecord` + `render_memory_summary` + `search`. No Agent, no TensorZero gateway. |
| [`main.py`](main.py) | **Intermediate** | Yes | Full `Agent` loop with cross-session memory. Runs the same agent twice on the same `episode_id`, once with `InMemoryMemory` and once with `SqliteMemory` (including a simulated process restart). |
| [`advanced.py`](advanced.py) | **Advanced** | No | Custom `MemoryProvider` from scratch: per-tenant namespacing + per-episode LRU cap. Standalone, asserts Protocol conformance via `isinstance`. |

Tool-visible memory (an MCP server exposing `remember()` / `recall()`
tools the LLM calls directly) is demonstrated in the adjacent
[`server.py`](server.py), which `main.py` imports.

## Running each tier

```bash
# Easy — no extra setup
python easy.py

# Intermediate — boot the gateway and install the SQLite extra
docker compose up -d --env-file .env          # from repo root, requires GROQ_API_KEY
pip install 't0-agents[memory-sqlite]'
python main.py

# Advanced — no extra setup
python advanced.py
```

## Backends and strategies

| Backend | Persistence | Use case |
|---|---|---|
| `InMemoryMemory` | Process lifetime | Tests, prototyping |
| `SqliteMemory` | Disk (`.db` file) | Single-node production, local dev |

| Strategy | How it works | When to use |
|---|---|---|
| `"auto"` (default) | Currently resolves to `"prepended_message"` | Most cases |
| `"prepended_message"` | Prepends a text summary to the user message | Works with any TensorZero function or variant |
| `"system_template"` | Passes memory via `input.system.arguments.memory_context` | Your variant declares a MiniJinja system template with a `memory_context` variable |

See [`docs/memory.md`](../../docs/memory.md) for the full reference:
concepts, API signatures, error handling, production considerations,
and more custom-backend patterns (Redis sketch, RAG via a vector
store).
