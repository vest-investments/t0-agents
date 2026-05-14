# 16 - Parallel tool dispatch

Two demos that exercise the three layers of parallelism described in
[`docs/concurrency.md`](../../docs/concurrency.md):

| Demo | Layer | What it shows |
|---|---|---|
| `01_parallel_tools.py` | L1 intra-turn | Same prompt run twice (`parallel_tools=False` then `True`); side-by-side wall-clock timing |
| `02_parallel_subagents.py` | L3 multi-agent | An orchestrator with two sub-agents (`researcher`, `writer`) dispatched concurrently when delegated to in the same turn |

The L2 "batch tool" pattern is illustrated by `server.py` — the
`fetch_urls_batch` tool fans out N requests via `asyncio.gather`
inside a single tool call, sidestepping the case where the LLM emits
one tool per turn.

## Prerequisites

```bash
export GROQ_API_KEY=gsk_...
docker compose up -d           # boots the per-example gateway on :3000
uv sync
```

## Run

```bash
# Demo 01 — intra-turn parallel tool execution
uv run python examples/16_parallel/01_parallel_tools.py

# Demo 02 — multi-agent parallel composition
uv run python examples/16_parallel/02_parallel_subagents.py
```

## What to observe

### `01_parallel_tools.py`

The `fetch_url` tool sleeps for `LATENCY_S = 1.5` seconds per call.

- **`parallel_tools=False`** (sequential) — if the model emits three
  `fetch_url` calls in one turn, they run back-to-back and the run
  takes ~4.5 s.
- **`parallel_tools=True`** (parallel) — the same three calls run
  through `asyncio.gather` and the run finishes in ~1.5 s.

The INFO log on stderr shows `fetch_url START ... fetch_url END`
markers; under parallel dispatch the three START lines arrive before
any END line.

### `02_parallel_subagents.py`

The orchestrator has `parallel_tools=True` and sees `researcher_ask`
and `writer_ask` as MCP tools. When the orchestrator LLM delegates to
both in the same assistant turn, their sub-agent runs overlap. The
elapsed wall-clock approaches the slower of the two sub-agent runs
rather than their sum.

## Caveats and the L2 "batch tool" pattern

Whether `parallel_tools=True` accelerates your run depends on the
model deciding to emit multiple tool calls in one turn. Some
providers (and some smaller models) emit exactly one tool call per
turn, in which case there is nothing for the parallel executor to
parallelise.

When that happens, design the tool surface around the limitation:
expose a single tool that takes a list and parallelises inside.
`server.py` ships `fetch_urls_batch` as a worked example —
`await asyncio.gather(*[_simulate_fetch(u) for u in urls])` — so even
a one-tool-per-turn agent finishes the three fetches in one tool call.

See [`docs/concurrency.md`](../../docs/concurrency.md) for the full
decision table and the patterns the framework deliberately does not
provide.

## Stop the gateway

```bash
docker compose down
```
