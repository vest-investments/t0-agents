# 16 - Parallel tool dispatch

Two demos that exercise the three layers of parallelism described in
[`docs/concurrency.md`](../../docs/concurrency.md):

| Demo | Layer | What it shows |
|---|---|---|
| `01_parallel_tools.py` | L1 intra-turn | Same prompt run twice (`parallel_tools=False` then `True`); side-by-side wall-clock timing |
| `02_parallel_subagents.py` | L3 multi-agent | An orchestrator with two sub-agents (`researcher`, `writer`); side-by-side timing of sequential vs parallel delegation |

The L2 "batch tool" pattern is illustrated by `server.py` — the
`fetch_urls_batch` tool fans out N requests via `asyncio.gather` inside
a single tool call, sidestepping the case where the LLM emits one tool
per turn. It is not invoked by the demos (they exercise L1 / L3) but
ships as a code-level reference.

## Why the demos use a mocked LLM

Whether `parallel_tools=True` accelerates a run depends on the LLM
deciding to emit several tool calls in one assistant turn. Smaller
models (and some providers) emit exactly one tool call per turn — in
that case the parallel executor has nothing to parallelise and the
flag's effect is invisible.

To demonstrate the executor selection deterministically, both demos
replace the real gateway with
[`t0_agents.testing.MockGateway`](../../docs/testing.md), scripted to
emit the multi-tool turn the layer requires (three `fetch_url` calls
for L1; both `researcher_ask` and `writer_ask` in one turn for L3).
The FastMCP servers and the agent loop are otherwise unchanged, so the
wall-clock numbers reflect the real executor at work.

## Prerequisites

Just `uv sync`. **No docker, no API key, no gateway** — the demos run
offline.

## Run

```bash
uv run python examples/16_parallel/01_parallel_tools.py
uv run python examples/16_parallel/02_parallel_subagents.py
```

## Expected output

### `01_parallel_tools.py`

`fetch_url` sleeps for `LATENCY_S = 1.5` seconds per call. With three
calls in a single turn:

```
Run 1 (sequential, the default):
  parallel_tools=False ->  4.5Xs (2 turn(s))

Run 2 (parallel):
  parallel_tools=True  ->  1.5Xs (2 turn(s))

Speed-up: ~3x (sequential / parallel)
```

The INFO log on stderr shows three `fetch_url START` markers followed
by three `fetch_url END` markers. Under sequential dispatch the START
and END lines interleave one pair at a time; under parallel dispatch
all three STARTs arrive back-to-back and all three ENDs land ~1.5 s
later.

### `02_parallel_subagents.py`

Each sub-agent's tool sleeps for 1 s. With the orchestrator delegating
to both sub-agents in the same turn:

```
Run 1 (sequential, the default):
  parallel_tools=False ->  2.0Xs (2 turn(s))

Run 2 (parallel):
  parallel_tools=True  ->  1.0Xs (2 turn(s))

Speed-up: ~2x (sequential / parallel)
```

The orchestrator's `parallel_tools=True` is what enables the overlap:
sub-agents are exposed as MCP tools, so multi-agent parallelism reduces
to the same intra-turn dispatch as L1.

## Adapting the demos to a real LLM

`MockGateway` is a duck-typed `AsyncTensorZeroGateway`. To swap a real
gateway in, replace `_build_gateway()` (or the three `_..._gateway()`
helpers in demo 02) with `AsyncTensorZeroGateway.build_http(...)`, add
a `tensorzero.toml` declaring the functions used by each
`AgentConfig.function_name`, and boot the gateway with
`docker compose`. Whether you then observe the speed-up depends on
your model's parallel-tool-call behavior — see
[`docs/concurrency.md`](../../docs/concurrency.md) for the full
discussion.

## File layout

```
examples/16_parallel/
├── README.md                       ← this file
├── 01_parallel_tools.py            ← L1 demo with side-by-side timing
├── 02_parallel_subagents.py        ← L3 demo with side-by-side timing
├── server.py                       ← FastMCP server: fetch_url + fetch_urls_batch
├── research_server.py              ← researcher sub-agent's MCP server
└── writing_server.py               ← writer sub-agent's MCP server
```
