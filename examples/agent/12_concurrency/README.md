# 12 — Concurrency (canonical multi-agent patterns)

Three runnable examples that cover the multi-agent design space using
plain Python plus the v1.0 `Agent` surface — no framework abstraction
stands between the steps, so each example shows exactly where you can
drop ad-hoc code (a transform, a validation, a branch, a log).

| Pattern      | Topology                                | File                                                    |
|--------------|-----------------------------------------|---------------------------------------------------------|
| Sequential   | pipeline `A → B → C`                    | [`01_sequential.py`](01_sequential.py)                  |
| Parallel     | fan-out + plain-Python reducer          | [`02_parallel.py`](02_parallel.py)                      |
| Orchestrator | `AgentConfig(sub_agents=[…])` + LLM     | [`03_orchestrator.py`](03_orchestrator.py)              |

For the same three patterns expressed via the experimental
`@app.agent` decorator surface, see
[`examples/glow/04_sequential.py`](../../glow/04_sequential.py),
[`05_parallel.py`](../../glow/05_parallel.py), and
[`06_orchestrator.py`](../../glow/06_orchestrator.py).

## Prerequisites

The three examples share the gateway with `examples/glow/`.  Boot it
once from the repo root:

```bash
docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
```

You need `GROQ_API_KEY` in `.env`.  The shared `tensorzero.toml`
under `examples/glow/config/` declares the `agent` chat function
(used by every leaf) and the `orchestrator` chat function (used by
`03_orchestrator.py`).

## Run

```bash
uv run python examples/agent/12_concurrency/01_sequential.py
uv run python examples/agent/12_concurrency/02_parallel.py
uv run python examples/agent/12_concurrency/03_orchestrator.py
```

Each example prints labelled output as it works.  No assertions —
these are demonstrations, not tests.

## What each example shows

### `01_sequential.py`

Two toolless Agents (`summarizer`, `translator`) chained with plain
`await` plus an ad-hoc truncate step in the gap.  Nesting: call from
another `async def`, expose as MCP tool via `fastmcp.Tool.from_function`,
or embed inside an orchestrator's `sub_agents=[...]`.

### `02_parallel.py`

Three toolless Agents (`optimist`, `pessimist`, `pragmatist`) run
concurrently via `asyncio.gather(..., return_exceptions=True)`.  A
plain Python reducer joins the survivors with `"\n---\n"` and drops
any branch that raised.

### `03_orchestrator.py`

An orchestrator delegates to `researcher` and `writer` sub-agents via
`AgentConfig(sub_agents=[researcher, writer])`.  The `SubAgentMiddleware`
(in `t0_agents.delegation`) installs each sub-agent as a
`<name>_ask` MCP tool on the orchestrator's FastMCP tool provider.
Includes a docstring note on `max_turns` as the anti-recursion guardrail.

## When to use which

- **Sequential** — fixed pipeline where each step's input is the
  previous step's output.  No LLM in the routing decision.
- **Parallel** — independent calls on the same input, combined by a
  pure function.  Pick the reducer that fits your data; failures stay
  inline under `return_exceptions=True`.
- **Orchestrator** — the right delegate depends on the input, and the
  LLM should decide.  More expensive (the LLM picks; you pay one
  inference for the decision), but flexible.

The patterns nest.  An orchestrator's sub-agent can itself be a
Sequential pipeline; a Parallel reducer can return a value that feeds
the next step in a Sequential outer chain; a Sequential step can
contain a `asyncio.gather` fan-out internally.  None of these need a
special framework type — just plain async Python.
