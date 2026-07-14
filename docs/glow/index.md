# `t0_agents.glow` — the eight-line path

> **Experimental.**  Public API on the
> [`experimental/glow`](https://github.com/sproutfi/t0-agents/tree/experimental/glow)
> branch is unstable until merged to `main`.

A Pythonic skin over the v1.0 primitives — `Agent`, `Run[T]`,
`OutputMode[T]`.  Where the base API takes ~25 lines to wire an
agent, glow takes eight — gateway included:

```python
from t0_agents.glow import App, llm

app = App()  # resolves TENSORZERO_URL / TENSORZERO_CONFIG at first use

@app.agent("tutor")
async def tutor(query: str) -> str:
    """An English tutor that explains grammar like you're 12."""
    return llm()

async with app:
    print(await tutor("How do I say hello in formal Japanese?"))
```

The docstring is the system prompt.  The `-> str` return annotation
selects [`ChatOutput`](../output-modes.md).  `await tutor(...)`
returns the unwrapped string directly.

Install with the extra:

```bash
pip install 't0-agents[glow]'
```

## Reading order

Each chapter is short enough to read in one pass; the runnable
companions live under [`examples/glow/`](../../examples/glow/).

1. **[Getting Started](getting-started.md)** — toolless chat,
   structured outputs by return type, per-call overrides
   (`.json()` / `.text()` / `.as_()`), and the three multi-agent
   patterns expressed in plain Python.
2. **[Streaming and events](streaming-and-events.md)** —
   `FluentRun.events(*selectors)`, the seven typed selectors
   (`Thought`, `ToolCall`, `Text`, `ToolResult`, `TurnEnd`, `Done`,
   `Error`), `.where(pred)` / `.map(fn)` composition, and the
   `Selector(TextDeltaEvent)` recipe for token-by-token streaming.
3. **[Depends and tools](depends-and-tools.md)** — value-style
   injection inside `@app.tool`, sub-agent injection inside
   `@app.agent`, and `app.dependency_overrides` for tests.
4. **[Typed MCP and guardrails](typed-mcp-and-guardrails.md)** —
   class-based `MCPServer` declarations, the `Annotated[...]`
   per-parameter override, and the `Annotated[T, guardrail(...)]`
   metadata pattern.
5. **[Escape hatch](escape-hatch.md)** — when glow stands in your
   way, `wrapper._agent` is the v1.0 `Agent` the decorator built.
   Includes `app.lifespan()` / `build()` / `shutdown()` and the
   drop-down recipes for middleware, custom `OutputMode`, hand-rolled
   event handling, and the `cancel()` / `fork()` / `replay()` surface.
6. **[Known limitations](limitations.md)** — what's deferred to
   v1.1: streaming JSON, per-agent MCP subsets, AsyncIterator tools,
   sync API.

## Drop down to Agent anytime

Glow compiles down to a real [`Agent`](../agent.md).  Every
`@app.agent` wrapper exposes `_agent` — the v1.0 surface — for the
moments the decorator is in your way.  See
[`escape-hatch.md`](escape-hatch.md) for the recipes.

## Where this fits

This is the glow ladder.  For the canonical v1.0 narratives — `Agent`,
`Run[T]`, `Middleware`, `OutputMode[T]`, memory, guardrails, streaming,
multi-agent, serving, the CLI, testing — see
[`../index.md`](../index.md).
