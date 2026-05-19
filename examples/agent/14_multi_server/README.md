# Example 04: Multi-Server Composition

Demonstrates how to combine tools from multiple MCP servers into a single agent using FastMCP's `mount()` method.

## Why server composition matters

Real-world agents often need tools from different domains -- math operations, text processing, database queries, API calls. Rather than cramming everything into one monolithic server, you can organize tools into focused servers and compose them at runtime.

## How mount() works

FastMCP's `mount()` method attaches one server's tools onto another with a namespace prefix:

```python
from fastmcp import FastMCP

composed = FastMCP("ComposedServer")
composed.mount(math_mcp, namespace="math")
composed.mount(text_mcp, namespace="text")
```

After mounting, tools are available under their namespaced names:

| Original server | Original tool | Composed name     |
|-----------------|---------------|-------------------|
| MathServer      | `add`         | `math_add`        |
| MathServer      | `multiply`    | `math_multiply`   |
| TextServer      | `word_count`  | `text_word_count` |
| TextServer      | `reverse`     | `text_reverse`    |

## Namespace collision resolution

If two servers expose a tool with the same name, the namespace prefix keeps them distinct. For example, both servers could have a `describe` tool, and they would appear as `math_describe` and `text_describe` with no conflict.

## No changes needed in run_agent

The `run_agent` function treats tool names as opaque strings. It receives the namespaced tool list from FastMCP's `Client`, passes those names to TensorZero, and routes tool calls back through the same client. The composition is entirely transparent to the agent loop.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A `GROQ_API_KEY` environment variable (get one at https://console.groq.com)

## How to run

```bash
# 1. Export your API key
export GROQ_API_KEY="gsk_..."

# 2. Start the gateway and ClickHouse
docker compose -f examples/14_multi_server/docker-compose.yml up -d

# 3. Run the example
uv run python examples/14_multi_server/main.py

# 4. Tear down when done
docker compose -f examples/14_multi_server/docker-compose.yml down
```

## What to observe

- On startup, the script prints the four namespaced tools available to the agent.
- The user message asks two questions that require different tool domains: word counting (text) and multiplication (math).
- The agent calls `text_word_count` and `math_multiply` (possibly in separate turns), then combines the results into a final answer.
- The output shows the response and the number of turns the agent took.
