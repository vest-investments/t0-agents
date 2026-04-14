# 02 - Basic Agent

Minimal end-to-end example of `run_agent` -- the full agent loop that sends a user query through TensorZero, lets the LLM call MCP tools, and returns a final response.

> This directory is currently named `01_basic_agent` and will be renamed to `02_basic_agent` in a future cleanup.

## What it demonstrates

1. Connecting to TensorZero via `AsyncTensorZeroGateway`.
2. Providing an in-process FastMCP server (with a `get_weather` tool) as the tool backend.
3. Running the agent loop with `run_agent` and inspecting the result.

## Prerequisites

- Python 3.11+
- Dependencies installed: `uv sync`
- `GROQ_API_KEY` set in your environment or in a `.env` file at the project root.

## Start the gateway

```bash
docker compose -f examples/01_basic_agent/docker-compose.yml up -d
```

## Run

```bash
uv run python examples/01_basic_agent/main.py
```

## Stop the gateway

```bash
docker compose -f examples/01_basic_agent/docker-compose.yml down
```

## What to observe

- The LLM receives the `get_weather` tool schema and decides to call it.
- The agent loop executes the tool, feeds the result back, and the LLM produces a natural-language answer.
- `AgentResult` fields show the episode ID, inference IDs, turn count, and response source.
