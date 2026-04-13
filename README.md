# t0-agents

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).

> **Status**: Work in progress — not ready for production use.

## What is this?

`t0-agents` is a Python library that bridges [TensorZero](https://www.tensorzero.com/) and [FastMCP](https://github.com/jlowin/fastmcp), enabling LLM agents to discover and execute tools exposed by MCP servers through TensorZero's inference gateway.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)

## Getting started

```bash
# Clone the repo
git clone https://github.com/sproutfi/t0-agents.git
cd t0-agents

# Install dependencies (uv creates the venv automatically)
uv sync

# Run checks
uv run pytest
uv run ruff check src/
uv run mypy src/
```

## Project structure

```
src/t0_agents/
├── core/           # Shared types and exceptions
└── mcp/            # TensorZero <-> FastMCP bridge
tests/
examples/
```
