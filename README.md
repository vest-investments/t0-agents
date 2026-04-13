# t0-agents

The Pythonic agentic framework for [TensorZero](https://www.tensorzero.com/).

> **Status**: Work in progress — not ready for production use.

## What is this?

`t0-agents` is a Python library that bridges [TensorZero](https://www.tensorzero.com/) and [FastMCP](https://github.com/jlowin/fastmcp), enabling LLM agents to discover and execute tools exposed by MCP servers through TensorZero's inference gateway.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker + Docker Compose (only for the TensorZero gateway used by integration tests / examples)

## Getting started

```bash
# Clone the repo
git clone https://github.com/sproutfi/t0-agents.git
cd t0-agents

# Install dependencies (uv creates the venv automatically)
uv sync

# Run offline checks (no gateway needed)
uv run pytest -m "not integration"
uv run ruff check src/
uv run mypy src/
```

### Running the integration tests / examples

The integration suite and examples talk to a real TensorZero gateway backed by ClickHouse.

```bash
# 1. Provide a Groq API key (or change the provider in config/tensorzero.toml)
cp .env.example .env
$EDITOR .env  # set GROQ_API_KEY=gsk_...

# 2. Boot the gateway + ClickHouse
docker compose up -d

# 3. Run the integration-marked tests
uv run pytest -m integration

# 4. When you're done
docker compose down
```

## Project structure

```
src/t0_agents/        # Library: TensorZero <-> FastMCP bridge
├── core/             # Shared types and exceptions
└── mcp/              # Adapter + agent loop
tests/                # Offline + integration tests
examples/             # End-to-end usage samples
config/               # tensorzero.toml mounted into the gateway container
docs/                 # Implementation plan + per-phase guides and logs
docker-compose.yml    # Local gateway + ClickHouse
```
