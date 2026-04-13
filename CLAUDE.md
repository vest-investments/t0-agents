# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

t0-agents is a Python library that bridges TensorZero (LLM inference gateway) and FastMCP (MCP server framework), enabling LLM agents to discover and execute tools exposed by MCP servers through TensorZero's inference gateway. Alpha status — not production-ready.

## Commands

```bash
uv sync                            # Install dependencies (creates venv automatically)
uv run pytest                      # Run all tests (integration tests fail unless gateway is up)
uv run pytest -m "not integration" # Offline tests only — safe default
uv run pytest -m integration       # Requires `docker compose up` first
uv run pytest tests/test_mcp/test_adapter.py  # Run a single test file
uv run pytest -k "test_name"       # Run a single test by name
uv run ruff check src/             # Lint
uv run ruff check --fix src/       # Lint with auto-fix
uv run mypy src/                   # Type check (strict mode)

# Local TensorZero gateway (needed for integration tests and examples)
docker compose up -d               # Boots gateway:2026.4.0 + clickhouse:lts; needs GROQ_API_KEY in .env
docker compose down                # Stop containers (ClickHouse data persists in named volume)
```

## Architecture

**Agent loop flow:** User query → TensorZero inference (with tool list) → LLM returns text + tool calls → execute tools via FastMCP client → feed results back → loop until done or max turns → return `AgentResult`.

Two key modules under `src/t0_agents/`:

- **`mcp/adapter.py`** — Converts FastMCP tools to TensorZero's JSON Schema format, serializes tool results back. Three functions: `convert_tools`, `serialize_tool_result`, `build_tool_result`.
- **`mcp/loop.py`** — Orchestrates the agent loop. `run_agent` is the main entry point (re-exported from package root). Also contains `execute_tool_calls` and `has_tool_calls`.
- **`core/types.py`** — `AgentResult` dataclass (response, episode_id, inference_ids, turns).
- **`core/exceptions.py`** — `MaxTurnsExceededError`.

Public API: `from t0_agents import run_agent`

## Code Style & Conventions

- Python >= 3.11, targeting 3.11/3.12/3.13
- mypy strict mode — all code must be fully typed
- Ruff rules: E, F, I, N, UP, RUF. Line length 100.
- Async-first: agent loop and tool execution are async. Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` decorator needed).
- `src/` layout with hatchling build backend

## Test Infrastructure

- `tests/server.py` — Shared FastMCP test server with `add(a, b)` and `greet(name)` tools
- `tests/conftest.py` — `mcp_client` fixture (typed `Client[Any]`) connected to the shared test server
- `tests/test_infrastructure.py` — Phase 0 validations (P0.2/3/4/5). The P0.2 inference test is marked `@pytest.mark.integration`
- Integration tests marked with `@pytest.mark.integration` require `docker compose up` to be running

## Design Documentation

- `docs/implementation-plan/` — Versioned design docs: integration contract (`01-`), validation checklist (`02-`), meta-plan (`03-`), implementation reference patterns (`04-`). Consult these when implementing adapter or loop functions.
- `docs/phase-0/` — Per-phase guides and the implementation log (`phase-0-implementation-log.md`) that records decisions and divergences from the plan.
