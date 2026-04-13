# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

t0-agents is a Python library that bridges TensorZero (LLM inference gateway) and FastMCP (MCP server framework), enabling LLM agents to discover and execute tools exposed by MCP servers through TensorZero's inference gateway. Alpha status — not production-ready.

## Commands

```bash
uv sync                            # Install dependencies (creates venv automatically)
uv run pytest                      # Run all tests
uv run pytest -m "not integration" # Skip integration tests (require running T0 Gateway)
uv run pytest tests/test_mcp/test_adapter.py  # Run a single test file
uv run pytest -k "test_name"       # Run a single test by name
uv run ruff check src/             # Lint
uv run ruff check --fix src/       # Lint with auto-fix
uv run mypy src/                   # Type check (strict mode)
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
- `tests/conftest.py` — `mcp_client` fixture providing a FastMCP Client connected to the test server
- Integration tests marked with `@pytest.mark.integration` require a running T0 Gateway

## Design Documentation

Numbered markdown files in the repo root (`01-` through `04-`) contain the integration contract, validation checklist, implementation plan, and code reference patterns. Consult these when implementing adapter or loop functions.
