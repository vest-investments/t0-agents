# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

t0-agents is a Python agent framework over the TensorZero gateway (inference, episodes, observability) and FastMCP (tools). Beta. Two surfaces, one engine: `t0_agents.glow` (`App`, `@app.agent`) is the primary API, and every glow agent compiles to a `t0_agents.Agent`, the explicit surface (`Agent`, `Run[T]`, `Middleware`, `OutputMode[T]`).

## Commands

```bash
uv sync --all-extras                                  # dev group + every extra; mypy src/ needs the extras
bash ci/checks.sh                                     # the PR gate (Cloud Build t0-agents-pr)
uv run pytest -m "not integration and not embedded"   # offline suite
uv run pytest tests/test_glow/test_app.py -k "name"   # one file / one test
uv run ruff check src/ tests/
uv run mypy src/                                      # strict
docker compose up -d                                  # gateway + ClickHouse for -m integration; GROQ_API_KEY in .env
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest -m embedded
```

## Layout

`src/t0_agents/`, one package per row of the module table in `docs/index.md`:

- `glow/`: `App`, `@app.agent`, `@app.tool`, `Depends`, `MCPServer`, conversations, `FluentRun`.
- `core/`: `Agent`, `AgentConfig`, `Run[T]`, the middleware chain, output modes, exceptions, the elicitation forwarder.
- `mcp/`: the tool adapter, the loop (`run_agent`) and `MCPRuntime` (one warm client per server). It imports only `core` and itself.
- `stream/`: the `AgentEvent` union, the streaming loop and assembler, status hints.
- `memory/`: `MemoryProvider` with in-memory, SQLite and TensorZero backends. `usage/`: `UsageLedger` with in-memory and SQLite backends.
- `guardrails/`: policy guardrails. `tooling/`: `FilteredToolProvider`, the `tools=` allow-list.
- `delegation/`: sub-agents as `{name}_ask` tools. `serve/`: the `create_app` FastAPI layer. `wire/`: the versioned frame contract for stream consumers.
- `testing/`: `MockGateway`, `MockToolProvider`, assertions and pytest fixtures, shipped in the base install.

Tests mirror the packages under `tests/test_<package>/`; shared fakes are `tests/_fake_t0.py`, `tests/_fake_mcp.py` and `tests/server.py`. Unmarked tests are offline and are the gate; `integration` needs `docker compose up`, `embedded` needs `RUN_EMBEDDED_TESTS=1`. `docs/` is a progressive tour (`docs/index.md`) and changes with the code.

## Code Style & Conventions

- Python >= 3.11 (3.11–3.13), mypy strict, ruff `E, F, I, N, UP, RUF`, line length 100.
- Async-first; tests run with `asyncio_mode = "auto"`.
- No new hard dependency: a third-party package goes behind an extra, imported in a `try`/`except ImportError` that names the extra.
- Feature PRs leave `CHANGELOG.md` alone. A release is its own PR touching `CHANGELOG.md`, `VERSION`, `pyproject.toml`, `src/t0_agents/__init__.py` and `uv.lock`; merging it tags `vX.Y.Z` (CONTRIBUTING.md § Before you open a PR).

## Commit & PR Conventions

The commit history is public and part of the deliverable — keep it readable.

- **Never open a PR with a single giant commit.** Split the work into commits that tell the story of the change, each a logical, self-contained step (e.g. new module → call-site migration → tests → docs). A non-trivial PR typically has 5–10 commits.
- **One concern per commit.** A refactor, a feature and unrelated test fixes go in separate commits.
- **Match the existing style.** Conventional-commit prefixes (`feat(scope): …`, `fix(scope): …`, `test(scope): …`, `docs(scope): …`, `refactor(scope): …`), imperative mood, lowercase summary, ≤ 72 char first line, body wrapped at ~72 for the "why".
- Co-authored-by trailers are fine and expected when Claude assisted.
