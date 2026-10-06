# Contributing to t0-agents

Thanks for considering a contribution. This guide covers the local
development loop — environment, test modes, style gates, and the
checks you should run before opening a pull request.

## Prerequisites

- Python ≥ 3.11 (3.11, 3.12, 3.13 supported).
- [`uv`](https://docs.astral.sh/uv/) for dependency management —
  `uv sync` creates the virtualenv and installs dev deps automatically.
- Docker + `docker compose` if you plan to run the integration tests.
- A provider API key (Groq by default, `GROQ_API_KEY`) if you plan to
  run the embedded-gateway tests.

## Setup

```bash
git clone https://github.com/sproutfi/t0-agents.git
cd t0-agents
uv sync
```

`uv sync` creates `.venv/` and installs every dep declared in
`pyproject.toml` including the dev group (mypy, pytest, ruff).

## Test modes

The suite is split into three mutually exclusive markers so contributors
can pick the right cost tier for the change at hand.

### Unit (default — always runs)

Offline, deterministic, fast. Every test file with no `pytestmark`
falls here. Runs in about 3 seconds end-to-end.

```bash
uv run pytest -m "not integration and not embedded"
```

This is the gate every PR must pass. It must stay green on a vanilla
Python installation — no Docker, no API keys, no network.

### Integration — requires Docker

Tests that drive a real TensorZero gateway + real FastMCP server.
Marked `@pytest.mark.integration` (module-level `pytestmark =
pytest.mark.integration` is idiomatic). Run locally with:

```bash
# In a separate shell, boot the gateway + ClickHouse:
docker compose up -d

# Then:
uv run pytest -m integration

docker compose down   # (ClickHouse data persists in a named volume)
```

You need `GROQ_API_KEY` in `.env` for the default config. `docker
compose -f …` invocations should always pair with `--env-file .env`
so the credentials flow in.

### Embedded — real inference, no Docker

Tests that use `t0_agents.testing.EmbeddedGatewayHelper` for real
TensorZero inference without ClickHouse or Docker. Marked
`@pytest.mark.embedded` and skipped unless `RUN_EMBEDDED_TESTS=1`.

```bash
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest -m embedded
```

Use this mode when a test needs real model behaviour but doesn't need
to assert on observability rows.

### Which marker should my new test use?

- **None** (unit) — if the test doesn't need a real gateway.
  `MockGateway` + `MockToolProvider` cover the overwhelming majority
  of cases. See [`docs/testing.md`](docs/testing.md).
- **`embedded`** — if the test needs real inference but not a real
  observability backend. Cheapest real-model path.
- **`integration`** — if the test needs to assert on ClickHouse rows,
  on end-to-end HTTP behaviour, or on containerised topology.

When in doubt, start unit. Promote to a higher tier only when the
question really requires it.

## Style gates

```bash
uv run ruff check src/ tests/
uv run mypy src/
```

Both must be green before a PR lands. Ruff auto-fix is available via
`uv run ruff check --fix src/`. Mypy runs in strict mode — every new
symbol must be fully annotated.

Project conventions:

- Python ≥ 3.11 (uses PEP 604 `X | None`, `list[T]` generics, match
  statements).
- Async-first. Every public API is `async def`. Sync wrappers are
  allowed only when they clearly simplify the caller's life, and must
  be named distinctly (e.g. `Agent.run_sync` — not currently exposed).
- Line length 100. Ruff enforces this; the default `88` is not
  inherited.
- `mcp/` submodule stays standalone. No higher-level submodule may be
  imported from `mcp/`.
- Extras are opt-in — no new hard dependency in base `pyproject.toml`.
  New third-party deps go under an extra (`[memory-sqlite]`,
  `[serve]`, `[cli]`, etc.) and the submodule that needs them wraps
  the import in a `try`/`except ImportError` that raises a clear
  `T0AgentsError` naming the missing extra.

## Running a single test

```bash
# By file
uv run pytest tests/test_mcp/test_adapter.py

# By name (substring match on test id)
uv run pytest -k "test_convert_tools"

# One class
uv run pytest tests/test_core/test_agent.py::TestAgentConfig
```

## Writing a new test

Default path: put it under `tests/test_<submodule>/` with a name that
mirrors the module you're testing (`tests/test_core/test_agent.py`
tests `src/t0_agents/core/agent.py`). Use `MockGateway` +
`MockToolProvider` from `t0_agents.testing` instead of hand-rolling
fakes. If you need fixtures, add a `conftest.py` next to the test file.

If your test drives an `Agent` end-to-end, the simplest skeleton:

```python
from typing import cast

from tensorzero import AsyncTensorZeroGateway, Text, ToolCall
from t0_agents.core.agent import Agent
from t0_agents.core.config import AgentConfig
from t0_agents.core.types import ToolProvider
from t0_agents.testing import MockGateway, MockToolProvider, assert_tool_called


async def test_happy_path() -> None:
    gateway = MockGateway.script([
        [ToolCall(id="c1", raw_name="lookup", raw_arguments='{}',
                  name="lookup", arguments={})],
        [Text(text="done")],
    ])
    tools = MockToolProvider({"lookup": lambda args: "42"})

    agent = Agent(
        AgentConfig(name="t", function_name="agent", max_turns=3),
        t0=cast(AsyncTensorZeroGateway, gateway),
        tool_provider=cast(ToolProvider, tools),
    )
    result = await agent.run("hi")

    assert_tool_called("lookup", tool_provider=tools)
    assert "done" in result.response
```

See [`docs/testing.md`](docs/testing.md) for the full testing
vocabulary — scenarios, assertions, fixtures, and the embedded path.

## Before you open a PR

Run `bash ci/checks.sh`, the gate Cloud Build's `t0-agents-pr` runs on
every PR. If anything fails, don't submit.

If you touched `pyproject.toml`'s dependency ranges, also resolve
without the lockfile — every command above reads `uv.lock`, so a range
that no longer installs stays green here and breaks only for people
installing the package:

```bash
uv run --isolated --no-project --with ".[cli,glow]" \
  python -c "import t0_agents; print(t0_agents.__version__)"
```

Optional but appreciated:

```bash
# If your change might affect integration behaviour:
docker compose up -d
uv run pytest -m integration
docker compose down

# If your change might affect real-inference behaviour:
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest -m embedded
```

Leave `CHANGELOG.md` alone. Release notes are written as a versioned
block in their own release PR, which touches exactly five files and no
`src/` or `tests/` — an empty `tests/` diff is what makes a hung suite
attributable to the environment rather than to the release. Adding an
`[Unreleased]` section here breaks that; the repo has never carried
one.

The release PR bumps `VERSION`, `pyproject.toml` and
`src/t0_agents/__init__.py` together (`ci/checks.sh` fails if they
differ). Merging it tags `vX.Y.Z` from `VERSION` (Cloud Build
`t0-agents-release`); don't tag by hand.

## Documentation

- User-facing docs live under `docs/`. Kebab-case filenames,
  GitHub-flavoured Markdown. Add a new page for a new public feature;
  extend an existing page for an enhancement.

Thanks for helping t0-agents get better.
