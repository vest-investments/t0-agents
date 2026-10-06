#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --all-extras
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest -m "not integration and not embedded"

uv run python - <<'PY'
import tomllib

import t0_agents

versions = {
    "VERSION": open("VERSION").read().strip(),
    "pyproject.toml": tomllib.load(open("pyproject.toml", "rb"))["project"]["version"],
    "t0_agents.__version__": t0_agents.__version__,
}
assert len(set(versions.values())) == 1, versions
PY
