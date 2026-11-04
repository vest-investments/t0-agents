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

wheel_dir=$(mktemp -d)
uv build --wheel --quiet --out-dir "$wheel_dir"
uv run python - "$wheel_dir"/*.whl <<'PY'
import re
import sys
import zipfile
from pathlib import Path

wheel = zipfile.ZipFile(sys.argv[1])
metadata = wheel.read(next(n for n in wheel.namelist() if n.endswith("METADATA"))).decode()
repo = re.compile(
    r"https://(?:github\.com/vest-investments/t0-agents/blob|"
    r"raw\.githubusercontent\.com/vest-investments/t0-agents)/main/([^#]*)"
)
broken = []
for link, src in re.findall(r'\]\(([^)]+)\)|src="([^"]+)"', metadata.partition("\n\n")[2]):
    target = link or src
    into_repo = repo.match(target)
    if into_repo:
        if not Path(into_repo[1]).exists():
            broken.append(target)
    elif not re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|//|#", target):
        broken.append(target)
assert not broken, f"PyPI description links that lead nowhere: {broken}"
PY
