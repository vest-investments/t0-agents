#!/usr/bin/env bash
# Launch the streaming demo on localhost:8080.
#
# Prereqs:
#   1. docker compose --env-file ../../../.env up -d   (gateway + clickhouse)
#   2. GROQ_API_KEY in .env at repo root
#   3. uv sync                                         (picks up sse-starlette)
#
# Open http://localhost:8080/static/index.html in a browser.

set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH="$PWD" exec uv run uvicorn agent_app:app --port 8080 --reload
