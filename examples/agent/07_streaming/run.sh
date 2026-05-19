#!/usr/bin/env bash
# Launch the streaming demo on localhost:8080.
#
# Prereqs:
#   1. docker compose up -d   (from repo root — gateway + clickhouse)
#   2. GROQ_API_KEY in .env   (or whichever provider your agent function uses)
#   3. uv sync                (picks up the sse-starlette dep)
#
# Open http://localhost:8080/static/index.html in a browser.

set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run uvicorn examples.07_streaming.agent_app:app --port 8080 --reload
