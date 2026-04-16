#!/usr/bin/env bash
# Launch the HTTP agent on port 8080.
#
# Prerequisites:
#   - `docker compose up -d` from the repo root (gateway + clickhouse).
#   - `GROQ_API_KEY` in `.env` (or exported).
#
# Then exercise the endpoints with `./client.sh` or `curl` directly.
set -euo pipefail

cd "$(dirname "$0")/../.."
uv run uvicorn examples.11_http_agent.agent_app:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info
