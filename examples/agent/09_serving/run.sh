#!/usr/bin/env bash
# Launch the HTTP agent on port 8080.
#
# Prerequisites:
#   - docker compose --env-file ../../../.env up -d (boots gateway + clickhouse).
#   - GROQ_API_KEY in `.env` at repo root.
#
# Then exercise the endpoints with `./client.sh` or `curl` directly.
set -euo pipefail

cd "$(dirname "$0")"
PYTHONPATH="$PWD" exec uv run uvicorn agent_app:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info
