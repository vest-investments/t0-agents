#!/usr/bin/env bash
# Exercise the HTTP agent demo from the command line.
#
# Assumes `./run.sh` is running on :8080.  Pipes through `jq` for
# readability — skip with `--raw` to drop the pipe.
set -euo pipefail

BASE="${BASE:-http://localhost:8080}"
JQ="${JQ:-jq .}"
RAW=0
for arg in "$@"; do
    if [[ "$arg" == "--raw" ]]; then
        RAW=1
        JQ="cat"
    fi
done

prt () { echo -e "\n\033[1m$1\033[0m"; }

prt "GET /health"
curl -sS "$BASE/health" | $JQ

prt "GET /info"
curl -sS "$BASE/info" | $JQ

prt "POST /chat (maths tool)"
curl -sS -X POST "$BASE/chat" \
    -H 'content-type: application/json' \
    -d '{"message": "Use the tools to compute 7 + 5."}' | $JQ

prt "POST /chat (same episode — continuation)"
EP=$(curl -sS -X POST "$BASE/chat" \
    -H 'content-type: application/json' \
    -d '{"message": "Remember the code 42."}' | jq -r .episode_id)
echo "episode_id=$EP"
curl -sS -X POST "$BASE/chat" \
    -H 'content-type: application/json' \
    -d "{\"message\": \"What code did I give you?\", \"episode_id\": \"$EP\"}" | $JQ

prt "POST /feedback (will 404 unless 'agent_response_quality' is declared in tensorzero.toml)"
curl -sS -X POST "$BASE/feedback" \
    -H 'content-type: application/json' \
    -d '{"metric_name": "agent_response_quality", "value": true, "episode_id": "'"$EP"'"}' | $JQ
