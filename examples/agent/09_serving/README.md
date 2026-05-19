# 09 — Serving (FastAPI)

One-line wrap of an `Agent` in a FastAPI service. `uvicorn` serves the
four endpoints documented in `docs/serving.md`:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Run the agent against a user message; returns an `AgentResult` JSON |
| `POST` | `/feedback` | Passthrough to `AsyncTensorZeroGateway.feedback` for manual metric emission |
| `GET` | `/health` | Per-subsystem readiness (`gateway`, `tool_provider`) |
| `GET` | `/info` | Redacted snapshot of the agent's config |

## Running the example

```bash
docker compose --env-file .env -f examples/agent/09_serving/docker-compose.yml up -d
export GROQ_API_KEY=gsk_...                                  # or live in .env
pip install 't0-agents[serve]'                               # one-off
./examples/agent/09_serving/run.sh                           # uvicorn :8080
# In a second terminal:
./examples/agent/09_serving/client.sh                        # curl the four endpoints
```

The bundled `docker-compose.yml` + `config/tensorzero.toml` make this
folder self-contained — no shared gateway needed.

## Highlights

* **Per-session Client** — `client_factory=lambda: Client(mcp)` builds
  a fresh FastMCP client per HTTP request.  This is what keeps
  elicitation demultiplexed in deployments that enable it; the
  browser-based WebSocket path in `docs/streaming.md` reuses the same
  abstraction unchanged.
* **`InMemoryMemory` attached** — `client.sh`'s second `/chat` call
  sends the first call's `episode_id` back; the agent hydrates the
  prior exchange so the follow-up ("What code did I give you?")
  answers correctly.  Swap to `SqliteMemory("./demo.db")` from the
  `[memory-sqlite]` extra for process-restart-safe persistence.
* **`deployment_label="demo"`** — every inference carries
  `tags["t0_agents.deployment"]="demo"`.  Verify via
  `GET /info` or ClickHouse: `SELECT tags FROM InferenceModelInference
  LIMIT 1`.
* **Synchronous gateway construction** — the module uses
  `AsyncTensorZeroGateway.build_http(..., async_setup=False)` so the
  `app` object is built at import time.  That is what `uvicorn
  agent_app:app` expects; the gateway's async HTTP pool spins up on
  first use.

## Hosting multiple agents on one process

```python
from fastapi import FastAPI
from t0_agents.serve import mount_agent

parent = FastAPI()
mount_agent(parent, mobile_agent, "/mobile",
            client_factory=lambda: Client(mcp_mobile),
            deployment_label="mobile")
mount_agent(parent, analytics_agent, "/analytics",
            client_factory=lambda: Client(mcp_analytics),
            deployment_label="analytics")
# uvicorn parent:parent --port 8080
```

Each mounted sub-app runs its own `lifespan` and its own
`SessionFactory`; tags stay isolated per agent.

## `share_client=True` trade-off

For deployments that never use elicitation, you can skip the per-request
Client open/close overhead:

```python
# Note: keep the shared Client entered across the uvicorn lifetime yourself.
client = Client(mcp)
async with client:  # or manage elsewhere
    agent = Agent(config, t0=t0, tool_provider=client)
    app = create_app(agent, share_client=True)
    # uvicorn ...
```

Elicitation attempts on a shared client cannot be routed back to the
original caller — the framework's default forwarder declines them with
an explicit error.  The streaming extension (`streaming=True` on
`create_app`, covered in `docs/streaming.md`) lifts this limitation via
an SSE / WebSocket elicitation channel.

## What this example does *not* show

* **Elicitation over HTTP.**  Plain request/response has no return
  channel for human-in-the-loop prompts; the non-streaming surface
  declines elicitation with an explicit error so tools fail cleanly
  rather than hanging.  Add `streaming=True` on `create_app` for a live
  HITL channel — see `docs/streaming.md`.
* **Authentication / rate limiting.**  Compose your own FastAPI
  middleware; see `docs/serving.md` for examples.
* **Streaming deltas.**  Also `streaming=True` — see
  `docs/streaming.md`.
