# 07 — Streaming (SSE + WebSocket + elicitation)

Live demo of streaming agents over HTTP. Shows:

* **Token-level streaming** — assistant text renders delta-by-delta as the
  LLM produces it.
* **Tool-call spinners** — each MCP tool call gets a `Calling add…` bubble
  that swaps to the result when the tool returns.
* **Thought bubbles** — when the model is a reasoning model
  (DeepSeek R1, o1, etc.) intermediate thinking streams into a muted
  bubble.
* **Elicitation round-trip** — `delete_task` / `create_task` are gated by
  `ctx.elicit()` and surface as a browser modal the user can accept or
  decline.

## Run

```bash
docker compose --env-file .env -f examples/agent/07_streaming/docker-compose.yml up -d
export GROQ_API_KEY=gsk_...    # or whichever provider your function uses
uv sync                        # picks up sse-starlette under [serve]
examples/agent/07_streaming/run.sh
# → open http://localhost:8080/static/index.html
```

The bundled `docker-compose.yml` + `config/tensorzero.toml` make this
folder self-contained — no shared gateway needed.

## What is wired

```
┌────────────────────────────────────────────────────────────────┐
│  Browser  (index.html + app.js)                                │
│     │                                                          │
│     │  WS /chat/ws                                             │
│     ▼                                                          │
│  FastAPI app  (create_app(streaming=True, ...))                │
│     │                                                          │
│     │  per-connection Client(mcp, elicitation_handler=h)       │
│     │  per-connection ElicitationForwarder (on_request=ws)     │
│     ▼                                                          │
│  Run.events(message, episode_id=...)                         │
│     │                                                          │
│     │  stream=True                                             │
│     ▼                                                          │
│  AsyncTensorZeroGateway.inference → ChatChunks → AgentEvents   │
└────────────────────────────────────────────────────────────────┘
```

Every AgentEvent type maps to a small rendering routine in `static/app.js`
— the file fits on one screen so the wire protocol is visible end-to-end.

## Try it

1. `What is 2 + 3?` → text streams token-by-token, `add` spinner flashes.
2. `List my tasks.` → text-only turn.
3. `Delete task T-103.` → approval modal appears; click **Accept** or
   **Decline**. The tool completes and the agent acknowledges.
4. `Create a new task.` → structured-input modal with `title`, `priority`,
   `assignee` fields. Accept with values filled in.

### Caveat: LLM may still ask in chat first

Smaller / older models (`gpt-oss-20b`, Llama-3-8B, etc.) sometimes read
a tool description like "requires human approval" and interpret it
literally — they respond in chat with "Are you sure?" instead of calling
the tool, then only invoke it after the user replies "yes". The
elicitation modal still shows up, just one turn late.

Mitigations:

- **Prompt directly in the imperative**: `Delete task T-103` rather than
  `Can you delete task T-103?`
- **Sharpen the tool docstring**: tell the model to call the tool
  directly and not to ask in chat first. This example's tools already
  include that explicit instruction.
- **Force the tool with `tool_choice`**: if your app can detect the
  intent, `AgentConfig.tool_choice={"specific": "delete_task"}` removes
  the question entirely.
- **Use a larger model**: GPT-4-class / Sonnet / Opus consistently skip
  the intermediate chat turn.

This is a model-behaviour caveat, not a framework bug.

## SSE smoke-test

The SSE endpoint is one-way and does **not** wire elicitation (no return
channel), so tools that call `ctx.elicit()` will surface the forwarder's
default decline as a tool error. For read-only streams:

```bash
curl --no-buffer \
  'http://localhost:8080/chat/stream?message=Say+hello&episode_id='
```

## Highlights

* `agent_app.py` shows the two-factory recipe (``client_factory`` for the
  non-streaming / SSE path; ``elicitation_client_factory`` for WS).  For
  the advanced `SessionProvider` recipe (auth, multi-tenant), see
  `docs/streaming.md`.
* `InMemoryMemory()` is attached so multi-turn conversations remember the
  prior context (swap for `SqliteMemory` behind the `[memory-sqlite]`
  extra in production — see `docs/memory.md`).
* `deployment_label="streaming-demo"` tags every inference with
  `t0_agents.deployment=streaming-demo` for ClickHouse observability.
