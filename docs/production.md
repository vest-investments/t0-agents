# Production Guide

## Transport selection

FastMCP supports multiple transports. Choose based on your deployment model.

| Transport | Use case | Code | Elicitation | Notes |
|-----------|----------|------|-------------|-------|
| In-memory | Development, testing | `Client(mcp_instance)` | Yes | No network overhead. Server runs in-process. |
| HTTP Streamable | Production | `Client("https://host:8000/mcp")` | No | Stateless HTTP breaks bidirectional elicitation sessions. |
| SSE | Production, legacy | `Client("http://host:8000/sse")` | Yes | Stateful session supports elicitation. |
| stdio | Subprocess model | `Client(StdioTransport("python", "server.py"))` | Yes | Server runs as child process. |

## External MCP servers

The FastMCP client accepts URLs directly. Auth is handled at the transport level.

```python
from fastmcp import Client

client = Client("https://mcp.example.com/mcp")
# For auth: use StreamableHttpTransport with headers={"Authorization": "Bearer <token>"}
```

## TensorZero gateway deployment

**Development**: use Docker Compose. The project includes a compose file that boots the gateway and ClickHouse.

```bash
docker compose up -d  # gateway:2026.4.0 + clickhouse:lts
```

**Production**:

- Run the TensorZero gateway as a dedicated service (Docker container or binary).
- Point it at a persistent ClickHouse instance (not the ephemeral dev container).
- Pass provider API keys as environment variables, never in config files.
- TensorZero configuration (`tensorzero.toml`) should be mounted read-only.

```bash
docker run -d \
  -v ./config/tensorzero.toml:/app/config/tensorzero.toml:ro \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -p 3000:3000 \
  tensorzero/gateway:2026.4.0
```

## System prompts for tool-use reliability

System prompts significantly affect tool-use behavior. LLMs may default to asking clarification questions instead of calling available tools. Configure the system prompt in TensorZero via the function variant's template.

Recommended guidelines for the system prompt:

- Instruct the model to prefer tool calls over asking clarifying questions
- Direct the model to keep reasoning brief and act on available information
- Tell the model to give up on a tool after repeated failures, not retry indefinitely
- List the general categories of tools available so the model knows what to expect

Example system prompt (`system.minijinja`):

```
You are a helpful assistant with access to tools.

When a user asks a question that can be answered using the available tools,
call the appropriate tool immediately. Do not ask for clarification unless
the request is genuinely ambiguous.

Keep your reasoning brief. If a tool call fails, try a different approach
or explain what went wrong. Do not retry the same failing call more than twice.
```

See `examples/agent/02_agent_class/` for a complete example with Docker Compose and TensorZero configuration.

## Tuning max_turns

The `max_turns` parameter (default 20) caps the number of inference rounds before `MaxTurnsExceededError` is raised. Tune it per task:

| Task type | Suggested max_turns | Rationale |
|-----------|-------------------|-----------|
| Simple lookup (1 tool call) | 3-5 | One tool call + answer. Extra budget for retries. |
| Multi-step workflow | 10-15 | Several tool calls in sequence. |
| Complex research | 15-20 | Many tools, branching paths, error recovery. |

Setting `max_turns` too high risks the LLM entering retry loops on failing tools. Setting it too low causes premature `MaxTurnsExceededError` on legitimate multi-step tasks. Monitor the `turns` field in `AgentResult` to calibrate.

```python
result = await run_agent(
    t0, client, query, "agent",
    max_turns=5,  # Tight budget for simple queries
)
```

## Parallel vs sequential tool execution

`run_agent` and `AgentConfig` accept `parallel_tools: bool` (default `False`). The flag selects between two executors in `t0_agents.mcp.loop`:

**Sequential** (default): `execute_tool_calls` awaits each `ToolCall` block in turn. Safer for tools that share resources or have side effects.

**Parallel** (`parallel_tools=True`): `execute_tool_calls_parallel` dispatches concurrently via `asyncio.gather`. Faster for I/O-bound tools. Result order is preserved regardless of completion order.

```python
result = await run_agent(
    t0, client, query, "agent",
    parallel_tools=True,  # Concurrent tool execution
)
```

### Checklist before flipping `parallel_tools=True`

Confirm each of the following for the agents that will run with the flag on. If any item fails, leave the flag at its default.

- [ ] **Tool idempotency.** Every tool the agent might call in the same turn is either read-only or safe to run alongside its siblings. A retry of the same call should not corrupt state.
- [ ] **No shared mutable state across same-turn calls.** Tools do not write to the same record, file, or in-memory cache that another tool in the same turn reads.
- [ ] **Guardrails are race-safe.** Built-in `max_cost`, `max_tool_calls`, and `block_tools` are. **`require_approval` is not** (see [Guardrails / `parallel_tools=True`](guardrails.md#parallel_toolstrue)) — it fans out N approval prompts to one human. Custom guardrails that mutate shared state across an `await` boundary need an `asyncio.Lock`.
- [ ] **Provider rate limits absorb the burst.** Parallel dispatch collapses N latency budgets into one but multiplies your peak QPS to the tool backend and the LLM provider. Confirm your quotas and circuit breakers can take it.
- [ ] **Tracing tolerates concurrent calls.** TensorZero tags propagate per inference, not per parallel tool call. ClickHouse traces will show overlapping `call_tool` spans under the same `inference_id`; downstream dashboards should not assume serial ordering inside one turn.
- [ ] **The model actually emits multi-tool turns.** `parallel_tools=True` only helps when the LLM emits several `ToolCall` blocks in the same assistant turn. Check the model card for "parallel function calling" / "parallel tool use" support, confirm your provider is on TensorZero's `parallel_tool_calls` allowlist (Groq is currently not; OpenAI and Fireworks AI are), and run a one-inference probe against `dryrun=True` that counts the `ToolCall` blocks in `response.content`. If you get one tool call per turn, the L1 flag is a no-op — use the L2 batch-tool pattern from [Concurrency](concurrency.md#l2-inter-turn-dependencies-and-the-batch-tool-pattern) instead.

The full mental model — including what parallelism cannot fix (inter-turn dependencies) and how multi-agent dispatch composes with the flag — lives in [Concurrency](concurrency.md).

## Elicitation timeout

When tools use [elicitation](elicitation.md) (human-in-the-loop prompts),
the default `tool_timeout=30.0` may be too short. The timeout wraps the
entire `call_tool()` duration, including human response time.

| Scenario | Recommended `tool_timeout` |
|----------|--------------------------|
| No elicitation-capable tools | `30.0` (default) |
| Mixed tools (some with elicitation) | `300.0` (5 minutes) |
| All tools may elicit | `None` (disable timeout) |

```python
# Generous timeout for mixed workloads
result = await run_agent(
    t0, client, query, "agent",
    tool_timeout=300.0,
)
```

See [Elicitation](elicitation.md) for the full guide.

Note: the `parallel_tool_calls` inference parameter (which tells the LLM to emit multiple tool calls in one turn) is a separate provider-level concern. Only some providers support it (OpenAI, Fireworks AI). Even without provider support for parallel emission, the LLM may still emit multiple tool calls in one turn, which `parallel_tools=True` will execute concurrently.

## Integration with other libraries

### Pydantic

FastMCP tools accept Pydantic models as parameters. The model's JSON Schema is generated automatically and passed to TensorZero as `additional_tools`. Keep schemas to 2 or fewer nesting levels for broad provider compatibility -- deeply nested schemas may cause issues with some LLM providers.

```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    max_results: int = 10

@mcp.tool
def search(params: SearchParams) -> list[dict]:
    """Search the database."""
    ...
```

### asyncio

The library is async-first. Use `asyncio.run()` at the entry point. For existing event loops (web frameworks), call `run_agent` directly from an async context.

### Web frameworks

FastAPI/Starlette can wrap `run_agent` in endpoint handlers. Initialize the gateway and MCP client at startup, share across requests:

```python
@app.post("/agent")
async def agent_endpoint(query: str) -> dict:
    result = await run_agent(app.state.t0, app.state.client, query, "agent")
    return {"response": result.response, "turns": result.turns}
```

## Rate limits and quotas

t0-agents intentionally does **not** auto-retry on `HTTP 429` from the
TensorZero gateway.  Rate-limit policy is a gateway concern
(`tensorzero.toml` `rate_limiting.rules.*`, Valkey/Postgres-backed,
scoped by api_key / tags); the agent loop propagates the
`TensorZeroError(status_code=429)` to the caller, who decides whether
to back off, fail the request, or queue.

For per-invocation budgets (cost cap, tool-call cap), use the t0-agents
`max_cost` / `max_tool_calls` guardrails — they raise
`GuardrailBlockedError` mid-run.  See
[Guardrails vs. TensorZero rate limits](guardrails.md#guardrails-vs-tensorzero-rate-limits)
for the layered model.

## Integration-test sensitivities by model

The integration suite (`pytest -m integration`) is smoke-validated against
Groq's `openai/gpt-oss-20b`.  Two model-specific failure modes are
handled with `pytest.skip` (not `assert`) so the gate is not held hostage
by the LLM picking a flake path on any given run.  Both skips include
the recognised Groq error fragment in the message so they are easy to
spot in CI output.

- **gpt-oss-20b sometimes short-circuits trivial arithmetic** and
  answers `"5"` as plain text when asked to `add 2 + 3` — no tool call
  is emitted, so any test that asserts a guardrail/audit-log entry for
  the tool invocation cannot run.  Pinned by
  `tests/test_guardrails/test_integration.py::test_require_approval_auto_decline_surfaces_to_llm`:
  the test uses non-trivial operands (`1247 + 8932`) and skips if the
  rejection branch still wasn't exercised.  Strict mechanism pinning
  lives in the offline `tests/test_guardrails/test_require_approval.py`.
- **gpt-oss-20b sometimes emits malformed retry arguments** after a
  tool failure (timeout, MCP error).  Groq's *own* server-side validator
  rejects these with `400 Tool call validation failed`, which the
  TensorZero gateway forwards as `502 TensorZeroError`.  Pinned by
  `tests/test_mcp/test_integration.py::test_tool_timeout_does_not_crash_run_agent`
  and four sibling tests
  (`test_run_agent_survives_failing_tool`, `test_run_agent_allows_llm_retry`,
  `test_max_turns_raises_against_live_gateway`,
  `test_none_description_tool_is_usable_end_to_end`): all catch
  `TensorZeroError`, pattern-match the validation/`tool_use_failed`
  fragment, and `pytest.skip`.  Any other `TensorZeroError` is
  re-raised so a genuine regression is still loud.

Both classes of flake disappear when the integration suite runs
against a frontier model (Claude Sonnet 4.6, GPT-4o, Gemini 2.5 Pro).
Swap the model in `examples/agent/01_first_agent/config/tensorzero.toml`
(or whichever compose your run targets) if you want the
zero-skip variant for a release-gate signal.

## Monitoring checklist

- Enable Python logging at INFO level in production (`t0_agents.mcp.loop` logger)
- Track `MaxTurnsExceededError` frequency -- high rates indicate prompt or tool issues
- Send episode-level boolean feedback (`task_success`) to measure task completion rates
- Monitor `AgentResult.turns` distribution to detect regressions in agent efficiency
- Set up ClickHouse dashboards for inference latency and tool call frequency
- Alert on elevated ERROR-level log rates (tool failures, invalid arguments)
- Track `response_source == "thought"` occurrences -- indicates reasoning-model fallback behavior
