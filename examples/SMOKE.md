# Release-gate smoke verification

Manual checklist run by a single operator before tagging each release.
Covers all 24 example scripts in `examples/` plus four manual UI
checks.  **Not CI** — these runs consume real Groq tokens, two
batches drive scripts whose default model is occasionally
non-deterministic, and two scripts shell out to interactive
prompts.  Wall-clock budget on first try: ~25 minutes.

## When to run

- Before tagging any minor (`1.x.0`) or patch (`1.0.x`) release on
  `experimental/glow` or `main`.
- After landing any PR that touches `examples/`, `src/t0_agents/`,
  or the gateway image pin in `examples/*/docker-compose.yml`.

## Setup

The release operator does this once per smoke session.

```bash
# 1. From the repo root, create both .env files:
cp examples/agent/.env.example examples/agent/.env
cp examples/glow/.env.example  examples/glow/.env
# 2. Fill in GROQ_API_KEY in BOTH (https://console.groq.com/keys).
#    T0_GATEWAY_VERSION is already set to a tested default; only
#    bump it if validating a gateway upgrade.

# 3. Install the project with the extras the smoke uses:
uv sync --extra cli --extra memory-sqlite --extra pydantic

# 4. Confirm Docker is reachable (the smoke uses docker compose v2):
docker version
```

## How to use this document

Work top to bottom.  Each batch tells you which compose to bring up
(if any), lists the scripts to run with a literal copy-paste command,
and gives the **signal** that confirms the row is green.  Tick the
corresponding row in the per-batch checklist at the bottom as you
go.  Re-run any flaky row **once** before declaring it red — see
[Known integration flakies](#known-integration-flakies) for the two
documented sensitivities and the model-swap escape hatch.

The four manual UI checks (`10_cli` REPL ×2, `09_serving` HTTP +
curl, `07_streaming` SSE) live in their own section because they
need a human at the keyboard.

---

## Batch A — Offline (no docker, no provider key)

No gateway needed.  Five scripts, ~1 minute end-to-end.

```bash
uv run python examples/agent/03_output_modes/recovery_demo.py
uv run python examples/agent/05_memory/easy.py
uv run python examples/agent/05_memory/advanced.py
uv run pytest examples/agent/11_testing/test_agent_unit.py -q
uv run python examples/agent/15_mcp_internals/main.py
```

**Signals (exact-match):**

- `03_output_modes/recovery_demo.py` — stdout begins with a header
  line `scenario  parse_strategy  attempts  inference_ids  parsed`
  and exactly four data rows labelled `strict`, `permissive`,
  `retry`, `none`.  Exit 0.
- `05_memory/easy.py` — stdout contains `demo episode has 2
  records`, then `search('COLOUR') → 1 match`, then `'other'
  episode has 1 record (isolation ok)`.  Exit 0.
- `05_memory/advanced.py` — stdout contains `Alice LRU cap
  enforced: kept 3 of 5 records`, `Bob isolation verified: 1
  record under 'session-1'`, and `(no cross-tenant leakage)`.
  Exit 0.
- `11_testing/test_agent_unit.py` — pytest summary line `N passed`
  with **0 skipped**.  Exit 0.
- `15_mcp_internals/main.py` — stdout contains `Discovered 2
  tool(s):` followed by two JSON schema blocks, `add(3, 7) = 10.0`,
  and `get_weather('Tokyo') = ...` with a non-empty body.  Exit 0.

---

## Batch B — Per-example gateway (cycle compose per folder)

Each folder has its own `docker-compose.yml` because the
`config/tensorzero.toml` differs.  Cycle them one at a time.
Eight cycles, ~13 minutes if every run is first-try green.

Per-folder pattern:

```bash
docker compose --env-file examples/agent/.env \
    -f examples/agent/<folder>/docker-compose.yml up -d
# Optional: wait for gateway health
docker compose -f examples/agent/<folder>/docker-compose.yml ps
<python invocations>
docker compose -f examples/agent/<folder>/docker-compose.yml down
```

### B.1 — `01_first_agent` (1 script)

```bash
uv run python examples/agent/01_first_agent/main.py
```

Signal: stdout shows `Response:` (non-empty), `Turns:`, `Episode
ID:`, `Inference IDs: [<uuid>]`.  Exit 0.

### B.2 — `02_agent_class` (3 scripts, shared compose)

```bash
uv run python examples/agent/02_agent_class/basic.py
uv run python examples/agent/02_agent_class/intermediate.py
uv run python examples/agent/02_agent_class/advanced.py
```

Signal: each script exits 0 with no traceback and prints a
non-empty `Response:` line.

### B.3 — `03_output_modes` (5 scripts, shared compose)

```bash
uv run python examples/agent/03_output_modes/main.py
uv run python examples/agent/03_output_modes/structured_agent.py
uv run python examples/agent/03_output_modes/structured_pydantic.py
uv run python examples/agent/03_output_modes/ner_dynamic_schema.py
uv run python examples/agent/03_output_modes/recovery_real_provider.py
```

Signals:

- `main.py` — `Parsed:` with a non-empty dict, `Parse strategy:`
  one of `strict`/`permissive`/`retry`.  Exit 0.
- `structured_agent.py` — two `--- run 1 ---` / `--- run 2 ---`
  blocks, each with `function_kind: json` and `parsed:` non-None.
- `structured_pydantic.py` — two runs, `value type: Customer` in
  each.  Requires the `pydantic` extra (installed during Setup).
- `ner_dynamic_schema.py` — four `=== Scenario N: ... ===` headers
  (1..4); `parsed:` populated in at least scenarios 1-3.  Scenario 4
  may legitimately land in `none` — accept either.
- `recovery_real_provider.py` — `parse_strategy:` populated; `parsed:`
  non-None on `strict`/`permissive`/`retry` and `None` on `none`.

### B.4 — `04_elicitation` (1 script, interactive — stdin stub)

```bash
printf 'y\nWrite the docs\nhigh\nGus\n' | \
    uv run python examples/agent/04_elicitation/main.py
```

Why the stub: the agent is told to list tasks, then delete task
`T-103` (one binary `Approve? [y/n]` prompt), then create a new
task (three structured fields: `title`, `priority`, `assignee` in
property order — see `04_elicitation/server.py`).  `y` accepts the
delete; the three following lines feed the structured create.

Signal: stdout contains `Deleted task: T-103 (Update README)`,
then `Created task: 'Write the docs' (priority: high, assignee:
Gus)`, then `Response:`, `Turns:`, `Inference IDs: [...]`.  Exit 0.

### B.5 — `07_streaming` (1 script)

`run.sh` and `agent_app.py` are the FastAPI/SSE server flow —
those go to [Manual UI verification](#manual-ui-verification).
Batch B exercises only the script demo:

```bash
uv run python examples/agent/07_streaming/advanced_run_handle.py
```

Signal: `SUCCESS: advanced_run_handle ran end-to-end.` after four
demo sections (`=== 1. Run.events() ===`, `=== 2. Run.cancel()
===`, `=== 3. Run.fork(extra_message) ===`, `=== 4. Run.replay()
===`).  Exit 0.

### B.6 — `08_multi_agent` (2 scripts, shared compose)

```bash
uv run python examples/agent/08_multi_agent/main.py
uv run python examples/agent/08_multi_agent/advanced.py
```

Signal: each exits 0 with a non-empty `Response:` line.

### B.7 — `13_observability` (1 script)

```bash
uv run python examples/agent/13_observability/main.py
```

Signal: stdout contains a `Boolean feedback sent:`, `Float
feedback sent:`, `Comment feedback sent:` line each with a UUID,
and the final line `Feedback is now stored in ClickHouse.`  Exit 0.

### B.8 — `14_multi_server` (1 script)

```bash
uv run python examples/agent/14_multi_server/main.py
```

Signal: `Available tools:` followed by `math_add`, `math_multiply`,
`text_word_count`, `text_reverse`.  Then `Response:` with a
non-empty body, `Turns:` ≥ 1.  Exit 0.

### B.9 — `11_testing` embedded (pytest, no docker)

Runs against the embedded gateway helper — no compose to cycle, but
needs `GROQ_API_KEY` since the helper performs real inference.

```bash
RUN_EMBEDDED_TESTS=1 uv run pytest examples/agent/11_testing \
    -m embedded -q
```

Signal: pytest summary `N passed` with 0 skipped, 0 failed.  Exit 0.

---

## Batch C — Shared composes (2 cycles, 12 rows)

Two compose files power many scripts; bring each up once, run all
its dependants, tear it down.  ~6 minutes if first-try green.

### C.1 — `01_first_agent` compose powers 5 follow-ons

```bash
docker compose --env-file examples/agent/.env \
    -f examples/agent/01_first_agent/docker-compose.yml up -d
```

| Script | Cmd | Signal |
|---|---|---|
| C.1.1 `05_memory/main.py` | `uv run python examples/agent/05_memory/main.py` | `=== InMemoryMemory ===` banner, `Turn 1:`, `Turn 2:`, `Memory records: 2`.  If `t0-agents[memory-sqlite]` is installed (yes per Setup), also `=== SqliteMemory ===` with two turns. |
| C.1.2 `05_memory/tensorzero_backend.py` | `uv run python examples/agent/05_memory/tensorzero_backend.py` | `=== Turn 1 (fresh episode) ===` with `episode_id:` and `reply:`, then `=== Turn 2 (same episode) ===` with `reply:` referencing "teal".  Exit 0. |
| C.1.3 `06_guardrails/main.py` (interactive — stdin stub) | `printf 'y\ny\ny\n' \| uv run python examples/agent/06_guardrails/main.py` | `APPROVAL REQUEST` banner with the `send_email` request; `Response:` non-empty, then `Guardrail log:` with ≥1 entry.  Extra `y` lines are harmless (EOF after the agent finishes).  Exit 0. |
| C.1.4 `06_guardrails/custom_guardrail.py` | `uv run python examples/agent/06_guardrails/custom_guardrail.py` | The literal line `Blocked by custom guardrail: recipient 'eve@evil.io' is not in the allowlist ['example.com']`, followed by INFO logs from `after_run` reporting one attempt with `verdict=BLOCKED`.  Exit 0. |
| C.1.5 `06_guardrails/advanced_middleware.py` | `uv run python examples/agent/06_guardrails/advanced_middleware.py` | `Response:`, `Elapsed: <N>s`, `Inferences seen: N`, per-inference lines, final `SUCCESS: advanced_middleware ran end-to-end.`  Exit 0. |

Teardown:

```bash
docker compose -f examples/agent/01_first_agent/docker-compose.yml down
```

### C.2 — `examples/glow` compose powers 8 glow + 3 concurrency

```bash
docker compose --env-file examples/glow/.env \
    -f examples/glow/docker-compose.yml up -d
```

| Script | Cmd | Signal |
|---|---|---|
| C.2.1 `glow/01_hello.py` | `uv run python examples/glow/01_hello.py` | `SUCCESS: hello-world ran end-to-end.` |
| C.2.2 `glow/02_structured.py` | `uv run python examples/glow/02_structured.py` | `SUCCESS: structured-output ran end-to-end.` |
| C.2.3 `glow/03_streaming.py` | `uv run python examples/glow/03_streaming.py` | `=== live event stream ===` block then `SUCCESS: streaming ran end-to-end.` |
| C.2.4 `glow/04_sequential.py` | `uv run python examples/glow/04_sequential.py` | `=== Step 1: summarize ===` and `=== Step 2: translate ===` blocks both render non-empty text.  No SUCCESS banner; exit 0 is the signal. |
| C.2.5 `glow/05_parallel.py` | `uv run python examples/glow/05_parallel.py` | `=== Fan-out result ===` followed by three labelled lines (`Optimist`, `Pessimist`, `Pragmatist`).  Exit 0. |
| C.2.6 `glow/06_orchestrator.py` | `uv run python examples/glow/06_orchestrator.py` | `=== Orchestrator output ===` with non-empty text.  Exit 0. |
| C.2.7 `glow/07_di_tools.py` | `uv run python examples/glow/07_di_tools.py` | `SUCCESS: di-tools ran end-to-end.` |
| C.2.8 `glow/08_mcp_typing.py` | `uv run python examples/glow/08_mcp_typing.py` | `Visible tools: ['math_add']`, `Response:` non-empty, then `SUCCESS: mcp-typing ran end-to-end.` |
| C.2.9 `12_concurrency/01_sequential.py` | `uv run python examples/agent/12_concurrency/01_sequential.py` | Step banners + non-empty per-step text.  Exit 0. |
| C.2.10 `12_concurrency/02_parallel.py` | `uv run python examples/agent/12_concurrency/02_parallel.py` | Fan-out section with labelled lines.  Exit 0. |
| C.2.11 `12_concurrency/03_orchestrator.py` | `uv run python examples/agent/12_concurrency/03_orchestrator.py` | Orchestrator output block.  Exit 0. |

Teardown:

```bash
docker compose -f examples/glow/docker-compose.yml down
```

---

## Manual UI verification

Four flows that need a human at the keyboard.  The first two are
the Typer + Rich TUI; the last two are HTTP servers exercised
with `curl` or a browser.

### M.1 — `10_cli/agent.py` REPL

1. Bring up the `01_first_agent` compose if it is down.
2. Run: `uv run t0-agents run examples.agent.10_cli.agent:assistant`.
3. Type `What is 2 + 3?` and press Enter.
4. Type `Ctrl-D` (EOF) to exit cleanly.

Outcome: a cyan panel titled `tui-assistant` shows the tool call
(`🔧 add(...)`) then its result (`✓ add(...)`) then a final
answer mentioning `5`.  No traceback on exit.

### M.2 — `10_cli/advanced.py` REPL with audit log

1. Run: `uv run t0-agents run
   examples.agent.10_cli.advanced:assistant_with_audit`.
2. Type a reminder-style prompt (e.g. `Remind me tomorrow at 9 to
   water the plants.`) and press Enter.
3. When the Rich elicit dialog appears, accept defaults (Enter
   through each field).
4. Type `Ctrl-D` to exit.
5. Inspect the audit log: `cat tui-audit.jsonl`.

Outcome: at least two JSON lines in `tui-audit.jsonl` with
matching timestamps — one `"kind": "elicit_request"` and one
`"kind": "elicit_response"`.

### M.3 — `09_serving` FastAPI + curl smoke

1. Bring up the `09_serving` compose:
   `docker compose --env-file examples/agent/.env -f
   examples/agent/09_serving/docker-compose.yml up -d`.
2. In a second terminal, start the server:
   `bash examples/agent/09_serving/run.sh`.
3. In a third terminal, exercise the endpoints:
   `bash examples/agent/09_serving/client.sh`.
4. `Ctrl-C` the server, then `docker compose -f
   examples/agent/09_serving/docker-compose.yml down`.

Outcome: `client.sh` prints labelled blocks for `GET /health`
(status `ok`), `GET /info` (model + function name), `POST /chat`
(non-empty `response` + `episode_id`), the continuation `POST
/chat` (same `episode_id`), and `POST /feedback`.  The feedback
call may return `404` if the demo metric is not declared in
`tensorzero.toml` — see the comment in `client.sh`; that is
acceptable.

### M.4 — `07_streaming` SSE

1. Bring up the `07_streaming` compose:
   `docker compose --env-file examples/agent/.env -f
   examples/agent/07_streaming/docker-compose.yml up -d`.
2. Start the server: `bash examples/agent/07_streaming/run.sh`.
3. In a second terminal, hit the stream:
   `curl -N 'http://localhost:8080/chat/stream?message=Say%20hello&episode_id='`.
4. Optional: open `http://localhost:8080/static/index.html` in a
   browser and send one message through the UI.
5. `Ctrl-C` the server and tear down the compose.

Outcome: the curl call emits `event: ...` SSE lines and ends with
a terminal `event: done` (or equivalent close marker).  The
browser, if used, renders without console errors.

---

## Known integration flakies

Two `-m integration` tests are sensitive to Groq's `openai/gpt-oss-20b`
default model and may fail in ways that are not regressions of the
code under test.  Both are documented here so the operator
recognises the symptom, rerolls once, and only declares red if the
swap-to-alt-model escape hatch also fails.

If the smoke session runs `uv run pytest -m integration` end to end,
expect to see these two surface intermittently.

### F.1 — `test_require_approval_auto_decline_surfaces_to_llm`

Path: `tests/test_guardrails/test_integration.py`

Symptom (shape):

```
FAILED tests/test_guardrails/test_integration.py::test_require_approval_auto_decline_surfaces_to_llm
AssertionError: assert any(e.kind == 'rejection' for e in result.guardrail_log)
 +  where [] = list(result.guardrail_log)
```

Root cause: the test asserts the LLM observed a `rejection`
guardrail entry, which only lands if the model actually invokes the
gated tool.  `openai/gpt-oss-20b` occasionally answers in plain
text without calling any tool, so the guardrail log stays empty.

Reroll: re-run once.

```bash
uv run pytest -m integration -q \
    tests/test_guardrails/test_integration.py::test_require_approval_auto_decline_surfaces_to_llm
```

If it fails twice, swap to a non-Groq model per the
[model-swap escape hatch](#model-swap-escape-hatch) below.

### F.2 — `test_tool_timeout_does_not_crash_run_agent`

Path: `tests/test_mcp/test_integration.py`

Symptom (shape):

```
FAILED tests/test_mcp/test_integration.py::test_tool_timeout_does_not_crash_run_agent
tensorzero.types.TensorZeroError: 502 ... Tool call validation failed ...
```

Root cause: the test exercises a tool-timeout path that produces an
error `ToolResult`.  When the loop retries, Groq occasionally emits
malformed tool-call arguments and rejects them with `400 Tool call
validation failed`; the gateway re-wraps that as
`502 TensorZeroError`, which is outside the test's accepted-outcomes
set.

Reroll: re-run once.

```bash
uv run pytest -m integration -q \
    tests/test_mcp/test_integration.py::test_tool_timeout_does_not_crash_run_agent
```

If it fails twice, swap to a non-Groq model per the
[model-swap escape hatch](#model-swap-escape-hatch) below.

### Model-swap escape hatch

Both flakes go away on tool-deterministic providers (Claude, GPT-4o,
Gemini).  To swap the variant used by the dev integration stack:

1. Open `config/tensorzero.toml` at the repo root (this is the file
   the root `docker-compose.yml` mounts and integration tests hit).
2. Replace the `openai/gpt-oss-20b` provider block with one of:

   ```toml
   # Anthropic
   [models.smoke_alt.providers.anthropic]
   type = "anthropic"
   model_name = "claude-3-5-sonnet-latest"
   ```

   ```toml
   # OpenAI
   [models.smoke_alt.providers.openai]
   type = "openai"
   model_name = "gpt-4o"
   ```

   ```toml
   # Google
   [models.smoke_alt.providers.google_ai_studio_gemini]
   type = "google_ai_studio_gemini"
   model_name = "gemini-2.0-flash"
   ```

   …and re-point each variant's `model` to `smoke_alt` (or rename
   in place; either works).
3. Put the matching API key (`ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`, or `GEMINI_API_KEY`) in the repo-root `.env`.
4. Restart the dev stack: `docker compose down && docker compose up
   -d`.  Re-run the two tests.

**Revert the TOML before tagging** — the swap is a smoke-time
expedient only.  Trunk stays pinned to the gpt-oss-20b default.
Record the reroll (re-run vs swap) in the release notes so future
operators have the same map.

---

## Pre-release checklist

Tick each row as it goes green.  Re-runs allowed once per row
before declaring red.

### Batch A — Offline

| ✓ | Row | Cmd |
|---|---|---|
| ☐ | A.1 | `uv run python examples/agent/03_output_modes/recovery_demo.py` |
| ☐ | A.2 | `uv run python examples/agent/05_memory/easy.py` |
| ☐ | A.3 | `uv run python examples/agent/05_memory/advanced.py` |
| ☐ | A.4 | `uv run pytest examples/agent/11_testing/test_agent_unit.py -q` |
| ☐ | A.5 | `uv run python examples/agent/15_mcp_internals/main.py` |

### Batch B — Per-example docker

| ✓ | Row | Compose | Cmd |
|---|---|---|---|
| ☐ | B.1 | 01_first_agent | `…/01_first_agent/main.py` |
| ☐ | B.2a | 02_agent_class | `…/02_agent_class/basic.py` |
| ☐ | B.2b | 02_agent_class | `…/02_agent_class/intermediate.py` |
| ☐ | B.2c | 02_agent_class | `…/02_agent_class/advanced.py` |
| ☐ | B.3a | 03_output_modes | `…/03_output_modes/main.py` |
| ☐ | B.3b | 03_output_modes | `…/03_output_modes/structured_agent.py` |
| ☐ | B.3c | 03_output_modes | `…/03_output_modes/structured_pydantic.py` |
| ☐ | B.3d | 03_output_modes | `…/03_output_modes/ner_dynamic_schema.py` |
| ☐ | B.3e | 03_output_modes | `…/03_output_modes/recovery_real_provider.py` |
| ☐ | B.4 | 04_elicitation | `…/04_elicitation/main.py` (stdin stub) |
| ☐ | B.5 | 07_streaming | `…/07_streaming/advanced_run_handle.py` |
| ☐ | B.6a | 08_multi_agent | `…/08_multi_agent/main.py` |
| ☐ | B.6b | 08_multi_agent | `…/08_multi_agent/advanced.py` |
| ☐ | B.7 | 13_observability | `…/13_observability/main.py` |
| ☐ | B.8 | 14_multi_server | `…/14_multi_server/main.py` |
| ☐ | B.9 | (none) | `RUN_EMBEDDED_TESTS=1 uv run pytest examples/agent/11_testing -m embedded` |

### Batch C — Shared composes

| ✓ | Row | Compose | Cmd |
|---|---|---|---|
| ☐ | C.1.1 | 01_first_agent | `…/05_memory/main.py` |
| ☐ | C.1.2 | 01_first_agent | `…/05_memory/tensorzero_backend.py` |
| ☐ | C.1.3 | 01_first_agent | `…/06_guardrails/main.py` (stdin stub) |
| ☐ | C.1.4 | 01_first_agent | `…/06_guardrails/custom_guardrail.py` |
| ☐ | C.1.5 | 01_first_agent | `…/06_guardrails/advanced_middleware.py` |
| ☐ | C.2.1 | glow | `examples/glow/01_hello.py` |
| ☐ | C.2.2 | glow | `examples/glow/02_structured.py` |
| ☐ | C.2.3 | glow | `examples/glow/03_streaming.py` |
| ☐ | C.2.4 | glow | `examples/glow/04_sequential.py` |
| ☐ | C.2.5 | glow | `examples/glow/05_parallel.py` |
| ☐ | C.2.6 | glow | `examples/glow/06_orchestrator.py` |
| ☐ | C.2.7 | glow | `examples/glow/07_di_tools.py` |
| ☐ | C.2.8 | glow | `examples/glow/08_mcp_typing.py` |
| ☐ | C.2.9 | glow | `…/12_concurrency/01_sequential.py` |
| ☐ | C.2.10 | glow | `…/12_concurrency/02_parallel.py` |
| ☐ | C.2.11 | glow | `…/12_concurrency/03_orchestrator.py` |

### Manual UI

| ✓ | Row | What | Where |
|---|---|---|---|
| ☐ | M.1 | `10_cli/agent.py` REPL | 01_first_agent compose |
| ☐ | M.2 | `10_cli/advanced.py` REPL + audit | 01_first_agent compose |
| ☐ | M.3 | `09_serving` FastAPI + curl | 09_serving compose |
| ☐ | M.4 | `07_streaming` SSE | 07_streaming compose |

### Known flakies — acknowledged

| ✓ | Test | Acknowledged green / rerolled |
|---|---|---|
| ☐ | `test_require_approval_auto_decline_surfaces_to_llm` |  |
| ☐ | `test_tool_timeout_does_not_crash_run_agent` |  |

---

## Sign-off

```
Operator:      ________________________
Date:          ________________________
Git rev:       ________________________  (output of `git rev-parse HEAD`)
Status:        ☐ All batches green
               ☐ Known flakies acknowledged (or rerolled and green)
```
