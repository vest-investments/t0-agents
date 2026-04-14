# Phase 7 — Harness Engineering for Tool-Use Reliability

> **Status:** future work. This document is a punch list of candidate
> improvements to reduce the rate at which agents fail at the
> *tool-use layer*: selecting the right tool, producing clean tool
> arguments, recovering from errors, and terminating with a useful
> response. Nothing here ships in Phase 5 or Phase 6 — it is the catalog
> we will triage from when reliability becomes the focus.

> **Provenance.** Every item below is traced to a concrete failure
> observed either in the Phase 5 integration suite (`tests/test_mcp/
> test_integration.py`, `tests/test_mcp/test_robustness.py`) or in a
> manual sanity-check run against the production `vest-mcp-gcp` MCP
> server (CF-Access-protected BigQuery / Discovery Engine tools). The
> underlying inference in all cases was Groq's
> `openai/gpt-oss-20b` via TensorZero — some items are specific to that
> model/provider, others are model-agnostic loop-engineering concerns.

> **Post-Phase-6 update (2026-04-14).** Phase 6 shipped structured
> logging (F6.7), feedback integration (F6.2-F6.5), and ClickHouse
> dynamic-tools verification (F6.6). Several items below are now
> partially or fully addressed; these are marked with a **[P6]** tag
> and a note on what remains. New items discovered during Phase 6 are
> marked **[NEW-P6]**.

---

## 1. Observed failure modes

These are the patterns we saw during Phase 5 validation. Each gets a
label so the mitigations in §§ 2–7 can reference them.

| # | Label | What it looks like | Where we saw it |
|---|---|---|---|
| O1 | **`<\|channel\|>` token leak** | Tool name arrives with a commentary artifact (`add<\|channel\|>commentary`). T0 rejects with `400 Bad Request: tool call validation failed: … not in request.tools`. Evidence that the retry actually fired, but the name is unusable. | Phase 5 `test_f5_1_...` / `test_f5_5_...` full-suite runs; Groq gpt-oss-20b. |
| O2 | **Thought-only response** | `response.content = [Thought(...)]` with no `Text` block. `result.response` ends up empty string. `turns == 1`. | Phase 5 sanity-check run against `vest-mcp-gcp` (F5.1 prompt). |
| O3 | **Ambiguity → clarify instead of call** | LLM asks the user for a missing argument in text instead of invoking an available tool. e.g. "¿Cuál es el `project_id` de GCP?" | Phase 5 sanity-check run against `vest-mcp-gcp` (F5.4 prompt without explicit project_id). |
| O4 | **Retry-until-exhausted** | When every tool call fails (F5.1 failing tool, F5.4 timeout), the LLM keeps calling the same tool until `max_turns` runs out. `MaxTurnsExceededError` fires instead of a graceful text wrap-up. | Phase 5 in-memory + sanity-check runs. |
| O5 | **Tool selection skipped** | On a prompt that reliably produces a tool call with `tool_choice={"specific": "X"}` in Phase 1/2, the same prompt without `tool_choice` flakes ≈5–10% of runs in Phase 5 (LLM wraps up in text without calling the tool). | Phase 5 `test_f5_7_...`, `test_f5_8_...` full-suite runs. |
| O6 | **Pre-existing Phase-1 flake** | `test_tool_choice_forces_tool_call` (Phase 1) fails intermittently with the same O1 symptom. | Phase 5 full-suite stability runs. Not Phase-5 scope to fix, but same root cause as O1. |

## 2. Mitigations at the agent-loop layer (`src/t0_agents/mcp/loop.py`)

These change `run_agent`'s behavior — they are the layer with the most
leverage because every caller sees them.

### 2.1 Capture `<|channel|>`-like artifacts and feed them back as errors (addresses O1)

Today, a malformed tool name causes T0 to reject the entire inference
with `TensorZeroError`, which bubbles out of `run_agent` uncaught. There
is no chance to recover in-loop.

**Proposal:** catch `TensorZeroError` whose body contains
`"not in request.tools"`, parse out the malformed name from the error
payload, and synthesize a `ToolResult` with a corrective instruction:

> `"Error: the tool name 'add<|channel|>commentary' is not valid. Strip
> any '<|...|>' tokens before emitting the tool call. Use one of: [...]"`.

Append to history, continue the loop. The LLM will typically retry
clean on the next turn.

**Trade-off:** couples the loop to an error-message substring. If T0
changes that wording the regex breaks. Mitigation: guard behind a
feature flag (`sanitize_malformed_tool_names: bool = False`) and pin
with a unit test that runs only when the flag is set.

### 2.2 Per-tool retry budget (addresses O4)

Today, `max_turns` is a **total** budget. A single misbehaving tool
(F5.1 / F5.4) can consume all of it via back-to-back retries. The LLM
has no signal that it should give up on that specific tool.

**Proposal:** keep a per-tool-call-id retry counter; after N consecutive
failures of the same tool, inject an extra sentence into the error
`ToolResult`:

> `"Error: ... (this tool has failed 3 times in a row. Consider a
> different tool or ask the user for clarification.)"`.

Also emit a structured log line so harness-level code can detect the
pattern and bail the loop before `max_turns` triggers.

**Trade-off:** more state. Default budget (3? 5?) is a tuning knob.

### 2.3 Expose `tool_choice` on `run_agent` (addresses O5)

Phase 1/2 direct-inference tests reliably produce tool calls with
`tool_choice={"specific": "X"}`. Phase 5 integration tests that use the
same prompt *without* `tool_choice` flake ≈5–10% of runs.

**Proposal:** add `tool_choice: Optional[T0ToolChoice] = None` to
`run_agent`'s signature. Forward it to `t0.inference()` on every turn.
Document that callers who know which tool should be used (benchmark
harnesses, directed integration tests, UI flows where the button
implies the tool) should pass it.

**Trade-off:** widens the public API. Current design deliberately left
`tool_choice` out because it's a per-inference concern — we'd be saying
"every turn uses the same tool_choice", which isn't quite right for
multi-step flows. A more nuanced signature would accept a callable
`tool_choice: Callable[[list[Message]], T0ToolChoice | None]`, but
that's harder to document.

### 2.4 Handle Thought-only responses (addresses O2) — **[P6] now observable, still unhandled**

Today, if the terminating turn contains only `Thought` blocks (common
for reasoning models), `result.response` is `""`. Downstream callers
that do `if not result.response: raise` will crash; callers that don't
check get a silent empty reply.

**Phase 6 update:** with structured logging (F6.7), a Thought-only
response now produces a visible `"Agent loop completed: turns=N"` INFO
log followed by an empty-string response. The failure mode is now
*discoverable* in logs but still *unhandled* in code. The text
extraction at `loop.py:247-249` filters `isinstance(b, Text) and
b.text` — `Thought` blocks with `text: Optional[str]` and
`summary: Optional[List[ThoughtSummaryBlock]]` fall through silently.
This makes § 2.4 higher priority: the user sees the empty response in
logs, expects the library to have handled it.

**Proposal:** when the final turn has no `Text` blocks but has one or
more `Thought` blocks, either:
1. Extract the last `Thought.text` as a fallback response (with a
   boolean flag `response_was_thought_only: bool` on `AgentResult`), or
2. Raise a new `EmptyResponseError` with the thought content attached,
   so the caller can decide whether to retry or surface it.

Preferred: option (1), default behavior changes but caller-visible flag
lets them opt into option (2).

### 2.5 Timeout-per-tool instead of global (partial O4)

Today `tool_timeout` is a single float applied uniformly. In practice,
`list_dataset_ids` (fast) and `execute_sql` on a billion-row table
(slow) want very different budgets.

**Proposal:** accept `tool_timeouts: dict[str, float] | None` that
overrides the default per tool name. Preserve the single-float form as
a default.

## 3. Mitigations at the TensorZero config layer (`config/tensorzero.toml`)

### 3.1 System prompt for the tool-using agent (addresses O3, O5)

Today `config/tensorzero.toml` references `functions/agent/baseline/
system.txt` — a minimal prompt. The Phase 5 sanity-check failures show
the LLM defaulting to "ask for clarification" rather than "call the
tool with what you have". That is prompt behavior, not loop behavior.

**Proposal:** craft a system prompt that explicitly:

- Lists the available tools at the start of every conversation (T0 /
  the adapter could inject this dynamically from `list_tools`).
- Directs the model to *try a tool call with a sensible default before
  asking the user* when possible.
- Specifies an error-recovery policy: "if a tool fails twice with the
  same args, try different args; if it fails three times, report the
  error to the user in plain text and stop".
- Caps reasoning to 1–2 sentences before the tool call (reduces the
  Thought-only response rate on reasoning models).

**Trade-off:** longer system prompt costs tokens every turn. Worth
measuring against flake-rate reduction before shipping.

### 3.2 Multi-variant config with a fallback model (addresses O1, O2)

Today the `agent` function has a single variant (`gpt_oss_20` on Groq).
Groq's `gpt-oss-20b` is the source of O1, O2, and O5.

**Proposal:** add a second variant backed by a different model (Claude
Haiku 4.5, OpenAI GPT-4.1-mini, or Groq's larger `gpt-oss-120b`) and
use T0's routing to fall back on the second model when the first
produces a `TensorZeroError`. Document which variant is "primary" based
on the reliability/cost trade-off at the time.

**Trade-off:** requires adding `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
to `.env.example`, plus the T0 config surfaces. User-visible scope
creep.

### 3.3 Evaluate model replacement (addresses O1, O2, O6)

`openai/gpt-oss-20b` is the cheapest tool-use-capable model on Groq
today (April 2026), but it exhibits the `<|channel|>` bug consistently.
Larger Groq models (`gpt-oss-120b`, `llama-3.3-70b-versatile`) and
non-Groq providers do not have the bug.

**Proposal:** run a benchmark of Phase 5's integration tests against 3
candidate models (Groq gpt-oss-120b, Anthropic Claude Haiku 4.5, OpenAI
GPT-4.1-mini) and measure the flake rate across 20 full-suite runs per
model. Ship the one with the best reliability-per-dollar.

## 4. Mitigations at the MCP-server layer

### 4.1 Descriptive tool errors with remediation hints (addresses O4)

Today, if a tool fails (e.g. BigQuery 404 on a nonexistent table), the
error comes back as a raw MCP error response containing whatever text
the tool author wrote. The LLM sees `"Error: tool execution failed:
Not found: Dataset X"` but has no hint what to do.

**Proposal:** MCP-server authors should wrap errors with remediation
hints:

```python
@mcp.tool
def execute_sql(project_id: str, query: str) -> str:
    try:
        return client.query(query, project=project_id).to_json()
    except NotFound as e:
        raise ToolError(
            f"Not found: {e}. "
            "Hint: call list_dataset_ids(project_id) first to check the "
            "available datasets, or list_table_ids(dataset) to check tables."
        )
```

This is a server-side convention, not a loop-side change — document it
in the FastMCP integration guide rather than enforce it.

### 4.2 Tool metadata: "typical latency" and "expensive" flags (addresses O4, 2.5)

Today a tool's schema has no machine-readable signal about how fast or
expensive it is. The adapter could convert FastMCP's `tool.tags`
(already supported) into per-tool timeout hints in `run_agent` if we
agreed on tag names (e.g. `"slow"`, `"expensive"`, `"interactive"`).

## 5. Observability / debugging improvements

Phase 6 shipped structured logging (F6.7) and feedback integration.
The items below reflect what remains after Phase 6.

### 5.1 Per-turn structured log (addresses all) — **[P6] mostly done**

Phase 6 added `logging.getLogger("t0_agents.mcp.loop")` with:

- INFO: loop start (function, max_turns, tool count), per-turn
  summary (inference_id, episode_id), loop completion (turns, episode_id),
  max-turns warning.
- DEBUG: tool dispatch count per turn, tool execution (name + args),
  tool result (name + truncated result).
- ERROR: tool failures (ToolError, McpError, unexpected Exception),
  name-not-in-schema, invalid-JSON-arguments.

**Three micro-gaps remain** from the original sketch:

1. **`duration_ms` per tool call.** The sketch showed `duration_ms=432`
   but Phase 6 does not time `call_tool()`. Adding
   `time.monotonic()` around the call in `_execute_single_tool_call`
   is ~4 lines and feeds § 2.5 (timeout-per-tool) and § 4.2 (latency
   metadata). **High value, low effort.**

2. **Tool names in the dispatch log.** Phase 6 logs the *count*
   (`"dispatching 3 tool call(s)"`) but not *which* tools. Changing to
   `"dispatching 3 tool call(s): [add, greet, search]"` costs one
   list comprehension.

3. **Content-type breakdown per turn.** The per-turn INFO log shows
   `inference_id` and `episode_id` but not what the response contains.
   Logging `content_types=[Thought, ToolCall, ToolCall]` would let
   O2 (Thought-only) and O5 (tool-selection skipped) be diagnosed
   from INFO logs alone, without needing DEBUG.

### 5.2 Capture `Thought` content (addresses O2) — **[P6] insertion point known**

Phase 6 revealed the exact location: `loop.py:246`
(`if not has_tool_calls(response.content)`) is where Thought blocks
disappear. `Thought.text: Optional[str]` and
`Thought.summary: Optional[List[ThoughtSummaryBlock]]` both carry
useful content. A single `logger.debug()` call there completes this
item. Now trivial — the infrastructure is in place.

### 5.3 Raw `ToolCall.raw_name` / `raw_arguments` always logged (addresses O1) — **[P6] partially done**

Phase 6 logs `block.raw_name` in the `name is None` error branch and
`block.raw_arguments` in the `arguments is None` branch. But in the
*normal* flow path, only `block.name` and `block.arguments` are
logged — the raw forms are invisible when things work. The original
intent was a corpus of *all* raw data, not just error cases.

### 5.4 ClickHouse trace utility (addresses all) — **[NEW-P6]**

Phase 6's F6.6 test proved we can query ClickHouse's HTTP API via
`httpx` with minimal code (credentials in `docker-compose.yml`, the
`tensorzero` database is always present). A debugging script or
library function that takes an `episode_id` and prints the full
conversation trace would be a force multiplier for diagnosing O1-O6:

```
scripts/trace_episode.py <episode_id>
→ Turn 1: inference=abc  input=[user:"compute 2+3"]  output=[ToolCall:add]
  Turn 1: tool_result=add  result="5"  is_error=False
  Turn 2: inference=def  input=[user:tools]  output=[Text:"The sum is 5"]
  Feedback: task_success=True (episode)  quality_score=9.0 (inference=abc)
```

Data sources per turn: `ChatInference.input` (JSON), `.output`
(JSON), `.dynamic_tools` (Array), plus join to
`BooleanMetricFeedback` / `FloatMetricFeedback` / `CommentFeedback`
/ `DemonstrationFeedback` on `target_id`.

### 5.5 Document `dryrun` constraint for feedback automation — **[NEW-P6]**

Phase 6 discovered (divergence #1 in `phase-6-implementation-log.md`)
that `t0.feedback()` validates the target ID exists in ClickHouse.
When inference uses `dryrun=True`, the IDs are generated but never
stored, so feedback calls fail with `"Episode ID: ... does not exist"`
(HTTP 400).

This is a **design constraint** for any future feedback automation
(§ 7.2's pluggable callback, auto-success-feedback after loop
completion, etc.): feedback cannot work when `dryrun=True`.
Document this explicitly in `run_agent`'s docstring and in the
FastMCP integration guide. If § 7.2 ships, the callback must either
skip feedback when `dryrun=True` or raise a clear error.

## 6. Testing / harness improvements

### 6.1 Flake-rate benchmark harness

Phase 5's integration tests already skip on Groq flakes. A separate
harness (not part of the default `pytest -m integration`) that runs
each F5.x scenario N times and reports the skip-rate / fail-rate /
pass-rate would let us measure the effect of each mitigation before
shipping.

Sketch:

```
scripts/flake_benchmark.py --runs 20 --suite phase5 --model groq:gpt-oss-20b
→ F5.1: 17 pass, 2 skip, 1 fail (5% fail)
  F5.4: 18 pass, 2 skip (0% fail)
  F5.7: 20 pass
```

Run this before/after each mitigation from § 2–5. The goal is monotone
improvement of pass-rate (no regressions).

### 6.2 Tool-call corpus from the sanity-check runs

The `scripts/phase5_sanity.py` snippet against `vest-mcp-gcp` generates
real failure cases every run (O1 leaks, O3 clarifications, O5 skips).
Bank them into a `tests/fixtures/flake_corpus.json` so unit tests can
replay them against candidate mitigations without needing a live
gateway.

## 7. Caller-facing improvements

### 7.1 `AgentResult.error: str | None` field (addresses O2, O4)

Today `run_agent` either returns a clean `AgentResult` or raises. If
the loop exhausts `max_turns` after catching every tool error, the
caller loses the last assistant response. We could either:

1. Return `AgentResult(response="", error="max_turns_exceeded", ...)`
   with the partial history attached, and let the caller decide whether
   to raise.
2. Keep the raise but attach `partial_result: AgentResult` to the
   `MaxTurnsExceededError` so the caller can recover useful context.

Option (2) is a strictly additive API change and easier to roll out.

### 7.2 Pluggable "give up" callback (addresses O4, O5)

Expose a `on_tool_error: Callable[[ToolCall, ToolResult], bool]`
callback that fires after each error ToolResult and lets the caller
decide whether to continue the loop. Returning `False` short-circuits
to a clean `AgentResult` with whatever text the model has produced so
far.

**[P6] constraint:** any callback that sends feedback must respect the
`dryrun` constraint discovered in Phase 6 — `t0.feedback()` validates
the target ID exists in ClickHouse, so feedback calls fail when
`dryrun=True`. The callback must either skip feedback when `dryrun` is
set, or the loop must propagate the `dryrun` flag to the callback. See
§ 5.5 for details.

---

## Priority ordering

**Updated after Phase 6 (2026-04-14).** Items marked [DONE] have been
fully delivered; items marked [MICRO] are residual gaps from a mostly-
done item.

1. **§ 3.1 system prompt engineering** — highest leverage, low code
   change, affects O3/O4/O5 directly. (unchanged)
2. ~~§ 5.1 structured per-turn logging~~ — **[DONE]** by Phase 6
   (F6.7). Three micro-gaps remain:
   - **[MICRO] § 5.1a** `duration_ms` per tool call (~4 lines).
   - **[MICRO] § 5.1b** tool names in dispatch log (1 list comp).
   - **[MICRO] § 5.1c** content-type breakdown per turn (1 log line).
3. **§ 5.2 Thought content at DEBUG** — now trivial: one
   `logger.debug()` at `loop.py:246`, exact insertion point known
   from Phase 6.
4. **§ 2.4 Thought-only handling** — *promoted* from #6 to #4.
   Phase 6 logging made O2 discoverable but still unhandled;
   users who enable logging will see the empty response and expect
   the library to have a fallback.
5. **§ 6.1 flake-rate benchmark harness** — required to evaluate
   § 3.3 without shipping it blindly. (unchanged)
6. **§ 3.3 model replacement evaluation** — likely kills O1/O2 at the
   source. (unchanged)
7. **§ 5.4 ClickHouse trace utility** — **[NEW-P6]** debugging
   multiplier, builds on F6.6 infrastructure (`httpx` + ClickHouse
   HTTP API). Low effort, high payoff for diagnosing O1-O6.
8. **§ 2.1 `<|channel|>` error capture** — last-resort mitigation if
   we have to stay on Groq gpt-oss-20b. (unchanged)
9. **§ 7.1/§ 7.2 caller-facing API additions** — after all loop-side
   mitigations have landed. § 7.2 now carries a `dryrun` constraint
   from Phase 6 (see § 5.5).
10. **§ 5.5 document `dryrun` constraint** — **[NEW-P6]** small but
    load-bearing for any future feedback automation.
11. **§ 2.2 / § 2.3 / § 2.5** — incremental loop-layer knobs. Tackle
    only if the benchmark harness shows they close a gap. § 2.5 now
    has a path via § 5.1a timing data.

Items not listed above (§ 3.2, § 4.1, § 4.2, § 5.3, § 6.2) are
worth doing but their effort-to-impact ratio is worse than the top 11.
