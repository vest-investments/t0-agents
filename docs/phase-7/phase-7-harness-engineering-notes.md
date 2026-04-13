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

### 2.4 Handle Thought-only responses (addresses O2)

Today, if the terminating turn contains only `Thought` blocks (common
for reasoning models), `result.response` is `""`. Downstream callers
that do `if not result.response: raise` will crash; callers that don't
check get a silent empty reply.

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

Phase 6 (Observability) is already in the meta-plan; these are
additional items specifically targeting tool-selection debuggability.

### 5.1 Per-turn structured log (addresses all)

Today `run_agent` emits no log lines. When a test flakes against the
real server, the only evidence is the final exception. Adding a
structured log like:

```
turn=1 inference_id=... tools=[add,greet,...] tool_call=execute_sql args={...} duration_ms=432
turn=1 tool_result len=145 is_error=False preview="[{"rows":..."
turn=2 inference_id=... response_content_blocks=[Thought,Text] response_text="..."
```

would make post-hoc diagnosis trivial. `logging` module at INFO level
with a `t0_agents.loop` logger so callers can silence.

### 5.2 Capture `Thought` content (addresses O2)

When `response.content` includes `Thought` blocks (reasoning models),
log them at DEBUG. They often reveal *why* the model chose (or skipped)
a tool call.

### 5.3 Raw `ToolCall.raw_name` / `raw_arguments` always logged (addresses O1)

When we reject a malformed tool call (§ 2.1), log the raw strings so
we can build a corpus of real failure cases — useful for model-switch
benchmarking (§ 3.3) and for prompt engineering (§ 3.1).

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

---

## Priority ordering

When Phase 7 is actually scoped for implementation, the order below
maximizes reliability-per-effort based on what we saw:

1. **§ 3.1 system prompt engineering** — highest leverage, low code
   change, affects O3/O4/O5 directly.
2. **§ 5.1 structured per-turn logging** — prerequisite for measuring
   everything else.
3. **§ 6.1 flake-rate benchmark harness** — required to evaluate § 3.3
   without shipping it blindly.
4. **§ 3.3 model replacement evaluation** — likely kills O1/O2 at the
   source.
5. **§ 2.1 `<|channel|>` error capture** — last-resort mitigation if
   we have to stay on Groq gpt-oss-20b.
6. **§ 2.4 Thought-only handling** — small caller-facing fix with
   outsized debuggability value.
7. **§ 7.1/§ 7.2 caller-facing API additions** — after all loop-side
   mitigations have landed.
8. **§ 2.2 / § 2.3 / § 2.5** — incremental loop-layer knobs. Tackle
   only if the benchmark harness shows they close a gap.

Items not listed above (§ 3.2, § 4.1, § 4.2, § 5.2, § 5.3, § 6.2) are
worth doing but their effort-to-impact ratio is worse than the top 8.
