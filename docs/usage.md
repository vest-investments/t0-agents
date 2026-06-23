# Token usage — live footprint, billing, and budgets

Every TensorZero inference reports token usage, and t0-agents surfaces it
two ways that answer two different questions:

- **"How much cost have I run up?"** — the *cumulative* sum of every
  inference in a `Run`, via `Run.usage()` / `Run.cost()`.
- **"How much of the context window is left — should I compact?"** — the
  *footprint*, the latest inference's `input_tokens` (≈ the current
  full-conversation prompt size), via `Run.context_footprint()`.

These are not the same number, and conflating them is the single most
common usage mistake.  Each turn of a conversation re-sends the growing
prompt, so **summing `input_tokens` double-counts** — the cumulative total
keeps climbing even when the live prompt fits comfortably in the window.
For the compaction decision you want the *level*, not the sum.

> This page is the canonical guide to token usage.  For the `Run` handle
> the in-process API hangs off, see [Run](run.md); for the always-on
> `CostTrackingMiddleware` that populates it, see
> [Middleware](middleware.md); for the wire shape, see
> [Serving](serving.md).  Budget enforcement and historical billing are
> deliberately **delegated to TensorZero** — the recipes are at the end.

---

## Footprint vs. billing at a glance

| | `Run.usage()` / `Run.cost()` | `Run.context_footprint()` |
|---|---|---|
| **Metric** | Cumulative sum across every inference | Latest inference's `input_tokens` (a *level*) |
| **Answers** | "What did this run cost?" | "How full is the context window?" |
| **Across turns** | Monotonically grows | Rises and falls with the live prompt |
| **Sub-agents** | Includes delegated children (rolled up) | Parent-only (a child has its own window) |
| **Type** | `RunUsage` (root-exported) / `float` | `Usage \| None` (`core.types`) |

The footprint is the signal an outer layer compares against the model's
context window to decide when to **compress or rotate** a conversation
before quality degrades.  The library exposes the number; the *decision*
and the *action* stay with the caller (see
[the caller-supplied window](#the-caller-supplied-window) below).

---

## Reading usage in-process

`Run` exposes the cumulative and footprint views side by side:

```python
from t0_agents.core.types import Usage   # value type — core.types, not the root

run = await agent.invoke("Summarise the attached transcript")

# Cumulative billing — the sum across every inference in this run.
total = run.usage()                       # RunUsage | None
print(f"${run.cost():.4f}  ({total.input_tokens} in / {total.output_tokens} out)")

# Footprint — the current prompt size, the compaction signal.
footprint = run.context_footprint()       # Usage | None
if footprint is not None:
    print(f"prompt is {footprint.input_tokens} tokens")
```

The full surface:

| Accessor | Returns | What it is |
|---|---|---|
| `run.usage()` | `RunUsage \| None` | Cumulative billing sum (`None` if no inference fired). |
| `run.cost()` | `float` | The `cost` field of `usage()`, or `0.0`. |
| `run.usages()` | `tuple[Usage, ...]` | Every inference's usage, in order — **including** cache-served zero turns. |
| `run.context_footprint()` | `Usage \| None` | The most recent inference with `input_tokens > 0`. |
| `run.context_remaining(window)` | `int \| None` | `window - footprint.input_tokens`, **unclamped**. |

`run.usages()` is the per-inference breakdown; `context_footprint()` is
the last *non-empty* entry of that list (see
[the caching caveat](#cache-tokens-cost_is_priced-and-the-caching-trap)).
All of these are **live mid-run** — `CostTrackingMiddleware` flushes them
after every inference, so you can read them from inside an
`events()` consumer or a multi-turn tool loop, not only after
`await run.result()`.

### The caller-supplied window

`context_remaining(window)` takes the window as an **argument** on
purpose.  The model's context size is *not knowable in-process*:
`AgentConfig` carries only the `function_name`, and TensorZero resolves
the concrete variant (and therefore the model) at inference time.  So the
library never guesses a window — the caller, who knows which model is
configured, passes it in:

```python
WINDOW = 128_000   # whatever the configured model's context window is

remaining = run.context_remaining(WINDOW)        # int | None
if remaining is not None and remaining < 8_000:
    # Close to full — compact or start a fresh episode before quality drops.
    ...
```

`context_remaining` is **not clamped**: an over-window prompt returns a
negative number, so the overflow magnitude reaches the caller rather than
being hidden behind a `0`.

---

## Across a conversation: `ConversationUsage`

A conversation is the many `invoke()`s that share one `episode_id`.  No
single `Run` spans them, so `ConversationUsage` accumulates across them —
summing **billing** while tracking the footprint as a **level**:

```python
from t0_agents import ConversationUsage     # root-exported

convo = ConversationUsage(episode_id)
for user_turn in conversation:
    run = agent.invoke(user_turn, episode_id=episode_id)
    await run.result()
    convo.record(run)

convo.total                       # RunUsage — billing summed across every turn
convo.context_footprint()         # Usage | None — the *latest* turn's prompt size
convo.remaining(WINDOW)           # int | None — window - footprint.input_tokens
convo.fraction_used(WINDOW)       # float | None — footprint.input_tokens / window
convo.runs_recorded               # int — how many Runs folded in
```

`.total` grows with every turn; `.context_footprint()` tracks the level of
the most recent turn — exactly the footprint-vs-billing split, lifted to
the conversation.

One deliberate subtlety: a `Run` that produced **no footprint** (a failure
before the first inference, or a whole-response cache hit) leaves the prior
level **intact** rather than blanking it.  A single empty turn must not
zero the live compaction signal mid-conversation.

`ConversationUsage` keeps only the running total and the latest level — no
per-Run history.  If you need the breakdown, keep your own list of `Run`s
and call `run.usage()` on each.

---

## Sub-agent delegation: billing rolls up, footprint does not

When an agent delegates to a sub-agent (`AgentConfig.sub_agents=[...]`),
the child runs its own `Run` with its own `CostTrackingMiddleware`.
t0-agents rolls the child's **billing** home automatically: the child's
`RunUsage` rides back on the delegated tool result and is folded into the
parent, so `parent_run.usage()` / `cost()` reflect the **whole delegation
tree**.

The **footprint is deliberately parent-only**.  A sub-agent runs in its
*own* context window — its prompt size says nothing about how full the
orchestrator's window is — so `parent_run.context_footprint()` reports the
orchestrator's own latest turn and never the child's.  Billing is additive
across windows; a footprint is not.

```python
parent = await agent.invoke("Research X, then draft a summary")  # delegates internally
parent.usage().input_tokens          # includes the sub-agent's tokens
parent.context_footprint()           # the orchestrator's own prompt only
```

No serve-side wiring is needed for this: any path that reads the
orchestrator's `run.usage()` already sees the delegated tokens.

---

## Cache tokens, `cost_is_priced`, and the caching trap

`Usage` (per-inference) and `RunUsage` (cumulative) both carry the cache
fields TensorZero reports:

```python
@dataclass(frozen=True, slots=True)
class Usage:                       # from t0_agents.core.types import Usage
    input_tokens: int
    output_tokens: int
    provider_cache_read_input_tokens: int | None = None
    provider_cache_write_input_tokens: int | None = None
    cost: float | None = None
    # .total_tokens -> input_tokens + output_tokens
    # .from_tz(tensorzero_usage) -> Usage
```

`RunUsage` mirrors these as summed `int`s (`None → 0`) plus one extra
field:

- **`cost_is_priced: bool`** — `True` only when TensorZero actually
  reported a price.  Without a `cost` block on the provider (see the
  [cost recipe](#real-cost-provider-pricing)), `cost` comes back
  `null` and t0-agents stores `0.0` — `cost_is_priced` is what
  distinguishes a *real* `$0.00` from *missing pricing*.  Don't read a
  `cost()` of `0.0` as free unless `usage().cost_is_priced` is `True`.

**The caching trap.**  When the gateway serves a *cached* model request,
the usage fields come back **`0`**.  A naive footprint would then read that
turn as "empty context" and wrongly conclude the window just emptied —
breaking the compaction decision.  So `context_footprint()` **skips
cache-hit zeros**: it returns the most recent inference with
`input_tokens > 0`, carrying the level forward over a zero turn.
`run.usages()` still exposes the zero entries for inspection.

> **Groq today:** Groq has no prompt caching, so `provider_cache_*` are
> `None`/`0` and `footprint.input_tokens` is exact for the current setup.
> The cache handling is forward-looking robustness for a provider (or a
> gateway `cache_options`) that does cache.

---

## The serving wire shape

The serving layer attaches an **optional, additive** `usage` block to
every path — `POST /chat`, the SSE `done` frame, and the WebSocket `done`
frame.  It is purely additive: every pre-existing response key and
`done`-frame key is byte-identical, so existing clients are unaffected.

The block is **top-level** (a sibling of the other response fields and of
the streaming frame's `result`), never nested inside `result`:

```jsonc
// POST /chat response, and the SSE / WS streaming "done" frame:
{
  "footprint":  { "input_tokens": 1840, "output_tokens": 95, "cost": 0.0,
                  "provider_cache_read_input_tokens": null,
                  "provider_cache_write_input_tokens": null } ,   // or null
  "cumulative": { "input_tokens": 1840, "output_tokens": 95, "cost": 0.0,
                  "provider_cache_read_input_tokens": 0,
                  "provider_cache_write_input_tokens": 0,
                  "cost_is_priced": false },
  // "session" is present ONLY on the WebSocket done frame (see below):
  "session":    { "total": { ...cumulative... }, "footprint": { ... }, "runs_recorded": 3 }
}
```

- **`footprint`** mirrors `Usage`; it is `null` when the turn produced no
  non-empty inference (a whole-response cache hit, or a failure before the
  first inference).  `cumulative` may still be present in that case.
- **`cumulative`** mirrors `RunUsage` — summed billing, including delegated
  sub-agents, with the cache fields and `cost_is_priced`.
- **The whole `usage` block is `null`** only when the run fired no
  inference at all.

### No `remaining` on the wire — the caller computes it

The serve block reports the **raw footprint** and nothing derived.  There
is no `remaining` / `fraction_used` field and **no `context_window`
request field**: the gateway can't know the model's window in-process and
nothing on the wire carries it, so a serving client computes "remaining"
against the window *it* knows — the same binding constraint as the
in-process [`context_remaining(window)`](#the-caller-supplied-window),
realized on the wire.

### `usage.session` — the WebSocket running total

Only the WebSocket `done` frame carries `usage.session`: a per-connection
running total (`total` billing, the latest `footprint`, `runs_recorded`)
that accumulates across the connection's turns.  It is keyed **per
`episode_id`**, so a client must thread one `episode_id` across its turns
to accumulate a single conversation; a new `episode_id` on the same socket
starts a fresh total, and distinct conversations on one socket never
conflate.  This is the channel that lets a streaming front-end watch the
footprint rise turn over turn and decide to compact.

The models (`UsageBlock`, `FootprintUsage`, `CumulativeUsage`,
`SessionUsage`) and the `build_usage_block(...)` builder live in
`t0_agents.serve.protocol`.

---

## Caveat: there is no after-the-fact usage in the SDK

The live in-process `Run` is the **only immediate** source of per-episode
usage.  TensorZero's `StoredInference` (what the query SDK returns) carries
**no token fields** — input/output messages, tags, and timings only.  So
there is no `t0.episode_usage(episode_id)` call to reach for after a
conversation ends; for historical or audit numbers you query ClickHouse
directly (the [history recipe](#historical-billing-clickhouse) below),
which is eventual-consistent, not a live signal.

---

## TensorZero setup (delegated facets)

Three facets are **delegated to TensorZero** rather than built into
t0-agents — they are gateway configuration (or SQL), not library code.
The recipes below are **copy-pasteable** and were validated against the
TensorZero docs in **June 2026**; the project's gateway pins
`tensorzero==2026.4.0`.  TOML keys can evolve — verify against the
[configuration reference](https://www.tensorzero.com/docs/gateway/configuration-reference)
for your gateway version.

For how these gateway-layer limits relate to t0-agents' per-invocation
guardrails (`max_cost`, `max_tool_calls`), see
[Guardrails vs. TensorZero rate limits](guardrails.md#guardrails-vs-tensorzero-rate-limits)
and [Production → Rate limits and quotas](production.md#rate-limits-and-quotas).

### Budget enforcement — `[[rate_limiting.rules]]` by tag

TensorZero has **native, tag-scoped** rate limiting on `tokens`,
`model_inferences`, and `cost`, per second / minute / hour / day (and
longer), persisted and shared across gateway instances.  Because
t0-agents already plumbs tags end-to-end
(`AgentConfig.tags` → `agent.invoke(tags=...)` → `t0.inference(tags=...)`),
per-user or per-session budgets are usable **today with zero code** — just
config:

```toml
# tensorzero.toml — cap one user's daily token spend.
[[rate_limiting.rules]]
priority = 1
tokens_per_day = 1_000_000
scope = [
    { tag_key = "user_id", tag_value = "u_12345" },
]

# A coarser org-wide ceiling, scoped by API key instead of a tag:
[[rate_limiting.rules]]
priority = 0
cost_per_day = 50.0
scope = [
    { api_key_public_id = "abcdef123456" },
]
```

Pass the matching tag at call time:

```python
await agent.invoke("…", tags={"user_id": "u_12345"})
```

Notes:
- The resource/window key is `RESOURCE_per_WINDOW` —
  `tokens_per_minute`, `model_inferences_per_hour`, `cost_per_day`, …
- **Token limits require `max_tokens`** on the request (set it on the
  variant) so the gateway can reserve the budget up front.
- **Cost limits** are estimated up front using `default_cost` (defaults to
  **$1.00** per inference) and reconciled once the real cost is known, so
  cost caps also need the [cost config](#real-cost-provider-pricing).
- The in-process `max_cost` guardrail complements this with an immediate
  in-loop stop within a single invocation;
  [`[[rate_limiting.rules]]`](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits)
  is the cross-request gateway layer.

### Real cost — provider pricing

Configured under `[models.*.providers.*]`.  `Usage.cost` is `null` unless
the provider declares pricing, and
**short-hand models** (e.g. `openai::gpt-5`) **cannot report cost at all** —
you must configure the model and provider explicitly.  Pricing is a `cost`
array of JSON-pointer rules into the provider's raw response:

```toml
# tensorzero.toml — price the project's Groq model (rates are illustrative;
# use your provider's published per-million-token rates).
[models.gpt_oss_20]
routing = ["groq"]

[models.gpt_oss_20.providers.groq]
type = "groq"
model_name = "openai/gpt-oss-20b"
cost = [
    { pointer = "/usage/prompt_tokens",     cost_per_million = 0.10, required = true },
    { pointer = "/usage/completion_tokens", cost_per_million = 0.50, required = true },
]
```

- `pointer` is a JSON Pointer into the provider's raw usage response
  (`/usage/prompt_tokens` and `/usage/completion_tokens` for
  OpenAI-compatible providers like Groq); use `cost_per_million` or
  `cost_per_unit`.
- If a `required = true` field is missing from a response, the gateway
  reports that inference's cost as `null` and logs a warning — which is
  exactly what `RunUsage.cost_is_priced` surfaces downstream.

See [Track usage and cost](https://www.tensorzero.com/docs/operations/track-usage-and-cost).

### Historical billing — ClickHouse

TensorZero persists every model call to the `ModelInference` table in
ClickHouse (database `tensorzero`).  That table has
`input_tokens`/`output_tokens` (`Nullable(UInt32)`) and `cost`
(`Nullable(Decimal64)`) but **no `episode_id`** — it joins up to
`ChatInference` / `JsonInference` (which carry `id` and `episode_id`) on
`inference_id`:

```sql
-- Per-episode token + cost totals for chat functions.
SELECT
    ci.episode_id,
    sum(mi.input_tokens)  AS input_tokens,
    sum(mi.output_tokens) AS output_tokens,
    sum(mi.cost)          AS cost
FROM ModelInference AS mi
INNER JOIN ChatInference AS ci ON mi.inference_id = ci.id
WHERE ci.episode_id = toUUID('00000000-0000-0000-0000-000000000000')
GROUP BY ci.episode_id;
```

For JSON functions, join `JsonInference` the same way (or `UNION ALL` the
two).  ClickHouse is **eventual-consistent** — these rows land after the
inference completes, so this is an audit/reporting source, not the live
compaction signal (that is `context_footprint()`).  See the
[data model](https://www.tensorzero.com/docs/gateway/data-model) for the
full schema.

---

## Related

- [Run](run.md) — the invocation handle whose `cost()` / `usage()` /
  `context_footprint()` this page documents.
- [Middleware](middleware.md) — the always-on `CostTrackingMiddleware`
  that populates usage on every inference.
- [Serving](serving.md) — `/chat`, SSE, and WebSocket; where the `usage`
  block rides on the wire.
- [Guardrails](guardrails.md#guardrails-vs-tensorzero-rate-limits) — the
  per-invocation budget layer (`max_cost`) vs. gateway rate limits.
- [API reference — `RunUsage`](api-reference.md#runusage-v10) — the field
  table for the cumulative type.
