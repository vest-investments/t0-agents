# Composition

`Composer` is the fourth v1.0 abstraction.  Nine built-in topology
builders take Agents (and other Composers) and stitch them into
industrial patterns — pipelines, fan-outs, races, ensembles,
scatter-gathers, triage routers, sagas, provider escalations,
state-machine breakers.

Every Composer satisfies the `Composable[Input, T]` protocol, so
they **nest uniformly** at arbitrary depth.
`sequential(race(a, b), conditional(routes={...}))` type-checks at
every layer; `composer.invoke(input)` returns the same `Run[T]`
handle as `agent.invoke(message)` so cancellation, audit, and
two-channel consumption work end-to-end on a composite.

v0.12 had one path to multi-agent work: declarative `AgentConfig.sub_agents=`
which lets the orchestrator's LLM choose which delegated agent to
invoke as a tool.  v1.0 keeps that path (unchanged) and adds the
**code-driven** path — explicit topology builders that compose at
construction time, without an LLM in the routing decision.  The
two paths are orthogonal; pick the one whose decision model fits
the problem.

> Companion reading: [`docs/run.md`](run.md) (the handle that every
> Composer produces), [`docs/multi-agent.md`](multi-agent.md) (the
> v0.12-era `sub_agents=` path),
> [`docs/middleware.md`](middleware.md) (each sub-Run runs its own
> middleware chain),
> [`docs/api-reference.md` — Composers](api-reference.md#composers-v10).

---

## The `Composable[Input, T]` protocol

```python
from typing import Protocol, runtime_checkable
from t0_agents import Run


@runtime_checkable
class Composable(Protocol[Input, T]):
    def invoke(self, input: Input, **kwargs: Any) -> Run[T]: ...
```

Both `Agent` and `Composer` satisfy this protocol — that's what lets
them nest.  An `Agent` accepts `user_message: str` and returns
`Run[Any]` (or `Run[T]` with a per-call `output=`); a Composer
accepts whatever input its topology defines and returns the same
`Run` type.

The narrow shape of `Composable.invoke` (the kwargs surface) differs
by builder; see [The narrow `Composer.invoke` surface](#the-narrow-composerinvoke-surface) below.

---

## The 9 topologies

| Builder | Shape | Typical use |
|---|---|---|
| `sequential(*nodes, chain=None)` | pipeline `A → B → C` | Multi-step processing where each step uses the previous result. |
| `parallel(*nodes, reducer=None, error_policy="fail_fast")` | fan-out + reducer | N agents on the same input; merge the answers. |
| `race(*nodes, cancel_others=True)` | first-completed wins | Latency-sensitive — first responder wins, losers cancel. |
| `consensus(*nodes, merge)` | N propose, merge decides | Voting / quorum for sensitive decisions. |
| `map_reduce(node, *, reducer, concurrency=None)` | scatter over an iterable | Same agent across many inputs, bounded concurrency. |
| `conditional(*, router, routes, default=None)` | sync routing on input | Triage to one branch out of N keyed by a sync function. |
| `saga(*steps)` | pipeline with compensation | Distributed transactions — undo on failure. |
| `retry_chain(*nodes, on_error=None)` | provider escalation | Primary fails → fall through to backup. |
| `circuit_breaker(node, *, threshold=5, reset_timeout=30.0)` | state-machine breaker | Stop calling a flaky downstream until it recovers. |

Each builder returns a `Composer` instance whose `.invoke(input)`
produces a `CompositeRun(Run[Any])`.

### `sequential` — pipeline

```python
from t0_agents import Agent, AgentConfig, sequential, ChatOutput

researcher = Agent(AgentConfig(name="research", function_name="agent"), t0=t0, tool_provider=mcp)
summariser = Agent(AgentConfig(name="summary", function_name="agent"), t0=t0, tool_provider=mcp)

pipeline = sequential(researcher, summariser)
run = pipeline.invoke("What's the latest on the Mars rover?")
final = await run.result()
print(final.value)
```

Each node's output feeds the next.  The default chain JSON-encodes
non-string values between steps; pass `chain=callable` to customise
the inter-step conversion.

### `parallel` — fan-out + reducer

```python
from t0_agents import parallel
from t0_agents.core.types import AgentResult


def majority_vote(results: list[AgentResult[str]]) -> str:
    """Pick the response that appears most often."""
    from collections import Counter
    counts = Counter(r.value for r in results)
    return counts.most_common(1)[0][0]


ensemble = parallel(model_a, model_b, model_c, reducer=majority_vote)
final = await ensemble.invoke("Is this a phishing email?").result()
```

`error_policy="fail_fast"` (default) cancels siblings the moment the
first sub-Run fails; the reducer sees `list[AgentResult[T]]`.
`error_policy="continue"` collects every settled state; the reducer
sees `list[AgentResult[T] | BaseException]` and decides how to
handle the failures (Q8 resolution).

### `race` — first-completed wins

```python
from t0_agents import race

fast = race(fast_provider, slow_provider, cancel_others=True)
result = await fast.invoke("Quick answer please").result()
```

The first sub-Run to finish wins; the others are cancelled under the
`cancel_timeout` budget (default 30s, per-instance override on the
factory or per-call on `.invoke`).  `cancel_others=False` lets the
losers continue best-effort.

### `consensus` — N propose, merge decides

```python
from t0_agents import consensus


def first_agreement(values: list[str]) -> str:
    if len(set(values)) == 1:
        return values[0]
    raise ValueError(f"No consensus: {values}")


jury = consensus(juror_a, juror_b, juror_c, merge=first_agreement)
verdict = await jury.invoke("Is this fraud?").result()
```

Reducer signature differs from `parallel`: `consensus.merge`
operates on `list[T]` (just the agreed values), not
`list[AgentResult[T]]`.  Sub-Run failures bubble before merge as
`ComposerError`.

### `map_reduce` — scatter over an iterable

```python
from t0_agents import map_reduce
from t0_agents.core.types import AgentResult


def concatenate_summaries(results: list[AgentResult[str]]) -> str:
    return "\n\n".join(r.value for r in results)


batch = map_reduce(summariser, reducer=concatenate_summaries, concurrency=5)
docs = [doc1_text, doc2_text, doc3_text, doc4_text]
combined = await batch.invoke(docs).result()
```

`concurrency=N` bounds active sub-Runs via `asyncio.Semaphore` —
useful when the underlying API or tool provider has its own quota.
`concurrency=None` is unbounded.

### `conditional` — sync routing

```python
from t0_agents import conditional


def route_by_category(query: dict) -> str:
    return query.get("category", "general")


triage = conditional(
    router=route_by_category,
    routes={
        "technical": tech_support,
        "billing": billing_agent,
        "general": general_agent,
    },
    default=general_agent,
)
result = await triage.invoke({"category": "technical", "text": "..."}).result()
```

The router is **synchronous** — `Callable[[Input], str]` — operating
on the raw input before any inference fires (Q9 resolution).  For
LLM-driven routing compose explicitly:
`sequential(classifier_agent, conditional(router=lambda r: r.value.label, routes=...))`.

Unknown route key + no `default` → `ComposerError(ValueError(...))`.
Router exceptions wrap in `ComposerError`.

### `saga` — sequential with compensation

```python
from t0_agents import saga


reserve_inventory = ...
charge_card = ...
ship_order = ...

release_inventory = ...
refund_card = ...
cancel_shipment = ...


checkout = saga(
    (reserve_inventory, release_inventory),
    (charge_card, refund_card),
    (ship_order, cancel_shipment),
)
final = await checkout.invoke({"order_id": "abc", ...}).result()
```

Each step is a `(action, compensate)` tuple.  On failure at step N,
compensations run in **reverse order**: `compensate(N-1)`,
`compensate(N-2)`, ..., `compensate(0)`.  Each compensation receives
the **same input** its paired action received — i.e., the chained
value flowing into `action_N`, not the original `saga.invoke(input)`
value (M4 D-4).

> **Compensations must be idempotent.**  A raising compensate logs
> WARN but does not block earlier compensations in the cascade — so
> a downstream compensate may end up being invoked twice (once
> during the failure cascade, once during the retry of the cascade).
> This is the classical saga constraint; t0-agents does not enforce
> idempotency at runtime, but the `saga` docstring spells it out
> (Q10 resolution).

### `retry_chain` — provider escalation

```python
from t0_agents import retry_chain
from tensorzero.types import TensorZeroError


def retry_on_5xx(exc: BaseException) -> bool:
    return isinstance(exc, TensorZeroError) and (exc.status_code or 500) >= 500


providers = retry_chain(primary_groq, backup_anthropic, fallback_openai, on_error=retry_on_5xx)
answer = await providers.invoke("Translate to French: hello").result()
```

Try nodes in order; if `on_error(exc) -> True` (or no filter
provided), fall through to the next.  If every node fails,
`retry_chain` **re-raises the original last exception un-wrapped** —
distinct from every other topology's `ComposerError` wrapping (M4 D-5).
This is so existing `except RateLimitError` / `except TensorZeroError`
blocks keep working as users escalate providers.

### `circuit_breaker` — state machine breaker

```python
from t0_agents import circuit_breaker

protected = circuit_breaker(
    flaky_provider,
    threshold=5,         # 5 consecutive failures → open
    reset_timeout=60.0,  # wait 60s before trying half-open
)
result = await protected.invoke("hello").result()
```

State machine: `closed` (normal) → `open` (refuses without invoking
inner) → `half_open` (one trial call) → back to `closed` (on success)
or `open` (on failure).  While `open`, every `.invoke(...)` raises
`T0AgentsFatalError` without touching the inner — protecting the
downstream provider from cascade load.

State lives on the Composer instance, so one breaker per protected
resource (one per provider, not per agent).

---

## The narrow `Composer.invoke` surface

`composer.invoke(input, ...)` accepts **three kwargs only**:

| Kwarg | Purpose |
|---|---|
| `eager: bool = True` | Same as `Agent.invoke(eager=)`; controls whether the drive task starts immediately. |
| `context: Mapping[str, Any] \| None = None` | User-supplied metadata copied onto `run.context`. |
| `cancel_timeout: float = DEFAULT_CANCEL_TIMEOUT` | Per-call override for the cancellation cascade timeout (default 30s). |

Agent-level kwargs (`tags`, `episode_id`, `system`,
`system_arguments`, `prior_messages`, `output`, `middleware`) are
**not accepted** — passing any raises `TypeError`.  This is
intentional (M4 D-3): per-topology semantics for those kwargs would
require ad-hoc decisions (`parallel(a, b, c)` with `tags={"tenant":
"acme"}` — do all three children inherit?  only the first?  none?),
and the user composes that explicitly today by configuring each
child's `AgentConfig.tags=`.

For per-tenant tagging across a topology, configure it on the child
agent configs:

```python
acme_classifier = Agent(
    AgentConfig(name="classifier", function_name="agent", tags={"tenant": "acme"}),
    ...,
)
acme_router = conditional(
    router=lambda r: r.value.label,
    routes={...},
)
acme_pipeline = sequential(acme_classifier, acme_router)
```

---

## Cancellation cascade

`await run.cancel()` propagates from the root `CompositeRun` down to
every sub-Run (asyncio task tree).  Each sub-Run gets up to
`cancel_timeout` seconds to clean up; overshoot logs a single
WARN on `t0_agents.compose.topology` per CompositeRun instance and
lets remaining children dangle best-effort.  The parent transitions
to `cancelled` regardless (Q3 resolution).

```python
run = topology.invoke(big_query)
await asyncio.sleep(2)
await run.cancel()

try:
    await run.result()
except RunCancelledError:
    print("Cancelled")
```

When to override `cancel_timeout`:

- **Tighter** (e.g. 5s) — when sub-Runs do quick I/O and you want
  fast UI feedback.
- **Looser** (e.g. 120s) — when sub-Runs hold transactions or large
  rollbacks that need time to unwind cleanly.

Per-instance via the factory: `parallel(a, b, c, cancel_timeout=60.0)`.
Per-call via the invoke: `topology.invoke(input, cancel_timeout=10.0)`
(per-call wins for that specific Run).

---

## Audit via `parent_episode_id`

Every Composer driver threads the parent's TensorZero `episode_id`
into each sub-Run as a TZ-native tag:
`tags={"t0_agents.parent_episode_id": "<parent>"}`.  The
ClickHouse query that reconstructs a topology's hierarchy is one
filter:

```sql
SELECT episode_id, function_name, ...
FROM tensorzero.inference
WHERE tags['t0_agents.parent_episode_id'] = '<root_episode_id>'
ORDER BY timestamp;
```

For composers-of-composers, the chain transitively threads — every
inference under any sub-Run, at any depth, carries the **root**
episode id, so a single `WHERE` reconstructs the entire run tree.

The `bind_parent_episode_id` ContextVar (in
`src/t0_agents/compose/orchestrator.py`) is the threading vehicle
for Composer-internal children; the TZ-native tag is the
attribution mechanism for Agent children.  Both reach the same
end state.

---

## The `CompositeRun` deferred APIs

`CompositeRun(Run[Any])` is the `Run` subclass produced by every
Composer.  It inherits the full `Run` surface (`.state`,
`.episode_id`, `.inference_ids`, `.result()`, `.events()`,
`.cancel()`, ...), but three methods are deliberately deferred
(M4 D-2):

- **`composite.feedback(...)`** — `NotImplementedError`.  A Composer
  has no single episode id of its own; routing `feedback(metric, value)`
  to "the right episode" requires per-topology decisions (the first
  sub-Run?  the last?  the failing one?).  Route feedback to
  individual sub-Runs explicitly until demand justifies a per-topology
  contract.
- **`composite.replay(...)`** — `NotImplementedError`.  Re-running a
  Composer is `composer.invoke(input)`, which is more explicit and
  idiomatic.
- **`composite.fork(...)`** — `NotImplementedError`.  "Extra user
  turn" doesn't map onto every topology (what does it mean for a
  `parallel` of 5 voters?).  Compose explicitly:
  `sequential(composer, follow_up_agent)`.

Each `NotImplementedError` message points at the right escape hatch.
A future minor can ship per-topology semantics if demand
justifies — recorded in M4 dev-log §2.

---

## Two paths to multi-agent: declarative vs explicit

t0-agents offers **two orthogonal** paths to multi-agent work.
Picking the right one is mostly a matter of *who decides the route*:

### Declarative (LLM-driven): `AgentConfig(sub_agents=[...])`

```python
config = AgentConfig(
    name="orchestrator",
    function_name="agent",
    sub_agents=[researcher, summariser, writer],
)
agent = Agent(config, t0=t0, tool_provider=mcp)
result = await agent.invoke("Research and summarise X").result()
```

The orchestrator's LLM **chooses** which sub-agent to invoke as a
tool, when, and how many times.  This is unchanged from v0.12 — the
`SubAgentMiddleware` materialised into the default chain runs the
orchestration.  Best when:

- The routing decision is inherently judgment-laden (analyses the
  user's intent, decides who to ask).
- The number of sub-agent invocations is dynamic.
- The orchestrator should explain its routing (LLM verbalises why
  it chose researcher over summariser).

See [`docs/multi-agent.md`](multi-agent.md) for the full sub-agent
delegation reference.

### Explicit (code-driven): `sequential(...)` / `parallel(...)` / etc.

```python
pipeline = sequential(
    researcher,
    conditional(
        router=lambda r: r.value.topic,
        routes={"academic": academic_summariser, "industry": industry_summariser},
    ),
    writer,
)
result = await pipeline.invoke("Research and summarise X").result()
```

The Python code **chooses** the topology at construction time.  This
is the new v1.0 path.  Best when:

- The routing decision is deterministic (categorise input, route
  accordingly).
- You need bounded concurrency, parent/child audit, or
  cancellation cascade.
- You want type-checking on the topology structure
  (`sequential(...).invoke(input)` type-checks; mismatches show up
  at construction time, not at runtime).

The two paths compose: a `Composer` can have an `Agent` child whose
own `AgentConfig.sub_agents=[...]` triggers further LLM-driven
delegation.  The `parent_episode_id` audit still threads through
both levels.

---

## Practical patterns

A few topologies from production use:

### Triage router for support tickets

```python
classifier_agent = Agent(AgentConfig(name="classifier", function_name="classify", output=StructuredOutput(TicketCategory)), t0=t0)

triage = sequential(
    classifier_agent,
    conditional(
        router=lambda result: result.value.category,
        routes={
            "technical": sequential(
                tech_researcher,
                consensus(senior_a, senior_b, senior_c, merge=majority_pick),
                writer,
            ),
            "billing": retry_chain(billing_groq, billing_claude),
            "general": race(general_a, general_b),
        },
        default=human_escalation,
    ),
)
```

### Provider escalation with breaker

```python
robust_provider = circuit_breaker(
    retry_chain(primary, backup, fallback, on_error=lambda e: not isinstance(e, AuthenticationError)),
    threshold=10,
    reset_timeout=120.0,
)
```

If `retry_chain` exhausts all three providers more than 10 times in
a row, the outer breaker opens and short-circuits — no more
inference calls until 2 minutes pass.

### Distributed transaction (saga)

```python
checkout = saga(
    (reserve_inventory, release_inventory),
    (charge_card, refund_card),
    (ship_order, cancel_shipment),
)
```

On `ship_order` failure: cascade runs `refund_card` then
`release_inventory`.  Each compensation receives the same input its
action received (the chained value flowing in), so encoding
uniqueness — order id, transaction handle, etc. — into the input is
the standard pattern.

---

## What Composers can't do

- **Mutate the input mid-topology.**  Each child sees the typed
  value flowing through `chain` (sequential) or the raw input
  (everything else).  Mutation belongs in middleware, not in
  topology shape.
- **Span gateways.**  Every sub-Run uses the gateway its `Agent`
  was constructed with.  For multi-gateway topologies, pass each
  agent a different gateway at construction.
- **Persist across runs.**  Composer instances are stateless except
  for `circuit_breaker` (state machine is per-instance).  Sub-Run
  state lives on each sub-Run's `Run` handle.

These constraints keep the topology composition predictable; the
extension points are middleware (cross-cutting concerns at the
sub-Run level) and `AgentConfig.tags=` (TZ-native metadata).

---

## Related

- [`docs/run.md`](run.md) — the handle every Composer produces.
- [`docs/multi-agent.md`](multi-agent.md) — the v0.12-era
  declarative `sub_agents=` path; orthogonal to the explicit
  topology builders.
- [`docs/middleware.md`](middleware.md) — each sub-Run runs its own
  middleware chain materialised from its own `AgentConfig`.
- [`docs/output-modes.md`](output-modes.md) — reducers in
  `parallel` / `consensus` operate on typed `AgentResult[T]`.
- [`docs/migration-v0.12-to-v1.0.md`](migration-v0.12-to-v1.0.md) —
  Composers are new in v1.0; opt-in adoption.
- [`docs/api-reference.md` — Composers](api-reference.md#composers-v10).
