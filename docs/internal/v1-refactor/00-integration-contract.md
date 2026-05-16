# t0-agents v1.0 — Integration Contract

> **Status:** Authoritative. Every milestone decision must trace back to this file.
> **Audience:** internal implementers and reviewers of the v1.0 refactor.
> **Scope:** the four-abstraction disruptive refactor (Run, Middleware, OutputMode, Composer).
> **Migration target:** two-tier — entry points (`agent.run`/`agent.stream`) deprecated in v1.0, removed in v1.1; legacy JSON config kwargs removed directly in v1.0 with documented one-line migration.  Read-side `AgentResult` properties preserved.  See §2.4.

---

## §1 — Why this contract exists

t0-agents v0.12.0 reached the end of a successful incremental roadmap (agentic + structured-outputs).  Two structural pain points remain:

1. **Chat / JSON asymmetry** — `function_kind` literal forces 4 JSON-only fields in `AgentConfig`, 4 in `AgentResult`, and ~80 lines of cross-field validation.
2. **Hardcoded cross-cutting concerns** — memory, guardrails, cost-tracking, JSON retry, sub-agents are all special-cased inline in `core/agent.py` (~1413 LoC).  Custom rate-limiters, PII redactors, observability hooks, recovery policies are impossible without forking.

Two structural opportunities appear at scale:

3. **No first-class invocation handle** — `agent.run()` returns the terminal `AgentResult`; there is no object representing the live invocation.  Replay, fork, lazy queueing, two-channel consumption (UI stream + business await) are all blocked.
4. **No topology primitive** — multi-agent composition is limited to flat `sub_agents` lists.  Industrial/military patterns (pipeline, race, consensus, saga, circuit-breaker, bulkhead, tenant isolation) require ad-hoc Python glue.

This contract specifies **four abstractions** that resolve all four points orthogonally, under a **strict additive-compatibility guarantee**.

---

## §2 — Invariants

These are non-negotiable.  Every milestone gate verifies them.

### §2.1 — `mcp/` standalone

`src/t0_agents/mcp/` must remain importable without any of `core/`, `memory/`, `guardrails/`, `serve/`, `stream/`, `compose/`, `cli/`, `testing/`.  `from t0_agents.mcp import run_agent` works with the base install.  `mcp/loop.py:run_agent` is the **terminal** called by the innermost layer of every pipeline; its public signature is **frozen**.  New optional kwargs allowed only with `None` defaults that preserve today's behaviour.

Verification per milestone: `git diff src/t0_agents/mcp/` shows zero changes.

### §2.2 — TensorZero-first

For every new abstraction, map onto existing TensorZero primitives:
- Functions / variants — `tensorzero.toml`, not dynamic prompts.
- Correlation — `episode_id`, not custom IDs.
- Namespace / tags — TZ tags, not custom headers.
- Runtime tools — `additional_tools`, not custom registries.
- Outcome capture — `t0.feedback(...)`, not a parallel metric stream.
- Streaming — `stream=True`, not Python-side queueing.

Inventing parallel systems duplicates effort and confuses operators.

### §2.3 — Chat byte-identity

The chat path of `Agent.run()`/`Agent.stream()` (via deprecated shims) and `mcp.run_agent` must remain functionally equivalent to v0.12.0 — same `episode_id`-shaped logging, same `inference_ids` count, same `turns`, same response shape.  Stdout is byte-identical; stderr may include `DeprecationWarning` emissions from the shims (starting M1).

Every milestone gate runs the 17 chat-mode examples (`examples/00..16, 18`) and verifies byte-identical stdout.  The JSON-mode example (`examples/17_structured/`) is migrated in M3 to use the new `output=JsonOutput(...)` shape; from M3 onward it runs against the new API.

Existing tests under `tests/test_mcp/`, `tests/test_core/`, `tests/test_stream/`, `tests/test_serve/`, `tests/test_compose/`, `tests/test_guardrails/`, `tests/test_memory/`, `tests/test_cli/`, `tests/test_testing/` must stay green unchanged, except when a test is intentionally updated for new behaviour (recorded in the dev-log).

### §2.4 — Migration policy (two-tier)

t0-agents v1.0 introduces ONE canonical entry point and ONE canonical config surface.  The v0.12 surface migrates on two distinct timelines, chosen to minimise disruption to high-frequency callers while removing legacy clutter aggressively where the migration is cheap.

**Tier 1 — Deprecated in v1.0, removed in v1.1 (entry points).**  Every inference goes through these, so the migration touches every callsite.

| v0.12 API           | v1.0 behaviour                         | v1.1 behaviour |
|---------------------|----------------------------------------|----------------|
| `agent.run(msg, **kw)`    | works; emits `DeprecationWarning` once per process | removed |
| `agent.stream(msg, **kw)` | works; emits `DeprecationWarning` once per process | removed |

Migration is one-liner per callsite:
- `await agent.run(msg)`     → `await agent.invoke(msg)`
- `async for ev in agent.stream(msg):` → `async for ev in agent.invoke(msg).events():`

**Tier 2 — Removed directly in v1.0 (config legacy kwargs).**  These touch only JSON-mode users; migration is bounded and cheap.

| v0.12 kwarg                 | v1.0 status |
|-----------------------------|-------------|
| `AgentConfig.function_kind` | **removed** |
| `AgentConfig.json_mode`     | **removed** |
| `AgentConfig.output_schema` | **removed** |
| `AgentConfig.permissive_json_parsing` | **removed** |
| `AgentConfig.max_json_retries` | **removed** |

Replacement: `AgentConfig(output=JsonOutput(schema=S, mode="strict", permissive=True, max_retries=2))` (or `StructuredOutput`, `ChatOutput`, etc.).  Passing any removed kwarg to `AgentConfig.__init__` raises `TypeError` at construction (standard dataclass behaviour).  Migration guide (M5) shows the explicit before/after for the only canonical example (`examples/17_structured/`).

**Tier 0 — Unchanged forever (everything else).**  These are guaranteed-stable APIs.

| API v0.12.0                                                 | v1.0 status         |
|-------------------------------------------------------------|---------------------|
| `run_agent(...)` (`t0_agents.mcp`)                          | unchanged terminal  |
| `run_structured_inference(...)`                             | unchanged helper    |
| `permissive_parse(...)`                                     | unchanged helper    |
| `AgentConfig(memory=, guardrails=, sub_agents=, ...)`       | unchanged kwargs; materialise middleware chain internally |
| `AgentResult.response` / `.parsed` / `.parse_strategy` / `.json_parse_attempts` / `.function_kind` | `@property` aliases over the new shape |
| `Guardrail`, `MemoryProvider`, `ToolProvider` protocols     | unchanged           |
| `agent_as_tool(...)`                                        | unchanged           |

**Why two tiers and not one.**  A single shared policy (e.g., "deprecate everything for one minor") would either drag the legacy JSON kwargs for longer than needed (Tier 2 is cheap to migrate) or hard-break high-frequency callsites with no grace period (Tier 1 affects every inference).  Two tiers keep the migration cost proportional to how invasive the change is.

### §2.5 — Async-first, mypy strict

All public APIs are `async def`.  No sync wrappers.  The codebase passes `mypy --strict` — every type explicit, no accidental `Any`.

### §2.6 — Opt-in per layer

The base install ships everything in `core/`, `mcp/`, `memory/InMemoryMemory`, `guardrails/`, `stream/`, `compose/`, `testing/`.  Heavier deps (SQLite, FastAPI, Uvicorn, Typer, Rich, Redis, OpenTelemetry) live behind extras in `pyproject.toml` with `ImportError` guards.

New industrial middleware introduced in M5 (Redis rate-limit, OTel observability, Datadog metrics) go behind appropriate extras.

---

## §3 — The four abstractions

### §3.1 — `Run[T]` — invocation handle

```python
class Run(Generic[T]):
    input: RunInput                     # immutable snapshot
    context: dict[str, Any]             # user metadata
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    episode_id: str | None              # populated mid-run
    inference_ids: tuple[str, ...]

    async def result(self) -> AgentResult[T]
    async def events(self, *, replay: bool = False) -> AsyncIterator[AgentEvent]
    async def stream_text(self) -> AsyncIterator[str]
    def __await__(self) -> Generator[Any, None, AgentResult[T]]

    async def cancel(self) -> None
    async def feedback(self, name: str, value: float | str | bool, *,
                       target: Literal["episode", "inference"] = "episode") -> None

    async def replay(self, *, agent: "Agent | None" = None,
                     variant: str | None = None,
                     tags: dict[str, str] | None = None) -> "Run"
    async def fork(self, extra_message: str, *,
                   agent: "Agent | None" = None) -> "Run"

    def transcript(self) -> list[Message]
    def cost(self) -> float
    def usage(self) -> Usage
```

**Lifecycle.** Eager by default; `eager=False` opt-in for queueing.  `weakref` to Agent (no gateway lifetime extension).  Events buffered + replayable.  Completes at `DoneEvent`; cancellation transitions to `cancelled` and fires guardrail `after_run(result=None)`.

**Entry point.**
```python
class Agent:
    def invoke(self, user_message: str, *,
               episode_id: str | None = None,
               tags: dict[str, str] | None = None,
               system_arguments: dict[str, Any] | None = None,
               system: str | None = None,
               # M1-deprecated per-call JSON overrides (removed in M3):
               output_schema: Mapping[str, Any] | None = None,
               json_mode: Literal["off", "on", "strict", "tool"] | None = None,
               max_json_retries: int | None = None,
               # New in M1 — used by Run.fork:
               prior_messages: Sequence[Mapping[str, Any]] | None = None,
               # M2 / M3 placeholders (accepted, threaded, not yet used):
               middleware: list[Middleware] | None = None,
               output: OutputMode[U] | None = None,
               # M1 lifecycle:
               eager: bool = True,
               context: dict[str, Any] | None = None,
              ) -> Run[T]
```

`agent.run()` and `agent.stream()` become deprecated shims (Tier 1) emitting `DeprecationWarning` once per source line; the shim of `agent.run()` unwraps `RunFailedError` so existing exception-catching callers continue to see the original exception type.

**M1 amendment (recorded in `logs/milestone-1-dev-log.md` §2):** the per-call legacy JSON kwargs (`output_schema`, `json_mode`, `max_json_retries`) were not in the original contract draft but are required at M1 to preserve Tier 1 byte-identity — v0.12 `Agent.run` accepted them as per-call overrides.  They are deprecated in M1 (no separate `DeprecationWarning` because the shim's warning is enough) and removed in M3 alongside the matching `AgentConfig` fields under Tier 2.  Direct v1.0 callers should treat them as transitional.

`prior_messages` is also new in M1.  It surfaces the `mcp.run_agent` kwarg of the same name so `Run.fork(extra_message)` can thread a recorded transcript into a follow-up invocation.  Explicit caller `prior_messages` wins over the memory-strategy-derived prior_messages when both are present.  This is a permanent kwarg and survives M3.

### §3.2 — `Middleware` — execution chain

```python
Next = Callable[["RunContext"], Awaitable[AgentResult]]

class Middleware(Protocol):
    name: str
    async def __call__(self, ctx: RunContext, next_: Next) -> AgentResult: ...
```

**RunContext** holds: `config`, `user_message`, `episode_id`, `tags`, `system`, `system_arguments`, `output`, `t0`, `tool_provider`, `on_inference`, `prior_messages`, `extras`, `result`, `run`.

**Built-ins (M2):** `MemoryMiddleware`, `GuardrailsMiddleware`, `CostTrackingMiddleware`, `SubAgentMiddleware`, `JsonRetryMiddleware`, `ObservabilityMiddleware`.

**Industrial (M5):** `BulkheadMiddleware`, `CircuitBreakerMiddleware`, `TenantIsolationMiddleware`, `PiiRedactorMiddleware`, `RedisRateLimitMiddleware`.

**Default chain materialisation.**  `AgentConfig.__post_init__` builds the default chain from existing kwargs (`memory=`, `guardrails=`, `sub_agents=`).  `middleware=[]` is an explicit "no middleware" override (semantically distinct from `middleware=None`, which means "use default").

**Order invariant.**  `SubAgentMiddleware` wraps tool_provider *inside* `GuardrailsMiddleware` (guardrails see delegations).  Tests enforce this.

### §3.3 — `OutputMode[T]` — output polymorphism

```python
class OutputMode(Protocol[T_co]):
    expects_tool_loop: bool
    supports_streaming: bool
    def shape_inference(self, req: InferenceRequest) -> InferenceRequest: ...
    async def materialize(self, response, attempts) -> AgentResult[T_co]: ...
    async def recover(self, failure, attempts) -> Recovery: ...
```

**Built-ins:**
- `ChatOutput` (T=str, expects_tool_loop=True, supports_streaming=True)
- `JsonOutput(schema, mode, permissive, retries)` (T=dict)
- `StructuredOutput[M](model)` (T=M, Pydantic)
- `ToolCallOutput(tool_choice)` (T=list[ToolCall], router pattern)
- `StreamingOutput(base)` (placeholder; activates when TZ adds mid-stream `parsed`)

**`AgentResult[T]` generic** with `value: T`, `diagnostics: OutputDiagnostics | None`.  Properties `response`, `parsed`, `parse_strategy`, `json_parse_attempts`, `function_kind` preserve v0.12.0 byte-identity.

**Recovery types.**
- `Recovery.retry(corrective_hint)` — re-emit inference with hint prepended.
- `Recovery.accept(value)` — bypass parsing, take this value.
- `Recovery.fail()` — surface the failure to caller.

### §3.4 — `Composer` — topology primitive

```python
class Composable(Protocol[T]):
    def invoke(self, input: Any, **kwargs: Any) -> Run[T]: ...

class Composer(Composable[T], Protocol):
    children: tuple[Composable, ...]
```

**Built-in topologies (9):**
- `sequential(*nodes, chain=None)` — pipeline.
- `parallel(*nodes, reducer=None, error_policy="fail_fast")` — fan-out + reducer.
- `race(*nodes, cancel_others=True)` — first-completed wins.
- `consensus(*nodes, merge=...)` — N propose, merge decides.
- `map_reduce(node, *, reducer, concurrency=None)` — scatter-gather.
- `conditional(*, router, routes, default=None)` — triage routing.
- `saga(*steps)` — sequential with compensation on failure.
- `retry_chain(*nodes, on_error=None)` — provider escalation.
- `circuit_breaker(node, *, threshold, reset_timeout)` — state machine.

**Uniformly composable.**  `Composer.invoke()` returns `Run[T]` identical to `Agent.invoke()`.  Composers of composers nest naturally.

**Audit.**  Each sub-Run carries `parent_episode_id` (already supported by v0.12 via `bind_parent_episode_id`).  ClickHouse query `WHERE parent_episode_id = '<root>'` reconstructs the hierarchy.

**Cancellation.**  `await root_run.cancel()` propagates to all sub-Runs (asyncio task tree).

---

## §4 — Module additions (by milestone)

| Milestone | New files | Modified files |
|-----------|-----------|----------------|
| M1 | `core/run.py`, `core/_runtime.py` | `core/agent.py`, `__init__.py` |
| M2 | `core/middleware/{__init__,base,builtins}.py` | `core/agent.py`, `core/config.py`, `__init__.py` |
| M3 | `core/output.py`, `core/outputs/{chat,json,structured,tool_call,streaming}.py` | `core/types.py`, `core/config.py`, `core/agent.py`, `core/structured.py`, `core/middleware/builtins.py`, `__init__.py` |
| M4 | `compose/{topology,builtins,_runtime}.py` | `compose/__init__.py`, `__init__.py` |
| M5 | `middleware/industrial.py`, 4 examples (`examples/19..22`), 4 docs (`docs/{run,middleware,output-modes,composition}.md`), `docs/migration-v0.12-to-v1.0.md` | `README.md`, `docs/agent.md`, `docs/index.md`, `CHANGELOG.md`, `pyproject.toml` |

`src/t0_agents/mcp/loop.py` and `src/t0_agents/mcp/adapter.py` are **untouchable** across all milestones.

---

## §5 — Error model

### §5.1 — Existing exceptions (preserved)

- `T0AgentsError` (base)
- `T0AgentsFatalError` (abort the loop)
- `ConfigurationError` (raise at construction or eager-validation time)
- `GuardrailBlockedError` (T0AgentsFatalError subclass)
- `MaxTurnsExceededError`
- `SubAgentError`

### §5.2 — New exceptions (additive)

- `RunCancelledError` (M1) — raised from `await run.result()` (or from a `run.events()` iterator) after `run.cancel()`.
- `RunFailedError(cause: Exception)` (M1) — **defensive fallback**: raised by `await run.result()` only when the drive ended in `"failed"` state without a captured exception (shouldn't normally happen).  In the normal failure path, `result()` raises the **original** exception (`MaxTurnsExceededError`, `GuardrailBlockedError`, `ValueError`, ...), matching `events()` semantics and v0.12 idioms.  See `logs/milestone-1-dev-log.md` §2 D-3 for the amendment rationale.
- `ComposerError(node, cause)` (M4) — wraps sub-Run failures with topology context.

### §5.3 — Error policies

- Tool execution errors → string `"ERROR: ..."` in `ToolResult` (unchanged from v0.12).
- Inference recoverable errors → recovery message + retry (unchanged).
- Guardrail / max_turns / configuration → propagate as `T0AgentsFatalError`.
- Middleware unhandled exceptions → propagate as `T0AgentsError` (caller decides).
- Composer error_policy: `"fail_fast"` (default) propagates first failure; `"continue"` collects failures and lets reducer decide.

---

## §6 — Migration guarantee

**Promise:** every v0.12.0 codebase upgrades to v1.0 with bounded, documented migration steps.  Tier 1 callers (`agent.run`, `agent.stream`) see `DeprecationWarning` but keep working until v1.1.  Tier 2 callers (`function_kind=` etc.) get a one-line migration documented in the migration guide.  Everything else is unchanged.

**Enforcement (per milestone gate):**

1. **Tier 1 (deprecated entry points).**
   - `agent.run(...)` still works in v1.0; emits `DeprecationWarning` exactly once per process (deduplicated by stacklevel + filter), pointing at `agent.invoke(...)`.
   - `agent.stream(...)` likewise emits `DeprecationWarning` pointing at `agent.invoke(...).events()`.
   - 17 of the 18 existing examples use `agent.run()` / `agent.stream()` — these continue to run; output is byte-identical modulo the warning to stderr.

2. **Tier 2 (removed config kwargs).**
   - `AgentConfig.__init__` no longer accepts `function_kind`, `json_mode`, `output_schema`, `permissive_json_parsing`, `max_json_retries`.  Passing any raises `TypeError`.
   - The canonical JSON example (`examples/17_structured/`) is migrated in M3 as part of the milestone deliverable.  After migration: 18 examples run; 17 use `invoke()` indirectly via shims, 1 uses the new `output=JsonOutput(...)` shape.
   - Tier 2 migration is "one line per agent" — call sites that build `AgentConfig` with JSON-mode kwargs swap them for `output=JsonOutput(schema=S, mode="strict", ...)`.

3. **Tier 0 (unchanged).**
   - All pre-existing tests under `tests/test_mcp/`, `tests/test_core/`, `tests/test_stream/`, `tests/test_serve/`, `tests/test_compose/`, `tests/test_guardrails/`, `tests/test_memory/`, `tests/test_cli/`, `tests/test_testing/` stay green unchanged, **except**:
     - `tests/test_core/` that exercise `function_kind=` etc. are updated to use `output=` (same coverage, new signatures).
     - Tests that explicitly check absence of `DeprecationWarning` need to either be updated to assert presence, or are split into "v0.12 shim" vs "v1.0 canonical" suites.
   - `AgentResult.response`, `.parsed`, `.parse_strategy`, `.json_parse_attempts`, `.function_kind` continue to read identically via `@property`.
   - Every symbol `from t0_agents import ...` exported in v0.12 (excluding the 5 removed config kwargs which weren't standalone exports) continues to import.

**Exception:** if a milestone surfaces an unavoidable break beyond Tier 1 and Tier 2 (e.g., SDK behaviour that changes upstream), it is documented in the dev-log §2 and surfaced to the user before continuing.

**v1.1 milestone (out of scope for the current roadmap):**
- Remove `agent.run()` and `agent.stream()`.
- Remove the deprecation shims.
- Bump major version constraint if any downstream relies on the shims at runtime.

---

## §7 — Open items at planning time

(Tracked in `02-open-questions.md`.  Initial seed reproduced here for context.)

- Q1: `Run` eager vs lazy default — recommended eager.
- Q2: Events buffer size — bounded queue + WARN log on overflow; threshold TBD.
- Q3: `Composer.cancel()` timeout configurable? — default 30s, kwarg in `circuit_breaker` / `saga`.
- Q4: `BulkheadMiddleware` lives in base or extra? — recommended base.
- Q5: `RedisRateLimitMiddleware` extra name — recommend `[ratelimit]` (lives with `redis` dep).
- Q6: `ObservabilityMiddleware` — bundled `[otel]` extra or standalone middleware that accepts any tracer?
- Q7: Run.replay should preserve `episode_id` or fork a new one? — recommend new episode (TZ episode_id is correlation, not state).
- Q8: `parallel` reducer signature — `Callable[[list[AgentResult]], T]` vs `Callable[[list[T]], T]`?
- Q9: `conditional` router signature — `Callable[[Input], str]` vs `Callable[[Run], str]` (wait for first sub-Run result)?
- Q10: `saga` compensation must be idempotent? — document yes; enforce by convention.

Each open question gets `Status: open | resolved` updates as milestones answer them.

---

## §8 — How this contract is read

- The **command** `/v1-milestone <N>` reads this file FIRST, before the milestone spec.
- A milestone spec **never overrides** this contract.  If a spec conflicts, the spec is wrong and must be amended.
- If implementation reveals that this contract is wrong (SDK behaviour, missing constraint), the dev-log §2 documents the deviation AND this contract is amended in the same commit.  Never let the contract drift silently.

---

## §9 — Reference: file layout target

```
src/t0_agents/
├── __init__.py                           # public API (~40 exports after M5)
├── mcp/                                   # INVARIANT — do not touch
│   ├── loop.py                            # run_agent terminal
│   └── adapter.py                         # convert_tools, serialize_tool_result, build_tool_result
├── core/
│   ├── agent.py                           # Agent class (~250 LoC after M2)
│   ├── config.py                          # AgentConfig (~150 LoC after M3)
│   ├── types.py                           # AgentResult[T], protocols
│   ├── exceptions.py                      # error hierarchy
│   ├── structured.py                      # run_structured_inference, permissive_parse (unchanged public)
│   ├── elicitation_forwarder.py           # unchanged
│   ├── run.py                             # NEW (M1) — Run[T], RunInput, RunState
│   ├── _runtime.py                        # NEW (M1) — async driver
│   ├── output.py                          # NEW (M3) — OutputMode protocol + types
│   ├── outputs/                           # NEW (M3)
│   │   ├── chat.py
│   │   ├── json.py
│   │   ├── structured.py
│   │   ├── tool_call.py
│   │   └── streaming.py
│   └── middleware/                        # NEW (M2)
│       ├── __init__.py
│       ├── base.py                        # Middleware proto + RunContext + build_pipeline
│       └── builtins.py
├── memory/                                # unchanged (MemoryProvider impls)
├── guardrails/                            # unchanged (GuardedToolProvider, builtins)
├── stream/                                # M1 adapts runner to emit to Run buffer
├── compose/
│   ├── __init__.py                        # exports topology builders + agent_as_tool
│   ├── agent_as_tool.py                   # unchanged (existing primitive)
│   ├── topology.py                        # NEW (M4)
│   ├── builtins.py                        # NEW (M4) — 9 topology functions
│   └── _runtime.py                        # NEW (M4)
├── middleware/                            # NEW (M5)
│   └── industrial.py                      # Bulkhead, CircuitBreaker, TenantIsolation, PiiRedactor, RedisRateLimit
├── serve/                                 # unchanged
├── cli/                                   # unchanged
└── testing/                               # unchanged
```

Public surface grows by ~25 symbols across the five milestones.  No symbol is removed.
