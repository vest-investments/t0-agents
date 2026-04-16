# 00 — Integration Contract

> Stable, cross-phase architectural contract for the `t0-agents` Agent
> abstraction work. Every phase spec in [`phases/`](phases/) restates the
> invariants in its own "Constraints" section. If you change something
> here, mark it as a contract amendment and update each phase spec that
> touches the affected surface.

**Related documents**

- [`01-implementation-plan.md`](01-implementation-plan.md) — phase ordering,
  milestones, cross-cutting checklists.
- [`02-open-questions.md`](02-open-questions.md) — questions whose answers
  may force a contract amendment.
- [`problemas.md`](problemas.md) — the original problem statement.
- [`developer-guide.md`](developer-guide.md) — the internal guide for the
  code *that exists today* (`mcp/` submodule only).
- [`tensorzero-developer-guide.md`](tensorzero-developer-guide.md) —
  TensorZero concepts this contract depends on.
- [`05-estructura-proyecto-t0-agents.md`](05-estructura-proyecto-t0-agents.md)
  — target project layout.

---

## 1. Purpose and audience

This document is the **load-bearing contract** for every new submodule
that lands as part of the Agent abstraction work. It defines:

- What stays stable across phases, so contributors don't relitigate
  decisions between A and H.
- The public `Agent` API surface (shape, lifecycle, composition).
- How `Agent` composes the existing `mcp/` submodule without forcing
  callers to migrate.
- The error model.
- What is and isn't the caller's responsibility.

Audience: contributors writing code under `src/t0_agents/`, reviewers
of PRs that touch `core/`, and anyone extending the framework with a
new submodule (e.g. a new memory backend or a new guardrail).

This is **not** a public API reference — that lives under `docs/`. Think
of this file as the internal design constitution; the public reference
restates the subset that end users need.

---

## 2. Architectural invariants

The six invariants below are non-negotiable without an explicit contract
amendment and a cross-phase review.

### 2.1 `mcp/` stays standalone

A developer who only wants the low-level bridge must be able to do:

```python
from t0_agents.mcp import run_agent, convert_tools
```

and be done. No higher-level machinery is imported, no extras need to be
installed, no `Agent` class is constructed.

**Why.** The user stated this explicitly: *"Quiero que el modulo mcp
siga funcionando stand-alone ya que la intención de t0-agents es que
puedas instalar lo que necesites para tener un nivel de manipulación y
granularidad cuando los developers lo implementen en sus proyectos."*

**Consequences.**

- `Agent.run()` is implemented by **delegating to `run_agent`** — not the
  other way around. `run_agent`'s signature is frozen (§6).
- `mcp/__init__.py` does not depend on anything under `core/` or any
  other new submodule.
- Tests under `tests/test_mcp/` continue to pass unchanged after every
  phase.

### 2.2 `core/` is the single shared import target

Every other submodule (`memory/`, `guardrails/`, `serve/`, `stream/`,
`compose/`, `testing/`, `cli/`) may import from `core/` and `mcp/`.
Lateral imports between sibling submodules are forbidden.

```
               ┌──────────┐
               │  core/   │   types, protocols, Agent, config, exceptions
               └────┬─────┘
                    │
      ┌────────┬────┴────┬──────────┬──────────┐
      │        │         │          │          │
   mcp/    memory/   guardrails/   stream/    testing/
      │        │         │          │          │
      └────────┴──┬──────┴──────────┘          │
                  │                            │
               compose/                       cli/
                  │
                serve/
```

Shape of the dependency graph: arrows go down. Each submodule depends on
`core/`; `compose/` additionally depends on `mcp/`; `serve/` depends on
`compose/`, `stream/`, `memory/`, `guardrails/`; `cli/` depends on
`stream/` and `core/`. No submodule imports a sibling at the same level.

**Why.** Prevents hidden coupling as the surface grows, keeps extras
genuinely opt-in, and makes the public API predictable.

### 2.3 Extras are opt-in

Each new submodule ships behind its own extra in `pyproject.toml`:

| Extra | Pulls in | Used by |
| --- | --- | --- |
| (no extra) | `tensorzero`, `fastmcp` | `core/`, `mcp/` |
| `memory-sqlite` | `aiosqlite` | `memory/sqlite.py` |
| `memory-redis` | `redis[hiredis]` | `memory/redis.py` (future) |
| `serve` | `fastapi`, `uvicorn[standard]` | `serve/` |
| `cli` | `typer`, `rich` | `cli/`, examples |
| `all` | aggregates all of the above | convenience |

**Why.** A user deploying only the HTTP server shouldn't pay for
`aiosqlite` or `rich`. Keeps core lean and deterministic.

**Consequence.** Every submodule does *try*/*except ImportError* at the
top if it imports a third-party library, raising a clear
`T0AgentsError` that names the missing extra.

### 2.4 TensorZero-first

Features must map cleanly onto existing TensorZero capabilities. When
TensorZero already provides a primitive, the framework wraps it rather
than reinventing it.

| Framework need | TensorZero primitive used |
| --- | --- |
| System prompts | `[functions.<f>.variants.<v>]` templates + `input.system.arguments` |
| Model selection | `[models.*]` + `[functions.*.variants.*.model]` |
| A/B testing | `[functions.*.experimentation]` |
| Fallback / retry | `routing = [...]` + variant `retries` |
| Correlation across turns | `episode_id` (captured from first response) |
| Per-deployment isolation | `namespace` (experimentation) + `tags` (observability) |
| Dynamic tools | `additional_tools` per inference |
| Outcome tracking | `[metrics.*]` + `t0.feedback(...)` |
| Token/event streaming | `t0.inference(stream=True)` |

**Memory injection path** (resolved via T0 Q4 — both supported):

- **System template path (recommended for production)** — variants
  declare `templates.system.path` + `schemas.system.path` with a
  `memory_context` argument; framework calls
  `t0.inference(input={"system": {"arguments": {"memory_context":
  "..."}}}, ...)`. Memory stays out of the conversation history.
- **Prepended message path (default for variants without templates)** —
  framework prepends a `RawText` content block (the SDK's
  `tensorzero.RawText`, mapping to `tensorzero::raw_text`) to the
  conversation. Bypasses templating; works with any function.

Phase B exposes both via `AgentConfig.memory_strategy: Literal["auto",
"system_template", "prepended_message"]`. `"auto"` picks
`"system_template"` when the variant declares a `memory_context`
argument; otherwise falls back to `"prepended_message"`. See
[`phases/phase-b-memory.md`](phases/phase-b-memory.md) for the runtime
detection rule.

**Anti-patterns** (explicitly forbidden):

- A separate prompt manager that duplicates variant templates.
- A separate provider abstraction (routing is T0's job).
- A separate experimentation system (A/B is T0's job).
- An internal RAG engine (memory is a Protocol; let users bring their
  own retriever).

### 2.5 Elicitation is the only human-in-the-loop primitive

Per the user's note on Problem 6: *"Ya tenemos Elicitations de fastmcp
y ya hay ejemplos en t0-agents, no vamos a usar Approval"*. No separate
Approval system is built.

**Consequences.**

- `guardrails/` (Phase C) is a thin layer of abstractions on top of
  `ctx.elicit()` — policy declaration, approval-gate helpers, response
  normalisers.
- The serving phase (Phase D) pipes elicitation through the browser;
  the TUI phase (Phase F) pipes it through Rich. Both use the same
  FastMCP `elicitation_handler` hook.
- No new protocol messages, no hidden "approval request" flow outside
  of MCP elicitation.

### 2.6 Async-first

Every new public API is `async def`. Sync convenience wrappers are
allowed only when they genuinely simplify the caller's life (e.g. a
Typer command that wraps an `asyncio.run`), and must be named distinctly
(e.g. `Agent.run_sync`).

**Why.** Matches `run_agent`, `execute_tool_calls`, FastMCP `Client`
usage, TensorZero SDK usage. Avoids bimodal APIs that are each half as
useful.

---

## 3. The `Agent` public contract

This is the shape every phase adheres to. Phase A (Agent core) lands
it; phases B–H extend it without widening the contract in incompatible
ways.

### 3.1 Module location

```
src/t0_agents/
├── core/
│   ├── __init__.py             ← re-exports Agent, AgentConfig
│   ├── agent.py                ← class Agent
│   ├── config.py               ← AgentConfig dataclass
│   ├── types.py                ← AgentResult, ToolProvider (Protocol), AgentEvent*
│   ├── exceptions.py           ← T0AgentsError hierarchy
│   └── elicitation_forwarder.py ← Future-based forwarder used by serve/ (Phase D)
└── __init__.py                 ← re-exports Agent, AgentConfig, AgentResult, run_agent, MaxTurnsExceededError, ToolChoice
```

### 3.2 Constructor signature (shipped in Phase A, extended in B–H)

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        *,
        t0: AsyncTensorZeroGateway,
        tool_provider: ToolProvider,      # typically a fastmcp.Client
    ) -> None: ...
```

`AgentConfig` carries every knob that exists on today's `run_agent`:

```python
@dataclass
class AgentConfig:
    name: str                             # human-readable; used for logging, tags
    function_name: str                    # the TensorZero function name
    max_turns: int = 20
    tool_timeout: float | None = 30.0
    parallel_tools: bool = False
    dryrun: bool = False
    tool_choice: ToolChoice | None = None
    tags: dict[str, str] | None = None    # forwarded to t0.inference (observability)
    namespace: str | None = None          # forwarded to t0.inference (experimentation)
```

`namespace` doubles as an observability tag — TensorZero stores it as
`tensorzero::namespace` automatically (T0 Q3). Deployers do NOT need to
add a duplicate entry to `tags`. If no namespace-specific
experimentation block exists in `tensorzero.toml`, the gateway falls
back to the default config silently — passing an unknown namespace is
not an error.

In later phases this dataclass grows:

| Phase | New field |
| --- | --- |
| B (memory) | `memory: MemoryProvider \| None = None` — when set, hydrates/persists per `episode_id`. |
| B (memory) | `memory_strategy: Literal["auto", "system_template", "prepended_message"] = "auto"` — see §2.4 memory injection path. |
| C (guardrails) | `guardrails: list[Guardrail] = field(default_factory=list)` — extra interception. |
| D (serving) | — (no field; `serve.create_app(agent)` wraps the existing instance). |
| E (streaming) | — (Agent gains `.stream()` method, no config change). |
| G (multi-agent) | `sub_agents: list[Agent] = field(default_factory=list)` — exposed as MCP tools via an in-process registry. |

**Principle.** No phase should remove or rename an `AgentConfig` field.
New fields default to `None`/empty-list so old code continues to work.

### 3.3 Methods

```python
class Agent:
    async def run(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
    ) -> AgentResult: ...

    async def stream(
        self,
        user_message: str,
        *,
        episode_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]: ...  # lands in Phase E
```

Semantics:

- `run()` is a one-shot blocking call. Returns `AgentResult` (same
  dataclass as today's `run_agent`).
- `stream()` yields `AgentEvent`s (delta events, tool-call events,
  tool-result events, terminal done event). Equivalent behaviour to
  `run()` when the async iterator is exhausted; the last event is
  `DoneEvent(result=AgentResult(...))`.
- Both methods are re-entrant: multiple concurrent `run()`s against
  one `Agent` instance are safe **as long as** the underlying `t0` and
  `tool_provider` are safe to share (see concurrency note in §3.5).
- Neither method owns the lifecycle of `t0` or `tool_provider` — the
  caller does.

**`AgentEvent` correlation key (resolved via T0 Q1).** Every event
that carries an `id` field uses the same `id` value emitted by the
TensorZero streaming chunk — `TextChunk.id`, `ToolCallChunk.id`,
`ThoughtChunk.id`. This stable ID is the correlation key that lets a
consumer pair a `TextDeltaEvent` with the `TextEndEvent` for the same
content block, accumulate `ToolCallDeltaEvent`s into a single
`ToolCallEvent`, and so on. Phase E formalises the event union; this
contract pins the correlation contract.

### 3.4 Lifecycle ownership

Same as `run_agent` today, restated:

```python
async with await AsyncTensorZeroGateway.build_http(...) as t0, Client(mcp) as client:
    agent = Agent(
        AgentConfig(name="assistant", function_name="agent"),
        t0=t0,
        tool_provider=client,
    )
    result = await agent.run("Hello")
    # agent holds references but does not close t0 or client.
```

`Agent` does **not**:

- Open / close the gateway or the MCP client.
- Set up logging.
- Install signal handlers.
- Spawn threads or subprocesses.

`Agent` **does**:

- Hold a reference to `t0`, `tool_provider`, and its memory /
  guardrail / sub-agent collaborators.
- Delegate each `run()` to `mcp.loop.run_agent`, surrounded by
  memory/guardrail hooks once those phases land.
- Raise `MaxTurnsExceededError` (and the new exceptions introduced in
  later phases — see §5) without mutating the caller's gateway state.

### 3.5 Concurrency (resolved via FastMCP Q3)

`fastmcp.Client` supports concurrent `call_tool` / `list_tools` via
reentrant context managers and reference-counted session state
(`fastmcp/client/client.py:96-111`, `:522-551`). However, the
`elicitation_handler` is a **single instance shared across every call
on the Client** — there is no per-call routing. Sharing one `Client`
across multiple users would route all elicitation requests to the same
handler, so the framework cannot demultiplex them.

Therefore:

- One `Agent` instance can serve many sequential `run()` calls against
  the same gateway + client pair safely.
- Concurrent `run()` calls on **one shared** `Agent`/`Client` pair are
  safe **only if no tool uses elicitation**. With elicitation, each
  concurrent caller needs its own `Client` (and therefore its own
  `Agent`).
- The `serve/` phase enforces one `Client` (and one `Agent`) per
  session because elicitation is on the table for any served agent.

Not supported:

- Calling methods on an `Agent` after the gateway or client has been
  closed. The caller owns the lifecycle.

---

## 4. How `Agent` composes `mcp/`

Phase A ships the minimal composition. The sketch below uses pseudo-code
so the contract is clear; real code lives in
[`phases/phase-a-agent-core.md`](phases/phase-a-agent-core.md).

```python
class Agent:
    async def run(self, user_message, *, episode_id=None) -> AgentResult:
        # 1. (Phase B) Hydrate memory
        prefix_messages = []
        if self._memory:
            prefix_messages = await self._memory.load(episode_id)

        # 2. (Phase C) Pre-run guardrails
        for g in self._guardrails:
            await g.before_run(user_message, config=self._config)

        # 3. Delegate to existing run_agent — contract unchanged
        result = await run_agent(
            self._t0,
            self._tool_provider,
            user_message,
            self._config.function_name,
            episode_id=episode_id,
            max_turns=self._config.max_turns,
            tool_timeout=self._config.tool_timeout,
            parallel_tools=self._config.parallel_tools,
            dryrun=self._config.dryrun,
            tool_choice=self._config.tool_choice,
            # tags + namespace added via kwargs forwarding once open Qs resolved
        )

        # 4. (Phase B) Persist memory
        if self._memory:
            await self._memory.save(result.episode_id, prefix_messages, result)

        # 5. (Phase C) Post-run guardrails
        for g in self._guardrails:
            await g.after_run(result, config=self._config)

        return result
```

Key property: steps 1, 2, 4, 5 are **no-ops** when the respective
collaborators are absent. Phase A ships with all four reduced to
nothing; the exact call to `run_agent` in step 3 is byte-identical to
what a developer calling `run_agent` directly would issue.

---

## 5. Error model

### 5.1 Existing, keeps

- `MaxTurnsExceededError` from `core/exceptions.py`. Unchanged.

### 5.2 Planned additions

| Exception | Raised by | When |
| --- | --- | --- |
| `T0AgentsError` | *(base)* | Abstract base for every custom exception in the framework. `MaxTurnsExceededError` inherits from it. |
| `ConfigurationError` | `AgentConfig.__post_init__` / `Agent.__init__` | Invalid config (e.g. missing function_name, negative max_turns, missing memory when memory_strategy demands it). |
| `MemoryBackendError` | `memory/*.py` implementations | Persistence layer failure (disk IO, Redis disconnected, etc.). Does **not** abort a run — the Agent logs and continues without memory. Raising is reserved for configuration-time failures. |
| `GuardrailBlockedError` | `guardrails/*.py` | A guardrail blocked an action (e.g. cost limit). Propagates; caller decides recovery. |
| `SubAgentError` | `compose/*.py` | Wraps an error from a delegated sub-agent. Unwrapping gives the original exception. |

**Non-goals.**

- No `T0AgentsWarning` — use Python `warnings` module directly if ever
  needed.
- No exception subclassing of `TensorZeroError` — those propagate from
  the gateway unchanged.
- No swallowing of tool errors; the existing `ToolResult("ERROR: ...")`
  contract stays exactly as documented in `developer-guide.md §10`.

### 5.3 Logging

All logging uses `logging.getLogger("t0_agents.<submodule>")`. The
package installs a `NullHandler` on import (already done today at
`src/t0_agents/__init__.py:14`). No phase replaces this with a
`StreamHandler` / `basicConfig`; that's the caller's choice.

---

## 6. Backwards compatibility guarantee

### 6.1 `run_agent` signature is frozen through the whole roadmap

The function signature at `src/t0_agents/mcp/loop.py:187-199` — public
kwargs, default values, return type — does not change. A new optional
kwarg may be added **only** if all of these hold:

1. It defaults to `None` or an equivalent "absent" value.
2. The behaviour when absent is byte-identical to today's behaviour.
3. It's documented in `developer-guide.md` and in the matching phase
   implementation log.

### 6.2 `AgentResult` dataclass is frozen

`response`, `episode_id`, `inference_ids`, `turns`, `response_source`
are the current fields. New fields may be added only with default values
that preserve existing consumer code (e.g. `list[str]` defaulting to
empty, not `None`).

### 6.3 `mcp/` public exports are frozen

Every symbol exported by `src/t0_agents/mcp/__init__.py` stays available
with the same signature:

- `convert_tools`, `serialize_tool_result`, `build_tool_result`
- `run_agent`, `execute_tool_calls`, `execute_tool_calls_parallel`,
  `has_tool_calls`
- `auto_approve_handler`, `auto_decline_handler`

### 6.4 Test quality floor

After every phase:

- `uv run pytest -m "not integration"` passes.
- `uv run ruff check src/` passes.
- `uv run mypy src/` passes in strict mode.

---

## 7. Resolved open items (2026-04-15)

All twelve open questions are answered in
[`02-open-questions.md`](02-open-questions.md). Summary of contract-
relevant resolutions:

| Question | Resolution | Contract impact |
| --- | --- | --- |
| **T0 Q1** (streaming chunks) | Per-content-block deltas with stable IDs (`text` / `tool_call` / `thought`); `usage` final chunk; `[DONE]` sentinel; no `finish_reason` field. | §3.3 — `AgentEvent` correlation key formalised. |
| **T0 Q2** (`additional_tools` mutation) | Per-call parameter; gateway does not enforce stability. LLM-side schema-change risk is on the framework. | §3.2 — `sub_agents` field is the declarative path; runtime mutation is a future enhancement. |
| **T0 Q3** (`namespace` vs `tags`) | Orthogonal; namespace falls back gracefully and is auto-stored as `tensorzero::namespace` tag. | §3.2 — both fields shipped; no duplication needed. |
| **T0 Q4** (memory injection) | Both `system_template` and `prepended_message` (via `RawText` block) are supported. | §2.4 + §3.2 — `memory_strategy` field documented. |
| **T0 Q5** (guardrail feedback) | No auto-emit. Metrics must be pre-declared; framework documents the manual `t0.feedback(...)` call. | None. |
| **T0 Q6** (episode lifecycle) | No close API. Persist memory upsert per `Agent.run()`. | None. |
| **T0 Q7** (embedded gateway) | `build_embedded` IS available in v2026.4.0 with optional backend URLs. Phase H ships both `MockGateway` and an embedded helper. | None. |
| **FastMCP Q1** (runtime tool registration) | `add_tool(Tool)` works at runtime; `Tool.from_function(handler, name=...)` produces the named instance. | None. |
| **FastMCP Q2** (remote elicitation) | No documented abstraction; framework builds its own `core/elicitation_forwarder.py`. | §3 — adds the forwarder module to the `core/` map (see `phases/phase-d-serving.md`). |
| **FastMCP Q3** (Client concurrency) | `call_tool` reentrant-safe; `elicitation_handler` is shared per Client. Per-session Client mandatory whenever elicitation is in play. | §3.5 — concurrency note rewritten above. |
| **FastMCP Q4** (MCP Resources for memory) | Technically valid; not the primary memory pattern. Document as opt-in. | None. |
| **FastMCP Q5** (elicitation schema) | Strict, validated subset: flat `type:object`, primitives + enum + const, inlined enums, `description`/`default` preserved, scalar wrapping in `{value: ...}`. | None. Phase F's renderer can rely on these guarantees. |

Future amendments — if any phase surfaces a new constraint — are
recorded in that phase's implementation log under
[`logs/`](logs/) and applied here in the same PR.

---

## 8. Glossary (contract-specific terms)

- **ToolProvider**: the structural type `Agent` depends on for tool
  discovery + execution. Today: any `fastmcp.Client`. Protocol defined
  in `core/types.py`.
- **Collaborator**: an optional object injected into `AgentConfig` or
  `Agent.__init__` that extends behaviour (memory, guardrail, sub-agent).
  Collaborators are non-invasive: absent ⇒ behaviour identical to Phase
  A baseline.
- **Phase**: a unit of work that ships a milestone (see
  [`01-implementation-plan.md`](01-implementation-plan.md)). Each phase
  produces at least one public-API change, one example, one doc page,
  and one implementation log.
- **Milestone**: the user-visible outcome at the end of one or more
  phases (e.g. M2 = "Stateful Agent" ships after A + B).
- **Contract amendment**: a change to this document. Amendments are
  always recorded in the corresponding phase log with a rationale.
