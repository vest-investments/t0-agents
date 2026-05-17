# M3 — `di-and-tools` — dev-log

> **Milestone:** `milestone-3-di-and-tools` ([spec](../milestones/milestone-3-di-and-tools.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-17.

## §1 Summary — what was done

M3 closes the FastAPI parallel by landing the
``Depends + tools + sub-agents`` trinity:

1. **`Depends(...)`** is the universal injection sentinel.  An
   internal `DependencyResolver` walks `inspect.signature(fn)`,
   recursively resolves provider chains (sync, async, sync-returning-
   awaitable), caches per-call by `id(Depends)` when `use_cache=True`,
   and consults `app.dependency_overrides[provider]` on every call
   (not snapshotted at registration).  The public `Depends` symbol
   is typed as a callable returning `Any` so a typed parameter
   default — `db: DB = Depends(get_db)` — passes mypy strict.
2. **`@app.tool`** registers user functions as FastMCP tools on the
   App-owned in-memory server.  Parameters whose default is a
   `Depends` sentinel are **stripped** from the auto-generated JSON
   Schema and re-injected at call time via the App's resolver.  Tool
   docstring becomes the MCP tool description.
3. **`Depends(<glow agent wrapper>)` on `@app.agent`** is the one
   piece of introspective magic glow ships: a parameter typed as any
   `Agent`-like with a `Depends(<wrapper>)` default has its provider's
   `_AgentSpec` appended to `_AgentSpec.sub_agent_specs`, and the
   parameter is compiled out of the runtime call signature.
   `_lookup_or_build` depth-first builds those sub-agents and threads
   them into `AgentConfig.sub_agents`.

The M2 §4 carry-forward option (b) lands here too: `App.__init__`
allocates a long-lived in-memory `FastMCP("glow-app")` server + matching
`Client`, every glow agent is constructed with
`tool_provider=app._owned_client`, and the v1.0 toolless-chat fallback
at `core/agent.py:_resolve_chat_tool_provider` is never reached for
glow agents.  `app.build()` / `app.shutdown()` / `app.lifespan()`
manage the Client lifecycle idempotently.

### Files touched

| Path | Change | Lines |
|---|---|---|
| `src/t0_agents/glow/di.py` | New — `Depends` (TYPE_CHECKING-aware function returning `Any`; runtime aliases the `_Depends` dataclass) + `DependencyResolver`. | +162 / −0 |
| `src/t0_agents/glow/__init__.py` | Edit — re-export `Depends`. | +2 / −0 |
| `src/t0_agents/glow/_spec.py` | Edit — add `sub_agent_specs: list[_AgentSpec] = field(default_factory=list)` slot. | +3 / −0 |
| `src/t0_agents/glow/app.py` | Edit — `_mcp_server` + `_owned_client` + `_exit_stack` + `_client_entered` + `_resolver` in `__init__`; new `tool()` method delegating to `_make_tool_decorator`; `_lookup_or_build` depth-first builds sub-agents and threads `sub_agents=` + `tool_provider=`; `build()` enters Client + constructs agents; new `shutdown()` (idempotent); `lifespan()` does `try: build; yield; finally: shutdown`. | +89 / −13 |
| `src/t0_agents/glow/decorators.py` | Edit — `_collect_sub_agent_specs` signature-walk helper; `_make_decorator` calls it before constructing the spec; `_AgentWrapper.__call__` wording updated; new `_make_tool_decorator(app)` with stripped-signature wrapper + Depends-stripping; rejects `Depends(<glow wrapper>)` inside `@app.tool` with pointer at `@app.agent`. | +220 / −12 |
| `tests/test_glow/test_di.py` | New — 7 unit tests (single, nested, cached, uncached, override, sync+async+sync-returns-awaitable, non-Depends-params-untouched). | +211 / −0 |
| `tests/test_glow/test_tool_decorator.py` | New — 6 unit tests (no-Depends schema, Depends-strip, runtime-injection, override-redirects, docstring-as-description, `Depends(<wrapper>)`-raises). | +225 / −0 |
| `tests/test_glow/test_app.py` | Edit — adapt strict-arg test wording; add 8 new tests for sub-agent registration, depth-first construction (with a local `AgentConfig.__post_init__` patch so the unit fake can participate), runtime-call rejection, eager forward-ref rejection, value-style rejection, message-typed-non-str rejection, missing-default rejection, plain-default rejection. | +212 / −10 |
| `tests/test_glow/test_integration.py` | Edit — wrap M2 tests in `async with app.lifespan():` (App-owned client now needs to be entered); add `test_orchestrator_with_glow_sub_agents` mirroring `examples/08_multi_agent` via glow. | +53 / −10 |
| `examples/glow/01_hello.py` | Edit — wrap hot path in `async with app.lifespan():`. | +3 / −2 |
| `examples/glow/02_structured.py` | Edit — wrap hot path in `async with app.lifespan():`. | +4 / −3 |
| `examples/glow/05_di_tools.py` | New — runnable demo (no LLM): `@app.tool`, `Depends(get_db)`, schema-strip assertion, mock override. Prints SUCCESS banner. | +84 / −0 |
| `examples/glow/config/tensorzero.toml` | Edit — add `[functions.orchestrator]` chat function. | +14 / −0 |
| `examples/glow/README.md` | Edit — document the three TZ functions. | +6 / −2 |
| `CHANGELOG.md` | Edit — append six M3 bullets under `#### Experimental — t0_agents.glow`. | +44 / −0 |

### Public glow API delta

`t0_agents.glow.__all__` grew from `["App", "Done", "Error",
"FluentRun", "Selector", "Text", "Thought", "ToolCall", "ToolResult",
"TurnEnd"]` (M2) to 11 entries with the addition of `Depends`.
`App.tool` is reached as a method, not a top-level import.
`DependencyResolver`, `_Depends`, `_make_tool_decorator`,
`_collect_sub_agent_specs` stay internal.

### Commits

The milestone landed as seven commits on `experimental/glow`, in order:

| # | SHA | Subject |
|---|---|---|
| 1 | `508ca43` | `feat(glow): Depends + DependencyResolver` |
| 2 | `40e8fa2` | `fix(glow): make Depends defaults pass mypy strict on typed parameters` |
| 3 | `4862cb9` | `feat(glow): App owns FastMCP server + Client (M2 §4 carry-forward)` |
| 4 | `78c78d5` | `feat(glow): @app.tool decorator with Depends stripping` |
| 5 | `ea37704` | `feat(glow): @app.agent signature walk + sub-agents via Depends(agent)` |
| 6 | `c392775` | `feat(glow): 05_di_tools example + orchestrator integration test` |
| 7 | `f5ddaeb` | `chore(glow): document M3 in CHANGELOG` |
| 8 | (this commit) | `docs(glow): milestone-3 dev-log` |

In the 14-commit roadmap from the plan file these correspond to commits
6, 7, and 12; the additional infrastructure (lifecycle, example,
CHANGELOG, dev-log) account for the remaining five.

### Tests

| File | New tests | Notes |
|---|---|---|
| `tests/test_glow/test_di.py` | 7 | Resolver unit tests. |
| `tests/test_glow/test_tool_decorator.py` | 6 | `@app.tool` unit tests. |
| `tests/test_glow/test_app.py` | 8 | Sub-agents + signature-walk rejection paths. |
| `tests/test_glow/test_integration.py` | 1 | Orchestrator via glow. |

Offline regression: **922 (M2 end) → 944 passing** (+22).
Integration suite: **2 → 3** tests collected.  The 22 new offline
tests run in well under a second.

## §2 Deviations from the milestone spec

Six deviations, all in service of either the project-wide
multi-commit-PR convention or the safer-defaults posture set by the
user during planning (see "Resolved design questions" in the plan
file at `/home/makarenko/.claude/plans/3-happy-island.md`).

### 2.1 — Spec's 3 commits split into 7 story-telling commits

The spec labels the milestone as three commits (Commits 6, 7, 12 of
the 14-commit roadmap).  The project-wide convention codified in
`CLAUDE.md` and reinforced by `feedback_multi_commit_prs` ("split
push/PR work into 5–10 story-telling commits; never single giant
commits") means collapsing those into three would have been a smell.
The chosen split was documented in the plan file:

1. `feat(glow): Depends + DependencyResolver` — adds `di.py` + tests.
2. `fix(glow): make Depends defaults pass mypy strict on typed
   parameters` — typing wrapper.  Surfaced after commit 1 by running
   `uv run mypy tests/test_glow/`; split rather than amended to keep
   commit 1 atomic.
3. `feat(glow): App owns FastMCP server + Client (M2 §4 carry-forward)`
   — plumbing only; no `@app.tool` yet.
4. `feat(glow): @app.tool decorator with Depends stripping` — exercises
   the App-owned client.
5. `feat(glow): @app.agent signature walk + sub-agents via Depends(agent)`
   — closes the agent-side surface.
6. `feat(glow): 05_di_tools example + orchestrator integration test` —
   demonstrates the full DX end-to-end.
7. `chore(glow): document M3 in CHANGELOG`.

**Impact on subsequent milestones:** none.  The plan file's
"Implementation plan — commits" items 6, 7, 12 remain accurate; the
seven-commit split is internal to M3.

### 2.2 — `fix(glow):` precedes the M3 feature commits

Commit 2 (`fix(glow): make Depends defaults pass mypy strict ...`) is
a follow-on correction for commit 1.  The `Depends` typing wrapper
(`if TYPE_CHECKING: def Depends(...) -> Any: ...; else: Depends =
_Depends`) is what FastAPI does and what mypy strict needs; the
initial commit 1 (`feat(glow): Depends + DependencyResolver`) shipped
the dataclass with a literal `Depends` symbol that tripped
`[assignment]` errors on every typed default in `tests/test_glow/test_di.py`.

The fix could have been amended into commit 1, but the M1 dev-log's
M5 carry-forward had flagged the missing `mypy tests/test_glow/` gate;
splitting honours the always-write convention (commit 1 is what was
actually written first; commit 2 is what was added when the gate
caught the issue).

**Impact on subsequent milestones:** none.

### 2.3 — `@app.agent` accepts ONLY `Depends(<glow agent wrapper>)`

The spec's "Detected during `@app.agent` registration when a
parameter is typed `Agent` (or the glow wrapper) and the provider is
a glow-registered agent" is silent on what should happen when the
provider is anything *else* (a value-style provider like
`Depends(get_db)`).  The plan file's example 6 uses value-style
`Depends` exclusively inside `@app.tool`; example 8 uses
`Depends(<wrapper>)` exclusively inside `@app.agent`.

M3 tightens this: `@app.agent` accepts ONLY `Depends(<glow agent
wrapper>)` for non-message parameters.  Value-style providers raise
`ConfigurationError` at decoration time with a pointer to `@app.tool`.
Rationale: M3 has only two defined runtime behaviours for
`@app.agent` parameters — the user message and the sub-agent special
case.  Silently accepting value-style `Depends` (which would have no
runtime effect at M3) would be a footgun.

**Impact on subsequent milestones:** M4 (or later) may relax this if a
runtime hook for value-style `Depends` on `@app.agent` emerges (per-
call context overrides, request-scoped state).  The relaxation is
backward-compatible (more shapes accepted, none rejected).

### 2.4 — App-owned FastMCP server + Client are allocated in `__init__`, not `build()`

The spec's checklist says `_mcp_server` and `_mcp_client` are "created
lazily in `build()` / `lifespan()` ; mounts user tools".  Implementation
splits this:

- `_mcp_server` and `_owned_client` are allocated **synchronously** in
  `App.__init__`.  Construction is cheap and synchronous; `@app.tool`
  must be able to register tools against `_mcp_server` BEFORE the
  user (or test) calls `build()`.
- Only the `Client.__aenter__` (which spawns the in-process transport
  task) is deferred to `build()`.  An internal `AsyncExitStack` owns
  the entry; `shutdown()` closes the stack and replaces it (so a
  subsequent `build()` cleanly re-enters).

This split is essential for the natural decorator order
(`@app.tool` → `async with app.lifespan():`).  Without it, the user
would have to register every tool inside `lifespan()`'s body, which
would defeat the FastAPI-style decoration pattern.

**Impact on subsequent milestones:** M4's typed-MCP work (`MCPServer`
class + `Annotated`) extends the same `_mcp_server` via
`FastMCP.mount(...)`.  The mount should happen inside `build()` (or
`lifespan()`'s pre-yield section), AFTER `_mcp_server` exists but
ideally BEFORE the client is entered — so the mounted tools are
visible at the first `list_tools()` call.  Update the deprecation
message in `compose/builtins.py:race()` if M4's `compose(set[...])`
deviates from the `parallel(*, reducer=..., error_policy="continue")`
shape (M0 §4 anchor).

### 2.5 — Eager forward-ref enforcement for `Depends(<wrapper>)`

The spec is silent on what happens when `Depends(<other_wrapper>)` is
declared against an unregistered (or differently-app'd) wrapper.  M3
raises `ConfigurationError` at decoration time with a clear pointer:
"register the sub-agent BEFORE the orchestrator that depends on it
(glow does not support forward references between agents)."

**Impact on subsequent milestones:** none.  FastAPI takes the same
stance; the constraint is well-understood.

### 2.6 — `Depends(<glow wrapper>)` inside `@app.tool` raises eagerly

The spec is silent on this case.  M3 chooses to reject: a tool that
re-entered the agent loop via a sub-agent value injection would
introduce non-trivial re-entrance semantics (cancellation, double
loops, episode-id confusion) with marginal benefit at M3.

**Impact on subsequent milestones:** M4 / M5 may revisit if the
use-case emerges; for now the surface stays minimal.

## §3 Open questions

### 3.1 — Per-agent MCP subsets remain a known limitation

Every glow agent gets the same `app._owned_client` as its
`tool_provider`, so every agent on a given `App` sees every
`@app.tool` registration plus every `<sub_agent>_ask` tool installed
by `install_sub_agents`.  The plan file's "Out of scope" explicitly
defers per-agent MCP subsets to a follow-up.

A real-world deployment that needs per-agent isolation either creates
multiple `App` instances (each with its own `_owned_client`) or uses
v1.0 directly with hand-rolled `Client`s.  This is fine for the
experimental phase; M5's tutorial should mention it as a known
limitation in `docs/glow/limitations.md`.

### 3.2 — `Depends(get_db)` rejection on `@app.agent` may be too strict

Value-style `Depends` on `@app.agent` raises `ConfigurationError` at
decoration time today (deviation §2.3).  If user feedback during
dogfooding shows demand for per-call context (e.g., a `Depends(get_user)`
that surfaces the current user inside the agent's system prompt or
sub-agent routing), M4+ may relax this with a real runtime hook.  The
relaxation is backwards-compatible.  Parked.

### 3.3 — `_AgentWrapper.__call__` accepts `*args` typed by ParamSpec

`_AgentWrapper.__call__` uses ParamSpec to preserve the wrapped
function's signature at the static level, but the runtime body only
accepts exactly one positional `str`.  mypy doesn't catch
`tutor("hi", research=researcher)` statically (because ParamSpec lifts
the WHOLE signature including the sub-agent param), but the runtime
raises a clear `TypeError`.  This dual-personality is acceptable
because the runtime error is loud and the type-check pass mirrors
the user's intent.  Could be tightened in a follow-up with a custom
`__call__` signature (no ParamSpec) at the cost of losing static
return-type inference for the chat case.  Parked.

## §4 Carry-forward notes

### For M4 (`composition-mcp-guardrails`)

- **`compose(set[Composable])` must use
  `parallel(*, reducer=..., error_policy="continue")`** — anchor from
  M0 §4 still applies.  If M4 surfaces a reason to deviate, record
  it in M4's dev-log §2 AND update the deprecation message in
  `compose/builtins.py:race()` accordingly.
- **`Annotated[Optional[T], meta]` peel order** — see M1 §3.1.  M3
  did not encounter the case because the signature walk only inspects
  parameter *defaults*, never the annotated types.  M4's guardrail-
  on-the-parameter pattern (`Annotated[T, Guardrail(...)]`) and
  guardrail-on-the-return-type pattern both need either an iterative
  peel or inverted peel order in `_typing.py`.
- **App-owned `_mcp_server` is the mount surface.**  Typed MCP
  (`MCPServer` class + `Annotated`) should call
  `await app._mcp_server.mount(server, namespace=...)` inside the
  `build()` body (or `lifespan()` between client entry and yield) so
  external MCP tools appear on `app._owned_client.list_tools()` along
  with `@app.tool` registrations and `<sub_agent>_ask` installs.  The
  mount order matters: do it BEFORE entering the client, OR re-enter
  the client after mounting — pick one and document it.
- **Optional: `App.__init__` may grow `servers=list[type[MCPServer]]
  | None = None`** for the class-based MCP pattern (M2 §4 D1 deferred
  this).
- **Value-style `Depends` on `@app.agent` is currently rejected.**
  M4 may relax if it introduces a real runtime hook for the value;
  see §3.2.

### For M5 (`polish`)

- **`app.lifespan()` is the recommended entry point.**  M3 makes the
  Client lifecycle explicit; the tutorial must call this out.
  Without `lifespan()` (or explicit `build()/shutdown()`), `list_tools()`
  against the unentered Client fails with whatever FastMCP error.
- **`05_di_tools.py` exercises only the MCP round-trip, not the
  agent-uses-tool path.**  A future
  `examples/glow/07_agent_with_tools.py` (or similar) would
  demonstrate the LLM-calls-glow-tool flow end-to-end.  Not blocking
  for M5 but worth noting.
- **Pre-public-polish backlog stays as-is.**  M0 §3 logged 4 race
  references not yet propagated to the public docs; M3 did not touch
  them.
- **`_AgentWrapper.__call__` dual-personality** (§3.3) — M5's polish
  may want to tighten or document this.

### For the polish backlog (`project_pre_public_polish.md`)

- The M3 lifespan requirement should be documented prominently in
  `docs/glow/getting-started.md` (M5 work).
- The M3 deviation §2.3 (value-style `Depends` rejected on
  `@app.agent`) should be cross-linked from the limitations page.

## §5 Verification

Captured commands and outcomes (run at the end of M3 implementation,
prior to writing this dev-log):

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/glow/ tests/test_glow/
All checks passed!

$ uv run mypy src/t0_agents/glow/ tests/test_glow/
Success: no issues found in 17 source files

$ uv run pytest tests/test_glow/ -q -m "not integration"
............................................................           [100%]
60 passed, 3 deselected in 0.21s

$ uv run pytest -m "not integration" -q
[...]
944 passed, 2 skipped, 60 deselected in 6.97s
# Pre-M3 baseline (end of M2): 922 passing.  M3 adds 22 new offline
# tests (7 test_di.py + 6 test_tool_decorator.py + 8 test_app.py +
# 1 test_app.py was adapted in place + 0 net for test_integration.py
# which adds 1 integration-marked test); total = 944.

$ docker compose -f examples/glow/docker-compose.yml --env-file .env up -d
[...]
 Container glow-clickhouse-1  Healthy
 Container glow-gateway-1  Started

$ uv run pytest tests/test_glow/test_integration.py -q
...                                                                      [100%]
3 passed in 3.89s
# All three integration tests pass against the live gateway:
# test_hello_world_chat_agent_runs_end_to_end (M2),
# test_structured_output_returns_pydantic_instance (M2),
# test_orchestrator_with_glow_sub_agents (M3).

$ uv run python examples/glow/05_di_tools.py
Advertised tool params: ['order_id']
Real DB:  Order #1 — A book
Mock DB:  [MOCKED in test override]

SUCCESS: di-tools ran end-to-end.

$ git diff src/t0_agents/mcp/ | wc -l
0
# (mcp/ submodule contract preserved per feedback_mcp_standalone)

$ git log --oneline experimental/glow ^main | wc -l
26
# 5 scaffold+M0 + 4 M1 + 8 M2 + 7 M3 code/CHANGELOG commits + 2 PR-
# merge artefacts.  The dev-log commit (this one) will bring the
# total to 27.

$ git log --oneline experimental/glow ^main | head -7
f5ddaeb chore(glow): document M3 in CHANGELOG
c392775 feat(glow): 05_di_tools example + orchestrator integration test
ea37704 feat(glow): @app.agent signature walk + sub-agents via Depends(agent)
78c78d5 feat(glow): @app.tool decorator with Depends stripping
4862cb9 feat(glow): App owns FastMCP server + Client (M2 §4 carry-forward)
40e8fa2 fix(glow): make Depends defaults pass mypy strict on typed parameters
508ca43 feat(glow): Depends + DependencyResolver
# The 7 M3 code/CHANGELOG commits, all with the (glow) scope.

$ git diff --stat origin/main..HEAD | tail -3
[...]
 50 files changed, 6227 insertions(+), 116 deletions(-)
# Allowed paths: src/t0_agents/glow/**, tests/test_glow/**,
# examples/glow/**, docs/internals/glow/**, CHANGELOG.md,
# pyproject.toml, uv.lock — plus the M2 D5 core exception
# (src/t0_agents/core/agent.py + tests/test_core/test_agent.py)
# inherited from the M2 PR.  No new collateral damage in M3.

$ uv run python -c "
from t0_agents.glow import App, Depends, FluentRun, Thought, ToolCall, Text
print('App:', App)
print('Depends:', Depends)
print('FluentRun:', FluentRun)
print('Selectors:', Thought, ToolCall, Text)
"
App: <class 't0_agents.glow.app.App'>
Depends: <class 't0_agents.glow.di._Depends'>
FluentRun: <class 't0_agents.glow.handle.FluentRun'>
Selectors: <t0_agents.glow.selectors.Selector object ...> ...
# End-to-end smoke confirms every M3 public symbol imports cleanly
# from the package root.  ``Depends`` is the `_Depends` dataclass at
# runtime (the static-typing wrapper kicks in only for mypy).
```

The full pre-push `uv run pytest -q` (no marker filter) is the user's
responsibility per `feedback_full_test_suite_before_push`; the
3-test integration suite was validated to pass against the gateway
in this session.
