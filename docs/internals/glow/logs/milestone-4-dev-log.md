# M4 — `composition-mcp-guardrails` — dev-log

> **Milestone:** `milestone-4-composition-mcp-guardrails` ([spec](../milestones/milestone-4-composition-mcp-guardrails.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-17.

## §1 Summary — what was done

M4 lands the three high-leverage glue features that round out the
public glow surface:

1. **`compose(spec, *, merge=None)` — native-collection composition.**
   `list`/`tuple` → `sequential`; `set`/`frozenset` → `parallel(*,
   reducer=merge or _default_merge, error_policy="continue")` (anchored
   on the M0 carry-forward); `dict[str, Composable]` → a custom
   `_LLMRoutedComposer` Composer subclass (a user-confirmed deviation
   from the spec's literal `sequential(classifier, conditional(...))`
   shape — see §2.1).  Returns a `Composer`, so nesting (`compose([...,
   compose({...}), ...])`) is free via the v1.0 `Composable` Protocol.

2. **Typed `MCPServer` integration.**  Class-based
   (`class GithubMCP(MCPServer, url=...): ...`) and Annotated
   (`Annotated[Any, MCPServer(url=...)]`) patterns both produce an
   `MCPServerSpec`.  `App(servers=[...])` activates the M2 §4 D1
   placeholder; `@app.agent` parameter walks via `_extract_mcp_specs`
   append discovered specs to `app._mcp_specs`.  `App.build()` mounts
   every spec onto the `glow-app` composed FastMCP BEFORE entering the
   owned Client — `mount(server)` for in-memory `FastMCP` instances,
   `mount(create_proxy(url))` for URL-based remotes.

3. **`Annotated[T, Guardrail(...)]` extraction.**  `@app.agent`
   parameter and return annotations are walked; every metadata entry
   satisfying the v1.0 `Guardrail` Protocol is harvested into a new
   `_AgentSpec.guardrails` slot and folded into
   `AgentConfig.guardrails` on first build.  Runtime path
   (`GuardrailsMiddleware`) unchanged.  Ships a glow-private
   `regex_guardrail(pattern, message, *, phase="after_run")` factory
   because the existing `t0_agents.guardrails` surface lacks regex
   blockers; the M5 carry-forward (§4) tracks the migration to
   `t0_agents.guardrails.regex` so every Agent consumer benefits.

A precursor commit also resolves the M1 §3.1 carry-forward (iterative
`_peel_to_core` for `Annotated[Optional[T], m]` and
`Optional[Annotated[T, m]]` ordering) — every M4 commit that walks
return-type metadata depends on it.

### Files touched

| Path | Change | Lines |
|---|---|---|
| `src/t0_agents/glow/_typing.py` | Add `_peel_to_core(tp)` iterative peel helper. | +18 / 0 |
| `src/t0_agents/glow/output_inference.py` | Switch to `_peel_to_core`; preserve both `cast(OutputMode[Any], …)` sites. | +5 / −2 |
| `src/t0_agents/glow/compose.py` | New — `compose()`, `_default_merge`, `_unwrap_node`, `_LLMRoutedComposer`, `_llm_routed`, `_make_router_agent`, `_find_leaf_agent`. | +387 / 0 |
| `src/t0_agents/glow/mcp_typing.py` | New — `MCPServer` (dual-personality class), `MCPServerSpec`, `_extract_mcp_specs`, `_is_mcp_parameter`. | +233 / 0 |
| `src/t0_agents/glow/guardrails_typing.py` | New — `regex_guardrail`, `_extract_guardrails`, `_guardrails_from_annotation`. | +179 / 0 |
| `src/t0_agents/glow/app.py` | Add `servers=` kwarg; `_mcp_specs`; mount loop in `build()` BEFORE client entry; `guardrails` threaded into `AgentConfig`. | +44 / −2 |
| `src/t0_agents/glow/decorators.py` | Call `_extract_mcp_specs` + `_extract_guardrails`; skip MCP-typed params in `_collect_sub_agent_specs`; pass `guardrails` to `_AgentSpec`. | +20 / 0 |
| `src/t0_agents/glow/_spec.py` | Add `guardrails: list[Guardrail]` slot. | +6 / 0 |
| `src/t0_agents/glow/__init__.py` | Re-export `compose`, `MCPServer`, `regex_guardrail`; `__all__` 11 → 14. | +4 / 0 |
| `tests/test_glow/test_output_inference.py` | +4 cases for iterative peel coverage. | +57 / −1 |
| `tests/test_glow/test_compose.py` | New — 18 unit tests. | +338 / 0 |
| `tests/test_glow/test_mcp_typing.py` | New — 17 unit + 1 in-process mount test. | +228 / 0 |
| `tests/test_glow/test_guardrails_typing.py` | New — 16 unit tests. | +189 / 0 |
| `tests/test_glow/test_integration.py` | +3 integration tests (LLM router x 2, regex guardrail x 1). | +88 / −1 |
| `examples/glow/04_composition.py` | New — set ensemble + composer-in-composer. | +96 / 0 |
| `examples/glow/06_mcp_typing.py` | New — class-based MCP + commented HTTP. | +95 / 0 |
| `CHANGELOG.md` | Append 9 M4 bullets under `Experimental — t0_agents.glow`. | +86 / 0 |

### Public glow API delta

`t0_agents.glow.__all__` grew from 11 to 14 entries: adds `compose`,
`MCPServer`, `regex_guardrail`.  `MCPServerSpec` stays private (the
class form `cls._mcp_spec` and the instance form `_to_spec()` are the
two stable accessors).  `_LLMRoutedComposer`, `_default_merge`,
`_extract_mcp_specs`, `_extract_guardrails`, `_peel_to_core` stay
underscore-prefixed.

### Commits

The milestone landed as eight commits on `experimental/glow`, in order:

| # | SHA | Subject |
|---|---|---|
| 1 | `fd50b1c` | `fix(glow): iterative Annotated/Optional peel in output_inference` |
| 2 | `1d414c3` | `feat(glow): compose() native-type dispatcher` |
| 3 | `caafb89` | `feat(glow): LLM-routed dict composer` |
| 4 | `a50d1f4` | `feat(glow): typed MCPServer integration` |
| 5 | `717293d` | `feat(glow): Annotated[T, Guardrail(...)] extraction + regex_guardrail` |
| 6 | `eff570b` | `feat(glow): 04_composition + 06_mcp_typing examples` |
| 7 | `c711499` | `chore(glow): document M4 in CHANGELOG` |
| 8 | (this commit) | `docs(glow): milestone-4 dev-log` |

In the 14-commit roadmap from the plan file these correspond to
commits 8, 9, 10, 11, plus the supporting precursor (peel order fix),
the examples chore, the CHANGELOG entry, and this always-write log.

### Tests

| File | New tests | Notes |
|---|---|---|
| `tests/test_glow/test_output_inference.py` | 4 | Iterative peel coverage. |
| `tests/test_glow/test_compose.py` | 18 | Native dispatch + LLM router unit tests. |
| `tests/test_glow/test_mcp_typing.py` | 18 | 17 unit + 1 in-process mount end-to-end (offline; no LLM). |
| `tests/test_glow/test_guardrails_typing.py` | 16 | Regex guardrail + Annotated extraction. |
| `tests/test_glow/test_integration.py` | 3 | LLM-routed dispatch (× 2) + return-side regex guardrail (all `@pytest.mark.integration`). |

Offline regression: **944 (M3 end) → 1002 passing** (+58).  Glow-only
offline: **60 → 118**.  Integration suite: **3 → 6 collected**.  The
58 new offline tests run in ~0.5 s.

## §2 Deviations from the milestone spec

Six recorded deviations.  §2.1 is the load-bearing structural one;
§2.4 is the largest scope expansion (HTTP mount was originally
planned to defer, but `create_proxy(url)` made it trivial); the rest
are mechanical accommodations forced by the v1.0 surfaces.

### 2.1 — D1: LLM router is a custom `_LLMRoutedComposer`, not `sequential(classifier, conditional(...))`

The spec literal (`return sequential(classifier, conditional(router=lambda r: r.value[0].name, routes=routes))`)
doesn't work for three reasons that surface when you read
`_drive_sequential` / `_drive_conditional` end-to-end:

1. `_drive_sequential`'s default `chain` (`_default_sequential_chain`
   at `compose/_runtime.py:60-75`) JSON-stringifies the classifier's
   `AgentResult.value` before the next child sees it.  By the time
   `conditional`'s router fires, the value is a string like
   `'[{"name": "math_route", ...}]'`, not a `list[ToolCall]`.
2. `conditional` (via `_drive_conditional` at `_runtime.py:499-538`)
   passes its `input_value` to BOTH the `router` AND
   `routes[route_key].invoke(input_value)`.  Even with a custom
   `chain=` that carries the `AgentResult` forward, the picked route
   would receive the AgentResult (or the tool name), not the user's
   original query.
3. A workaround using a tuple-passing `chain` (e.g. `chain=lambda r,
   orig: (r.value[0].name, orig)`, `router=lambda x: x[0]`, plus
   per-route adapters that extract `x[1]`) works but introduces a
   tuple convention that leaks into every route definition — the
   user-facing surface would be much worse than the spec's promise.

Resolution chosen during planning (user-confirmed): ship a small
dataclass `_LLMRoutedComposer(Composer)` whose `_driver` runs the
classifier, reads `result.value[0].name`, and dispatches to
`routes[key]` with the ORIGINAL input.  Concurrency-safe (no shared
state between invocations), composer-internal (no chain coupling),
and only ~50 LOC.

**Impact on subsequent milestones:** M5 docs must describe the LLM
router as a glow-internal composer rather than as `sequential +
conditional`.  The synthetic classifier construction (no-op FastMCP
tools, ToolCallOutput, `expects_tool_loop=False`) is the actual
mechanism — that's the right story to tell in the tutorial.  Post-M5
may consider exposing `_LLMRoutedComposer` publicly if the dict-router
DX proves popular (currently underscore-prefixed).

### 2.2 — D2: HTTP MCP transport supported in M4 (was deferred in the spec)

The original plan flagged HTTP MCP servers as out-of-scope for M4 (an
implementation footnote estimated 50–150 LOC of transport plumbing).
Reading `fastmcp/server/server.py:2259` showed
`fastmcp.server.create_proxy(target)` accepts URL strings, Clients,
Transports, MCPConfigs, and Paths directly and returns a mountable
`FastMCPProxy`.  Total HTTP support cost: 3 lines inside the
`App.build()` mount loop (`backend = create_proxy(spec.url) if
spec.server is None else spec.server`).

We took the path.  `examples/glow/06_mcp_typing.py` includes a
commented-out `class GithubMCP(MCPServer, url="...")` block to
document the surface; the auto-test only mounts in-memory servers
(offline) so the demo doesn't require a remote MCP endpoint or
authentication.

**Impact on subsequent milestones:** M5 examples and tutorial can
freely demo HTTP MCPs.  The known limitation is that `create_proxy`
inherits whatever transport semantics FastMCP supports; auth headers
and SSE-specific knobs are user-handled.

### 2.3 — D3: LLM router uses `additional_tools` (verified — not a deviation, but a planning correction)

The pre-implementation worry was that `ToolCallOutput` couldn't carry
dynamic synthetic tools, forcing a chat-output classifier as a
fallback.  Reading `tensorzero.pyi:578-622` and `mcp/adapter.py:14` +
`mcp/loop.py:343-349` showed `additional_tools` is the standard
`t0.inference()` kwarg AND t0-agents already auto-converts the
FastMCP `tool_provider`'s `list_tools()` into the TZ format.  So the
synthetic `FastMCP("glow-router-…")` server with one no-op tool per
route key gives us dynamic tools "for free" — `ToolCallOutput`-based
classification is spec-faithful and works without TZ-config coupling.

Documented here in case a future implementer is tempted to revisit
the "fall back to ChatOutput" idea — it is not necessary.

### 2.4 — D4: monkeypatching ``glow.compose.Agent`` needs ``importlib.import_module``

`glow/__init__.py` does `from t0_agents.glow.compose import compose`,
which rebinds the `compose` name on the parent package namespace from
the submodule (`t0_agents.glow.compose` as a module) to the function
(`compose`).  Subsequent attribute access on `t0_agents.glow` returns
the function; pytest's `monkeypatch.setattr("t0_agents.glow.compose.Agent",
fake)` walks the parent package and fails with `'function' object …
has no attribute 'Agent'`.

The fixture in `tests/test_glow/test_compose.py` resolves this by
`importlib.import_module("t0_agents.glow.compose")` to retrieve the
real module object from `sys.modules`, then
`monkeypatch.setattr(module, "Agent", fake)`.  Documented in the
fixture's docstring so a future contributor doesn't get tripped up.

**Impact on subsequent milestones:** none for M5 functionality, but
any future test that needs to patch `glow.compose.<symbol>` needs the
same workaround.

### 2.5 — D5: `_AgentSpec` slots additions instead of refactor

The spec's plan suggested a per-extractor approach (separate fields for
`mcp_specs` and `guardrails` on `_AgentSpec`).  Per-agent MCP subsets
are explicitly deferred (every glow agent on a given App sees every
mounted server), so the App-level `_mcp_specs` is the right home for
those — `_AgentSpec` only grew a `guardrails` slot.  This keeps the
spec dataclass slim and reflects the actual granularity.

**Impact on subsequent milestones:** if per-agent MCP subsets land in
a post-M5 milestone, `_AgentSpec.mcp_specs` can be introduced then
without breaking the M4 surface.

### 2.6 — D6: regex guardrail not migrated to v1.0 core in M4

User direction was clear: the regex guardrail stays glow-private for
now (the M4 spec test required it; the existing
`t0_agents.guardrails` surface doesn't have one).  Migration to
`src/t0_agents/guardrails/regex.py` so every Agent consumer benefits
is recorded as a carry-forward — see §4.

## §3 Open questions

### 3.1 — `_LLMRoutedComposer` enters its classifier Client per-invocation

Each `_LLMRoutedComposer.invoke()` opens a fresh `async with
self.classifier_client:` block.  FastMCP's in-memory transport is
cheap, so the per-call setup cost is negligible compared to the LLM
inference itself.  But: a long-lived composer accumulates many
short-lived transport task entries over its lifetime.  Tests in the
offline suite that exercise the composer (none today; M4's are all
construction-only) don't show any task-leak warnings.  Parked.

If the dict-router becomes hot-path in a real deployment, M5 polish
or post-M5 may want to cache the entered Client on first invocation
(at the cost of cleanup semantics — would need a context manager
shape or `__del__` hook).

### 3.2 — Per-agent MCP subsets remain a known limitation

Inherited from M3 §3.1 and unchanged: every glow agent on a given App
sees every mounted MCP server, including the user's `@app.tool`
registrations.  Real-world deployments wanting per-agent isolation
construct multiple `App` instances.  Tutorial / `docs/glow/limitations.md`
should mention this prominently in M5.

### 3.3 — Resolved questions from prior dev-logs

- **M1 §3.1 (peel order for `Annotated[Optional[T], m]`)** — resolved.
  Commit 1 (`fd50b1c`) replaces the single-pass peel with
  `_peel_to_core(tp)` (iterative `while changed:` loop) in
  `_typing.py`.  Four new test cases lock the behaviour in
  `test_output_inference.py`.
- **M2 §4 D1 (`App(servers=...)` deferred kwarg)** — resolved.  Commit
  4 (`a50d1f4`) adds the kwarg + the spec-list seeding logic.
- **M2 §3.1 (public surface for `_resolve_chat_tool_provider`)** —
  still parked; M4 didn't touch the chat fallback.  Out of scope.
- **M3 §3.1 (per-agent MCP subsets)** — restated in §3.2 above;
  unchanged in M4.

## §4 Carry-forward notes

### For M5 (`polish`)

- **Migrate `regex_guardrail` to `src/t0_agents/guardrails/regex.py`.**
  Same shape as the M2 D5 toolless-chat unlock — touches v1.0 core,
  requires CHANGELOG `### Added` under v1.0 (not just glow), benefits
  every Agent consumer.  Until migrated, `regex_guardrail` is glow-
  private (re-exported from `t0_agents.glow` only).  The Protocol
  conformance is already correct (the concrete class satisfies the
  v1.0 `Guardrail` Protocol), so the migration is move + re-export.
- **Document the dict-router shape in `docs/glow/getting-started.md`.**
  The composer is a custom `_LLMRoutedComposer`, not
  `sequential(classifier, conditional(...))` — the tutorial should
  describe the actual mechanism (synthetic no-op FastMCP tools +
  `ToolCallOutput(tool_choice="required")`) so users understand the
  cost model (one extra inference per route decision) and the
  inheritance rules (`function_name` and `t0` come from the first
  leaf Agent in the route graph).
- **Document the typed-MCP surface in the tutorial.**  Both patterns
  (class-based + Annotated) are usable; mention `App(servers=[...])`
  as the canonical wiring; cross-reference `examples/glow/06_mcp_typing.py`.
  Document the HTTP path via `create_proxy(url)` and call out the
  per-agent-MCP-subsets limitation.
- **Document the `Annotated[T, Guardrail(...)]` pattern** including
  the `regex_guardrail` helper, the input vs output phase distinction,
  and how to combine with existing `t0_agents.guardrails` factories
  (`max_cost`, `max_tool_calls`, `block_tools`, `require_approval`,
  `require_parsed`) via `Annotated[T, max_cost(0.5), block_tools(["x"])]`.
- **`examples/glow/03_streaming.py` is still pending.**  M2 dev-log
  noted this; M4 didn't address it.  M5 polish should ship it
  alongside the cookbook.
- **Verify integration tests pass against a live gateway.**  The M4
  integration suite (6 tests; 3 inherited + 3 new) was validated to
  collect cleanly but not yet run live by Claude.  The user must
  `docker compose up -d` + `uv run pytest -m integration -q` before
  pushing the M4 PR (per the M4 user manual checklist).

### Post-M5

- **Consider exposing `_LLMRoutedComposer` publicly** if the
  dict-router DX proves popular.  Power users may want to construct
  the composer with a pre-built classifier (e.g. a cheaper model
  variant) or override the system prompt.
- **Consider `dict[str, list[Composable]]` for branch fan-out.**  A
  natural extension if users routinely want a route to fan out into
  multiple subordinates.
- **Cache the classifier Client on first invocation** of a long-lived
  `_LLMRoutedComposer` to amortise the transport setup cost.  Would
  need a context-manager shape or finalizer for cleanup.

### For the polish backlog (`project_pre_public_polish.md`)

- The M0 race-coherence items (4 references in `docs/composition.md`,
  README, migration doc) are still outstanding.  M4 did not touch
  them.
- M3 §4 noted that `05_di_tools.py` exercises only the MCP
  round-trip, not the agent-uses-tool path; a future
  `examples/glow/07_agent_with_tools.py` would close that gap.

## §5 Verification

Captured commands and outcomes (run at the end of M4 implementation,
prior to writing this dev-log):

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/glow/ tests/test_glow/
All checks passed!

$ uv run mypy src/t0_agents/glow/ tests/test_glow/
Success: no issues found in 23 source files

$ uv run pytest tests/test_glow/ -q -m "not integration"
......................................................................   [ 61%]
..............................................                          [100%]
118 passed, 6 deselected in 0.44s

$ uv run pytest -m "not integration" -q
[...]
1002 passed, 2 skipped, 63 deselected in 8.53s
# Pre-M4 baseline (end of M3): 944 passing.  M4 adds 58 new offline
# tests (+4 output_inference + 18 compose + 18 mcp_typing + 16
# guardrails_typing + 2 elsewhere); total = 1002.  No regressions.

$ uv run pytest tests/test_glow/test_integration.py --collect-only -q
tests/test_glow/test_integration.py::test_hello_world_chat_agent_runs_end_to_end
tests/test_glow/test_integration.py::test_structured_output_returns_pydantic_instance
tests/test_glow/test_integration.py::test_orchestrator_with_glow_sub_agents
tests/test_glow/test_integration.py::test_llm_routed_dict_composer_dispatches_to_picked_route
tests/test_glow/test_integration.py::test_llm_routed_dict_composer_supports_composer_routes
tests/test_glow/test_integration.py::test_regex_guardrail_on_return_annotation_blocks_response

6 tests collected in 0.16s
# 3 inherited (M2 + M3) + 3 new (M4: 2× LLM router + 1× regex guardrail).
# Live execution against `docker compose -f
# examples/glow/docker-compose.yml --env-file .env up -d` is the
# user's pre-push step per the M4 user manual checklist.

$ git diff src/t0_agents/mcp/ | wc -l
0
# (mcp/ standalone-submodule invariant preserved per feedback_mcp_standalone)

$ git log --oneline experimental/glow ^main | head -8
c711499 chore(glow): document M4 in CHANGELOG
eff570b feat(glow): 04_composition + 06_mcp_typing examples
717293d feat(glow): Annotated[T, Guardrail(...)] extraction + regex_guardrail
a50d1f4 feat(glow): typed MCPServer integration
caafb89 feat(glow): LLM-routed dict composer
1d414c3 feat(glow): compose() native-type dispatcher
fd50b1c fix(glow): iterative Annotated/Optional peel in output_inference
d21f88c Merge pull request #40 from sproutfi/glow/m3-di-and-tools
# The 7 M4 code/example/CHANGELOG commits, all with the (glow) scope.
# This dev-log commit will bring the total to 8.

$ git diff --stat origin/main..HEAD | tail -1
 59 files changed, 8949 insertions(+), 116 deletions(-)
# All changes confined to the allowed M4 paths
# (src/t0_agents/glow/**, tests/test_glow/**, examples/glow/**,
# docs/internals/glow/**, CHANGELOG.md) plus the inherited M2 D5
# core touch (src/t0_agents/core/agent.py + tests/test_core/test_agent.py).
# No new collateral damage in M4.

$ uv run python -c "
from t0_agents.glow import App, MCPServer, compose, regex_guardrail
print('compose:', compose)
print('MCPServer:', MCPServer)
print('regex_guardrail:', regex_guardrail)
"
compose: <function compose at ...>
MCPServer: <class 't0_agents.glow.mcp_typing.MCPServer'>
regex_guardrail: <function regex_guardrail at ...>
# End-to-end smoke confirms every new M4 public symbol imports
# cleanly from the package root.  ``MCPServer`` is the dual-personality
# base class; ``regex_guardrail`` is the glow-private convenience
# factory; ``compose`` is the native-collection dispatcher.
```

The full pre-push `uv run pytest -q` (no marker filter) is the user's
responsibility per `feedback_full_test_suite_before_push`; the 6
integration tests must be validated against a live gateway before the
M4 PR is opened.  The user checklist in
`/home/makarenko/.claude/plans/4-squishy-phoenix.md` enumerates the
exact pre-push steps.
