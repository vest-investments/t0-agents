# M2 — `core-decorator` — dev-log

> **Milestone:** `milestone-2-core-decorator` ([spec](../milestones/milestone-2-core-decorator.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-17.

## §1 Summary — what was done

M2 lands the headline DX of `t0_agents.glow`: an `App` class and an
`@app.agent` decorator that compile to v1.0 `Agent` + `AgentConfig`,
plus first-class event selectors over the v1.0 event hierarchy.  The
plan file's four-line hello-world (Example 1) and structured-output
example (Example 2) now run end-to-end against a live TensorZero
gateway.

The milestone also unlocks a general v1.0 core improvement:
chat-mode agents may now omit `tool_provider` entirely.  The chat
terminals allocate an ephemeral in-memory `FastMCP("agent-default")`
for the invocation's duration when the caller passed `None` — the
LLM produces a text response and the loop terminates after the first
turn.  This was a user-approved scope expansion (deviation D5 below)
because the alternative — auto-allocating the same fallback inside
`App` — would have been a glow-only band-aid that didn't help any
other consumer of `Agent`.

### Files touched

| Path | Change | Lines |
|---|---|---|
| `src/t0_agents/core/agent.py` | Drop `ConfigurationError` validation block for chat-mode + `tool_provider=None`; add `_resolve_chat_tool_provider` `@asynccontextmanager` helper; refactor `_chat_terminal` + `_chat_streaming_terminal` to use it; update class docstring l. 52. | +37 / −37 |
| `tests/test_core/test_agent.py` | Adapt `test_chat_agent_without_tool_provider_raises` → `test_chat_agent_without_tool_provider_constructs`; add `test_toolless_chat_invoke_returns_text` exercising the end-to-end fake-Run path through `FakeT0`. | +21 / −10 |
| `pyproject.toml` | Add `glow = ["pydantic>=2.0"]` under `[project.optional-dependencies]`. | +3 / −0 |
| `uv.lock` | Regenerate for the `[glow]` extra. | +5 / −1 |
| `src/t0_agents/glow/_spec.py` | New — `_AgentSpec` dataclass (slots=True, frozen=False so the `agent: Agent | None` slot can be mutated on first build). | +43 / −0 |
| `src/t0_agents/glow/app.py` | New — `App` class with `__init__(t0=...)`, `agent(name, *, function_name)`, `_lookup_or_build`, `_invoke`, `build()`, `lifespan()`.  Public `dependency_overrides: dict` stub for M3.  ParamSpec/TypeVar on `agent` so static checkers preserve the wrapped function's signature. | +179 / −0 |
| `src/t0_agents/glow/decorators.py` | New — `_AgentWrapper[P, R]` Generic class with `__call__(*args: P.args, **kwargs: P.kwargs) -> FluentRun[R]`, `._agent` / `._spec` escape-hatch properties, and `_make_decorator(app, name, function_name)` factory.  Decoration-time enforcement of duplicate names + forward-ref resolution. | +182 / −0 |
| `src/t0_agents/glow/handle.py` | Extend — keyword-only `value_transform: Callable[[Any], T] \| None = None` constructor argument; `result()` applies the transform when no per-call override is active (constructs a fresh `AgentResult` preserving every other field); `events(*selectors)` now delegates to a new `_filter_with_selectors` module-level async generator implementing first-match-wins isinstance dispatch. | +73 / −10 |
| `src/t0_agents/glow/selectors.py` | New — `Selector[E]` generic with `.where()` (AND-chains predicates) and `.map()` (replaces transform); module-level instances `Thought`, `ToolCall`, `Text` (=`TextEndEvent`), `ToolResult`, `TurnEnd`, `Done`, `Error`. | +140 / −0 |
| `src/t0_agents/glow/__init__.py` | Re-exports grow from `["FluentRun"]` → `["App", "Done", "Error", "FluentRun", "Selector", "Text", "Thought", "ToolCall", "ToolResult", "TurnEnd"]`. | +20 / −2 |
| `tests/test_glow/test_app.py` | New — 11 unit tests with a patched `_FakeAgent` covering registration, lazy build, build()-idempotency, duplicate-name detection, forward-ref failure, list-of-Pydantic unwrap, wrapper escape hatches, ParamSpec typing via `typing.assert_type`, and strict M2 kwarg rejection. | +337 / −0 |
| `tests/test_glow/test_selectors.py` | New — 10 unit tests over a fake `_StreamFakeRun`; cover each module-level selector, `.where()` AND-chaining, `.map()` replacement, multi-selector union with first-match-wins, the empty-selectors smoke (M1 parity), and the bare-constructor passthrough defaults. | +184 / −0 |
| `tests/test_glow/test_integration.py` | New — 2 integration tests against the live gateway: hello-world chat (toolless) and structured-output Pydantic return.  Module-level `pytestmark = pytest.mark.integration`. | +79 / −0 |
| `tests/test_glow/test_fluent_run.py` | Tiny cleanup — drop a stale `# type: ignore[unreachable]` that became unused once mypy started seeing tests/test_glow/. | +1 / −1 |
| `tests/test_glow/test_output_inference.py` | Tiny cleanup — three `# type: ignore[empty-body]` markers on test stubs so mypy strict is happy across the whole `tests/test_glow/` directory. | +3 / −3 |
| `examples/glow/01_hello.py` | New — runnable 4-line hello-world demo. | +44 / −0 |
| `examples/glow/02_structured.py` | New — runnable structured-output demo with `TripPlan(BaseModel)`. | +49 / −0 |
| `examples/glow/config/tensorzero.toml` | New — gateway config with `[functions.agent]` chat function (mirrors `examples/02_agent_class`) and `[functions.plan_trip]` JSON-strict function for the structured example. | +46 / −0 |
| `examples/glow/config/functions/plan_trip/output_schema.json` | New — JSON Schema matching `TripPlan` exactly. | +17 / −0 |
| `examples/glow/config/functions/plan_trip/system_template.minijinja` | New — short prompt with two few-shot examples. | +13 / −0 |
| `examples/glow/docker-compose.yml` | New — mirror of `examples/02_agent_class/docker-compose.yml` pointing at `examples/glow/config/`. | +47 / −0 |
| `examples/glow/README.md` | Update from the M1 placeholder to a 6-row table with milestone numbers + boot instructions. | +30 / −14 |
| `CHANGELOG.md` | Extend the `Experimental — t0_agents.glow` subsection with 5 M2 bullets and add a v1.0 `### Changed` opener for the toolless-chat unlock. | +57 / −0 |

### Public glow API delta

`t0_agents.glow.__all__` grew from `["FluentRun"]` (M1) → 10 entries:
`["App", "Done", "Error", "FluentRun", "Selector", "Text", "Thought",
"ToolCall", "ToolResult", "TurnEnd"]`.  The `@app.agent` decorator
itself is reached as a *method* on `App`, not as a top-level import.

The internal `_AgentSpec` and `_AgentWrapper` stay underscore-prefixed
— users degrade to the base API via `wrapper._agent` rather than
constructing the spec directly.

### v1.0 core API delta

`Agent(config, t0=t0)` (no `tool_provider`) is now valid for every
output mode, including chat.  This is a strict superset of the prior
contract — every call site that previously worked still does.

### Commits

The milestone landed as eight commits on `experimental/glow`, in
order:

| # | SHA | Subject |
|---|---|---|
| 1 | `ee030bc` | `feat(agent): make tool_provider optional for chat-mode agents` |
| 2 | `7abf062` | `chore(glow): add [glow] pyproject extra` |
| 3 | `00e59c6` | `feat(glow): App + @app.agent decorator (chat + structured)` |
| 4 | `8b27480` | `feat(glow): event selectors with where/map/filter dispatch` |
| 5 | `2276d03` | `feat(glow): runnable hello-world + structured examples + TZ config` |
| 6 | `f350a0d` | `feat(glow): integration tests for App + @app.agent` |
| 7 | `2db6ffa` | `chore(glow): document M2 in CHANGELOG` |
| 8 | (this commit) | `docs(glow): milestone-2 dev-log` |

In the 14-commit roadmap from the plan file these correspond to
commits 4 + 5 of the original numbering, plus the cross-cutting
toolless-chat unlock (an opportunistic core fix made possible by the
user's explicit scope expansion).

### Tests

| File | New tests |
|---|---|
| `tests/test_core/test_agent.py` | 1 (the existing chat-rejects test was adapted in-place to assert the new construct-succeeds behaviour; the new `test_toolless_chat_invoke_returns_text` is a net add) |
| `tests/test_glow/test_app.py` | 11 |
| `tests/test_glow/test_selectors.py` | 10 |
| `tests/test_glow/test_integration.py` | 2 (`@pytest.mark.integration`; deselected by the offline filter) |

Offline regression: **900 (end of M1) → 922 passing.**  The 22 new
tests run in well under a second; no docker required for the
offline gate.  The 2 integration tests need
`docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`
before `uv run pytest -m integration -q` will collect them.

## §2 Deviations from the milestone spec

Six recorded deviations.  D5 is the load-bearing one — the rest
were either spec-anticipated scope refinements (D1, D2) or
mechanical mismatches between the spec text and on-disk reality
(D3, D4).  D6 is a M1 carry-forward we honored.

### 2.1 — D1: `App(servers=...)` kwarg deferred to M4

The spec's `App.__init__` signature was:

```text
App(t0: AsyncTensorZeroGateway, *, servers: list[type[MCPServer]] | None = None)
```

with a note that `servers` is "wired in M4".  Shipping an unwired
parameter — and a corresponding `MCPServer` placeholder class with
no implementation — is misleading: callers would discover the
silently-ignored kwarg, and the M4 implementation would have to
worry about backward compatibility with whatever stub shape M2 left
behind.  We dropped the parameter entirely; M2's `App.__init__` is
just `def __init__(self, *, t0: AsyncTensorZeroGateway) -> None`.
M4 will reintroduce `servers=` with the full typed-MCP wiring at
once.

**Impact on subsequent milestones:** M4 owns the `App.__init__`
signature change.  Forward-compat is trivial because we never
shipped the placeholder.

### 2.2 — D2: `App.lifespan()` actually builds, not just yields

The spec said the `lifespan()` async context manager "currently just
yields (M4 will add MCP teardown)".  We made it call `await
self.build()` before yielding because:

1. `build()` is the v1.0 fail-fast hook for configuration errors —
   delaying it to first invocation defeats the purpose.
2. `async with app.lifespan(): ...` is the natural place to surface
   forward-ref / duplicate-name errors.
3. The cost is zero on the happy path (the spec for M2 already
   constructs every agent at most once thanks to the spec-level
   `agent: Agent | None` cache).

**Impact on subsequent milestones:** M4 must add MCP `mount` /
`unmount` *inside* `lifespan()` around the `await self.build()` call
+ `yield`.  This is mechanical; the structural choice in M2 doesn't
constrain M4.

### 2.3 — D3: structured example ships its own TZ config

The M2 spec referenced `examples/17_structured/` for the structured
TZ function backing `02_structured.py`.  That directory does not
exist in the repo — the actual structured TZ function
(`extract_customer_info`) lives in `examples/03_output_modes/` and
uses a `Customer(BaseModel)` schema, not the plan file's `TripPlan`.

Per the user's resolution: ship a self-contained
`examples/glow/config/tensorzero.toml` with both functions (`agent`
chat + `plan_trip` JSON-structured) and a `TripPlan`-matching schema
under `examples/glow/config/functions/plan_trip/`.  Every artefact
stays inside the allowed `examples/glow/**` paths; the example
matches the plan file verbatim.  `docker compose` switches to
`-f examples/glow/docker-compose.yml --env-file .env`.

**Impact on subsequent milestones:** M5 inherits the precedent —
glow examples own their TZ config under `examples/glow/`, and the
M5 cookbook can extend the same `tensorzero.toml` with any
additional functions needed for the 03–06 examples.

### 2.4 — D4: `Text = Selector(TextEndEvent)` confirmed

The spec's checklist line "Text = Selector(TextEndEvent) (whole-text
variant is more useful as a selector default than the delta variant;
users can pick Text.where(...) for delta filters)" was already a
decision in the spec — we confirmed and adopted it exactly.  Listed
here only to make the dev-log inventory complete; not an actual
divergence.

### 2.5 — D5: v1.0 core touched in this PR (user-approved)

The biggest deviation.  The plan-file invariant is that M2 is
glow-additive — every milestone except M0 must leave everything
outside the glow surface untouched.  The plan's hello-world,
however, infers `ChatOutput` from `-> str`, and `Agent.__init__`
rejected chat agents constructed with `tool_provider=None` at
`core/agent.py:113` (the existing validation).  Three resolutions
were possible:

1. Auto-allocate a fallback inside `App.__init__` (glow-only band-aid).
2. Require the user to pass their own MCP client (kills the 4-line
   DX promise).
3. Relax the core rule and benefit every `Agent` consumer.

User chose (3) explicitly during planning ("si quiero (b) aunque se
tenga que tocar el core").  Implementation:

- Drop the `ConfigurationError` block at `core/agent.py:113-120`.
- Add a module-level `_resolve_chat_tool_provider` async context
  manager that yields the caller's provider unchanged or — when
  `None` — an ephemeral `Client(FastMCP("agent-default"))` for the
  invocation's duration.
- Refactor both `_chat_terminal` and `_chat_streaming_terminal` to
  use it; the chat loop sees an empty tool list, the LLM produces
  text, and the loop terminates after one turn.
- `src/t0_agents/mcp/` is unchanged (the standalone-submodule
  invariant per `feedback_mcp_standalone` is preserved — we did not
  touch `mcp/loop.py:run_agent`'s signature).
- One existing test (`test_chat_agent_without_tool_provider_raises`)
  is adapted in-place to assert the new constructs-succeeds
  behaviour; a new `test_toolless_chat_invoke_returns_text` exercises
  the end-to-end flow through `FakeT0` + a real in-memory FastMCP.

The CHANGELOG entry lives under v1.0 `### Changed` (not under
`Experimental — t0_agents.glow`) because every `Agent` consumer
benefits from the relaxation.  The standalone-submodule contract
holds; the `mcp/` diff against `origin/main` is empty.

**Impact on subsequent milestones:** M4 has slightly more freedom —
when typed `MCPServer` wiring lands, the fallback `FastMCP` can be
the same in-memory server that `FastMCP.mount(...)` extends.  M5
should cross-reference the toolless-chat behaviour in `docs/agent.md`
(added to `project_pre_public_polish.md`).  No M3 carry-forward
impact.

### 2.6 — D6: `FluentRun(value_transform=...)` added

M1 dev-log §4(d) carry-forward called for `list[BaseModel]` returns
to be unwrapped to `list[T]` "inside `@app.agent`'s wrapper
coroutine".  Cleanest path was a small additive extension of
`FluentRun`: a keyword-only `value_transform=` constructor argument
applied inside `result()` when no per-call override is active.
Per-call overrides (`.text` / `.json` / `.as_`) reset the transform
via the existing `_spawn` path (the new `FluentRun(self._build,
new_output)` instance defaults `value_transform=None`).

**Impact on subsequent milestones:** M3's `Depends(agent)` sub-agent
case writes into `AgentConfig.sub_agents`, not into
`value_transform` — these are orthogonal hooks.  M4's guardrail-on-
return-type pattern (`Annotated[T, Guardrail(...)]`) goes through
`AgentConfig.guardrails`; again no overlap.

## §3 Open questions

### 3.1 — Should `_resolve_chat_tool_provider` be public?

The toolless-chat fallback uses an internal module-level
`@asynccontextmanager`.  Users who want to construct *their own*
empty `FastMCP` plus `Client` boilerplate could reuse the helper
directly, but exposing it would lock the public surface around an
"agent-default" string and the precise lifecycle semantics.  M2
ships it as `_resolve_chat_tool_provider` (underscored) — if the
public-release polish pass wants to elevate it, that's a M5 call.
Parked.

### 3.2 — Should `_AgentWrapper.__call__` accept extra kwargs eagerly?

M2 rejects any kwargs and any extra positional arguments on the
wrapper.  This is the strictest possible M2 surface, and it forces
M3 to do real work (the `Depends(...)` introspection lands in M3
commit 6 per the roadmap).  Alternative: accept kwargs as a
permissive "M3 will use these" gesture — but that risks silently
ignoring a typo'd kwarg in user code today.  Strict wins; M3 lifts
it cleanly when the signature walker arrives.

### 3.3 — `# type: ignore[empty-body]` on test stubs

Multiple test stubs (`async def tutor(q: str) -> str:` with a
docstring and no body) trip mypy strict's `empty-body` check.  We
added `# type: ignore[empty-body]` per stub; the alternative would
be a per-file `[mypy] disallow_empty_body = false` override or a
runtime dummy return like `return ""`.  Inline `type: ignore` is
the closest match to M1's pattern (`UP037` / `UP045` noqa per site)
and keeps the override scope tight.

### 3.4 — Pre-existing ruff hits in `tests/`

`uv run ruff check src/ tests/` still reports 6 pre-existing errors
in `tests/test_cli/`, `tests/test_memory/`, `tests/test_core/test_agent_*`
that predate M2 (already documented in
`project_pre_public_polish.md`).  M2 did **not** touch these and
did **not** add any new error.  No action needed for M2.

No new questions surfaced from the M0 or M1 dev-logs' §3 / §4 that
M2 was the right place to resolve.  M1 §3.1 (`Annotated[Optional[T],
meta]` peel order) stays parked for M4.

## §4 Carry-forward notes

### For M3 (`di-and-tools`)

- **`_AgentWrapper.__call__` strict-arg rejection must be relaxed.**
  M2 raises `TypeError` for any kwargs or extra positionals; M3's
  `Depends(...)` introspection requires walking `inspect.signature(fn)`
  and matching every parameter (positional + keyword + default) to
  either the user-supplied message OR an injected dependency.  Replace
  the strict block in `decorators.py:_AgentWrapper.__call__` with the
  full sig walk; the first positional `str`-typed parameter still
  becomes the user message.
- **`Depends(agent)` writes into `_AgentSpec.sub_agents`, not
  `value_transform`.**  The introspective-magic case in the plan file
  installs the dependency as a sub-agent on the parent
  `AgentConfig.sub_agents=[...]` rather than as an injected runtime
  value.  M2's `_AgentSpec` does not have a `sub_agents` slot — add
  it then.  The existing `value_transform=` hook is independent and
  must not be repurposed.
- **`@app.tool` registration may want to share the in-memory
  `FastMCP` used by the toolless-chat fallback.**  Today the fallback
  in `_resolve_chat_tool_provider` allocates a fresh
  `FastMCP("agent-default")` per chat invocation.  M3's `@app.tool`
  registers user tools into the App-owned FastMCP; the chat
  terminals should then see those tools advertised.  Two paths: (a)
  App owns a long-lived in-memory `FastMCP` + `Client` whose
  lifecycle is managed by `app.lifespan()`, and the `_resolve_chat_tool_provider`
  helper switches to "yield app._owned_client" when in glow; (b)
  glow's invoke wires `tool_provider=app._owned_client` into
  `Agent(config, t0=t0, tool_provider=...)` explicitly so the
  fallback is never reached.  M3 picks one; (b) is probably cleaner.

### For M4 (`composition-mcp-guardrails`)

- **`App.__init__` grows `servers=list[type[MCPServer]] | None = None`.**
  Per deviation D1, M2 deliberately omitted the kwarg.  M4 introduces
  it with the full typed-MCP wiring and updates `App.lifespan()` to
  call `FastMCP.mount(...)` on the owned in-memory server inside the
  build/teardown cycle.
- **`Annotated[T, Guardrail(...)]` peel order** — see M1 dev-log
  §3.1.  M2 did not encounter the case because the `@app.agent`
  decorator only walks the *return* annotation, never parameter
  annotations.  M4's guardrail-on-input pattern will trip
  `_strip_annotated`/`_unwrap_optional` peel order under
  `Annotated[Optional[T], meta]`; refactor the helpers in `_typing.py`
  to either iterative peel or inverted order.
- **`compose(set[Composable])` must use `parallel(*, reducer=...,
  error_policy="continue")`** — anchor from M0 dev-log §4.

### For M5 (`polish`)

- **Pre-public-polish backlog has new items:**
  - The M2 docstring change to `core/agent.py:52` should be
    cross-referenced from `docs/agent.md` once that file is part of
    the public docs sweep.  Logged separately so M5 doesn't lose
    sight of it.
  - The toolless-chat behaviour should get a one-paragraph
    explanation in the user-facing glow tutorial: "no MCP plumbing
    required for chat-only agents".
- **Examples 03–06 still pending** — `03_streaming.py`,
  `04_composition.py`, `05_di_tools.py`, `06_mcp_typing.py`.
  `examples/glow/config/tensorzero.toml` already declares one chat
  function and one JSON function; the others can extend the same
  config or add new function blocks (no new compose file needed).
- **`test_assert_type_of_call_matches_return_annotation`** uses
  `typing.assert_type` which is a runtime no-op.  The real signal is
  `uv run mypy` on the test file — make sure the public CI invocation
  M5 introduces runs mypy over `tests/test_glow/` (not just
  `src/t0_agents/glow/`).

### For the polish backlog (`project_pre_public_polish.md`)

The four pre-existing race-deprecation references called out by the
M0 dev-log §3 are still outstanding.  M2 did not touch them.

## §5 Verification

Captured commands and outcomes (run at the end of M2 implementation,
prior to writing this dev-log):

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/glow/ tests/test_glow/ src/t0_agents/core/agent.py tests/test_core/test_agent.py
All checks passed!

$ uv run mypy src/t0_agents/glow/ tests/test_glow/
Success: no issues found in 13 source files

$ uv run mypy src/t0_agents/core/agent.py
Success: no issues found in 1 source file

$ uv run mypy src/
Success: no issues found in 74 source files

$ uv run pytest tests/test_glow/ -q -m "not integration"
......................................                                   [100%]
38 passed, 2 deselected in 0.13s

$ uv run pytest -m "not integration" -q
[...]
922 passed, 2 skipped, 59 deselected in 4.63s
# Pre-M2 baseline (end of M1): 900 passing.  M2 adds 22 new tests
# (1 in test_core/test_agent.py + 11 in test_glow/test_app.py + 10
# in test_glow/test_selectors.py); total = 922.  No regressions.

$ uv run pytest tests/test_glow/test_integration.py --collect-only -q
tests/test_glow/test_integration.py::test_hello_world_chat_agent_runs_end_to_end
tests/test_glow/test_integration.py::test_structured_output_returns_pydantic_instance

2 tests collected in 0.09s
# Integration tests collect cleanly; user runs them after
# `docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`.

$ git diff src/t0_agents/mcp/ | wc -l
0
# (mcp/ standalone-submodule invariant preserved per
# feedback_mcp_standalone)

$ git log --oneline experimental/glow ^main | wc -l
19
# 5 scaffold+M0 + 4 M1 + 2 M0-PR-merge artefacts + 8 M2 commits = 19.
# The 8 M2 commits map to the spec's commit 4 + commit 5 of the
# 14-commit roadmap, plus the cross-cutting toolless-chat unlock and
# the supporting infrastructure (pyproject extra, examples + TZ
# config, integration tests, CHANGELOG, this dev-log).

$ git log --oneline experimental/glow ^main | head -8
2db6ffa chore(glow): document M2 in CHANGELOG
f350a0d feat(glow): integration tests for App + @app.agent
2276d03 feat(glow): runnable hello-world + structured examples + TZ config
8b27480 feat(glow): event selectors with where/map/filter dispatch
00e59c6 feat(glow): App + @app.agent decorator (chat + structured)
7abf062 chore(glow): add [glow] pyproject extra
ee030bc feat(agent): make tool_provider optional for chat-mode agents
d98fd81 Merge pull request #38 from sproutfi/glow/m1-foundations
# The 7 M2 code/CHANGELOG commits, all with the (glow) or (agent)
# scope.  The dev-log commit (this one) will bring the total to 8.

$ git diff --stat origin/main..HEAD | tail -3
 ...
 42 files changed, 4350 insertions(+), 116 deletions(-)
# Allowed paths: src/t0_agents/glow/**, tests/test_glow/**,
# examples/glow/**, docs/internals/glow/**, CHANGELOG.md,
# pyproject.toml, uv.lock — plus the D5 core exception
# (src/t0_agents/core/agent.py + tests/test_core/test_agent.py).
# No collateral damage outside that set.

$ uv run python -c "
from t0_agents.glow import App, FluentRun, Thought, ToolCall, Text
print('App:', App)
print('FluentRun:', FluentRun)
print('Selectors:', Thought, ToolCall, Text)
"
App: <class 't0_agents.glow.app.App'>
FluentRun: <class 't0_agents.glow.handle.FluentRun'>
Selectors: <t0_agents.glow.selectors.Selector object ...> ...
# End-to-end smoke confirms every M2 public symbol imports cleanly
# from the package root.
```

The full pre-push `uv run pytest -q` (no marker filter) is the
user's responsibility per `feedback_full_test_suite_before_push`;
integration runs require
`docker compose -f examples/glow/docker-compose.yml --env-file .env up -d`.
The two integration tests in `tests/test_glow/test_integration.py`
were validated to collect successfully; live execution against the
TZ gateway is part of the M2 acceptance gate the user runs before
pushing the PR.
