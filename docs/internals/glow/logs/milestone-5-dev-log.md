# M5 — `polish` — dev-log

> **Milestone:** `milestone-5-polish` ([spec](../milestones/milestone-5-polish.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-17.

## §1 Summary — what was done

M5 closes the glow roadmap.  The branch is now PR-ready: every
public symbol is documented in the user-facing tutorial, every
deferred surface is enumerated in `limitations.md`, the line-count
comparison clears the spec's ≥40% reduction target, and the
top-level repo `README.md` points new readers at the experimental
surface.  The only v1.0 core change in M5 is the planned
`regex_guardrail` migration from glow-private to canonical
namespace — a strict superset of the prior surface (existing glow
imports keep working via re-export).

The glow public surface (14 symbols) is **frozen by M5** — no new
exports land here.  The migration just relocates `regex_guardrail`'s
canonical home; the symbol stays reachable from both
`t0_agents.glow` and `t0_agents.guardrails`.

### Files added

| Path | Lines | Role |
|---|---|---|
| `src/t0_agents/guardrails/regex.py` | 124 | Canonical home for `regex_guardrail` + `_stringify_result_value`. |
| `tests/test_guardrails/test_regex.py` | 96 | Ported regex tests + a new precompiled-pattern case. |
| `examples/glow/03_streaming.py` | 90 | Chat agent + `@app.tool` + 3-selector live event loop. |
| `docs/glow/getting-started.md` | 413 | User-facing tutorial — 14 sections covering every public symbol. |
| `docs/glow/limitations.md` | 178 | Known limitations + escape-hatch instructions. |
| `docs/internals/glow/comparison.md` | 188 | Reproducible code-only LOC comparison. |
| `docs/internals/glow/logs/milestone-5-dev-log.md` | (this file) | Always-write dev-log. |

### Files modified

| Path | Change |
|---|---|
| `src/t0_agents/guardrails/__init__.py` | Re-export `regex_guardrail`; doc-bullet update. |
| `src/t0_agents/glow/guardrails_typing.py` | Collapse `regex_guardrail` to a re-export of the v1.0 module; drop the M5-carry-forward docstring paragraph and the now-unused `_stringify_result_value` helper. |
| `tests/test_glow/test_guardrails_typing.py` | Trim the 9 regex-direct-hook tests now covered at the v1.0 home; keep the 6 Annotated-extraction + `@app.agent` integration tests. |
| `examples/glow/README.md` | Un-stub the `03_streaming.py` row; mark it M5. |
| `README.md` (top-level) | Insert a small "Experimental: `t0_agents.glow`" section between Documentation and Requirements. |
| `CHANGELOG.md` | New v1.0 `### Added` sub-section `#### Guardrails — regex content blocker` for the migrated factory; six new bullets under `#### Experimental — t0_agents.glow` for the M5 artefacts. |

### Public glow API delta

`t0_agents.glow.__all__` is **unchanged** from M4 — 14 entries:
`["App", "Depends", "Done", "Error", "FluentRun", "MCPServer",
"Selector", "Text", "Thought", "ToolCall", "ToolResult", "TurnEnd",
"compose", "regex_guardrail"]`.  The `regex_guardrail` symbol is now
backed by the v1.0 module rather than a glow-local
implementation; the importable name stays the same.

### v1.0 core API delta

`t0_agents.guardrails.regex_guardrail` is newly importable from the
canonical namespace (and the `t0_agents.guardrails.regex` module).
The factory is a strict superset of the prior glow-only behaviour —
no existing call site changes shape.

### Commits

| # | SHA | Subject |
|---|---|---|
| 1 | `0afcf0e` | `feat(guardrails): regex helper for input/output validation` |
| 2 | `a2a553b` | `refactor(glow): re-export migrated regex_guardrail from v1.0` |
| 3 | `f7eda37` | `feat(glow): 03_streaming example with event selectors` |
| 4 | `58bc526` | `docs(glow): user-facing tutorial — getting-started.md` |
| 5 | `ed5731a` | `docs(glow): known limitations — limitations.md` |
| 6 | `41902f0` | `docs(glow): line-count comparison vs base API` |
| 7 | `7359db5` | `docs(readme): point at experimental glow tutorial` |
| 8 | `6d606bc` | `chore(glow): document M5 in CHANGELOG` |
| 9 | (this commit) | `docs(glow): milestone-5 dev-log` |

Nine commits, all under the conventional-commit `(glow)` /
`(guardrails)` / `(readme)` scopes.  The 14-commit roadmap from the
plan file labels M5 as commits 13 + 14; the nine-commit split is
internal to M5 and follows the multi-commit-PR convention from
`CLAUDE.md` and `feedback_multi_commit_prs`.

### Tests

| File | Net delta |
|---|---|
| `tests/test_guardrails/test_regex.py` | +10 new tests (9 ported from glow + 1 new precompiled-pattern case). |
| `tests/test_glow/test_guardrails_typing.py` | −9 ported tests (the 6 Annotated-extraction + `@app.agent` integration tests stay). |

Offline regression: **1002 (M4 end) → 1003 passing** (+1 net new
test — the precompiled-pattern path).  Glow offline subset:
**118 (M4 end) → 109** (-9, because the regex tests are now under
`tests/test_guardrails/`).  Integration suite: unchanged at 6 tests
(the migration moves nothing in the integration coverage).

## §2 Deviations from the milestone spec

Three deviations, all anticipated at planning time (`5-starry-spring.md`
captured each as a resolved decision before the first edit).

### 2.1 — D1: `regex_guardrail` migrated to v1.0 core in M5

The M4 dev-log §4 carried this forward as a recommendation rather
than a hard requirement.  The user approved migration at M5 planning
time (D1 = migrate); the rationale matches the M2 D5 toolless-chat
precedent (touches v1.0 core, benefits every `Agent` consumer, adds
a `### Added` v1.0 CHANGELOG bullet, requires a strict-superset
guarantee for existing glow imports).

Implementation:

- `src/t0_agents/guardrails/regex.py` (new, 124 lines) — verbatim
  copy of the glow factory minus the "M5 carry-forward" docstring
  paragraph.
- `tests/test_guardrails/test_regex.py` (new, 96 lines) — 9 ported
  + 1 new precompiled-pattern test.
- `src/t0_agents/glow/guardrails_typing.py` collapses
  `regex_guardrail` to `from t0_agents.guardrails import
  regex_guardrail`; the now-unused `_stringify_result_value` helper
  is deleted from glow (the v1.0 module owns it).
- `tests/test_glow/test_guardrails_typing.py` trims the 9 ported
  tests; keeps the 6 glow-specific Annotated-extraction tests.
- CHANGELOG: a new `#### Guardrails — regex content blocker`
  sub-section under v1.0 `### Added` (commit 1); a cross-link bullet
  under `#### Experimental — t0_agents.glow` (commit 8).

**Impact on the post-M5 PR:**  the PR description should highlight
the v1.0 core change separately from the experimental glow
additions — it's a strict superset, but it's the only v1.0 core
behaviour change in this milestone (the rest are docs + one
example).  Reviewers should look at it the same way they looked at
the M2 D5 toolless-chat unlock — the test suite proves the contract
is preserved and the canonical home is now self-documenting.

### 2.2 — D2: code-only LOC methodology, not raw `wc -l`

Pre-M5 raw `wc -l` for `examples/glow/01_hello.py` vs
`examples/02_agent_class/basic.py` posted 49 / 65 = **24.6%
reduction** — below the spec's ≥40% threshold.  The gap was
bloated by docstrings + prerequisite blocks both sides ship for
usability.

Resolution chosen during planning (user-confirmed D2): report both
raw and code-only counts.  The code-only filter strips blank
lines, `#`-prefixed comment lines, and the leading module
docstring (the first triple-quoted block after the optional
`from __future__` import).  Applied identically to both files, it
yields 22 / 38 = **57.9% of baseline → 42.1% reduction**, clearing
the spec target.

`docs/internals/glow/comparison.md` embeds the Python snippet that
produces the table along with the verbatim expected output, so the
result is reproducible on a clean checkout.

**Impact on subsequent work:**  none.  If `01_hello.py` or
`basic.py` change, the documented methodology lets a future
maintainer refresh the table mechanically.  If the spec is ever
reinterpreted to require raw-`wc -l` reduction specifically, the
example would need tightening — but that interpretation discards
the visible-comment overhead inherent to runnable Python files.

### 2.3 — Path-allowance exception for `docs/glow/**` + top-level `README.md`

The `/glow-milestone` command's default allowed-paths set is the
five glow paths (`src/t0_agents/glow/**`, `tests/test_glow/**`,
`examples/glow/**`, `docs/internals/glow/**`, CHANGELOG.md) plus
the `[glow]` `pyproject.toml` extra.  The M5 spec explicitly
requires writes to `docs/glow/**` (new public docs subtree) and to
the top-level `README.md` (the experimental-glow blurb).  Per the
command's own conflict-resolution rule ("plan file and the
milestone spec are authoritative"), these are M5-only allowed
exceptions; the resolved decisions captured in
`5-starry-spring.md` documented them up-front.

The D1 migration adds three more exception paths:
`src/t0_agents/guardrails/regex.py`,
`src/t0_agents/guardrails/__init__.py`, and
`tests/test_guardrails/test_regex.py`.

Final touched-paths summary (relative to `origin/main`):

- `src/t0_agents/glow/**` — `guardrails_typing.py` collapse.
- `tests/test_glow/**` — glow guardrail-test trim.
- `examples/glow/**` — `03_streaming.py` + README row refresh.
- `docs/internals/glow/**` — `comparison.md` + this dev-log.
- `docs/glow/**` — NEW (tutorial + limitations).
- `README.md` (top-level) — the "Experimental: glow" section.
- `CHANGELOG.md` — M5 + v1.0 entries.
- `src/t0_agents/guardrails/**` — `regex.py` + `__init__.py`.
- `tests/test_guardrails/**` — `test_regex.py`.
- `src/t0_agents/mcp/` — empty diff (the
  `feedback_mcp_standalone` invariant holds).

`pyproject.toml` is unchanged (the `[glow]` extra already shipped
in M2; no new deps).

## §3 Open questions

### 3.1 — No new open questions raised in implementation

The plan resolved D1/D2/D3 before the first edit.  The runtime
behaviours referenced by the tutorial (the dict-router's
model-capability requirement; the toolless-chat fallback; the
ParamSpec dual-personality on `_AgentWrapper.__call__`; the
per-agent-MCP-subsets limitation) were all already documented in
prior dev-logs; M5 just surfaces them in the user-facing pages.

### 3.2 — Resolved questions from prior dev-logs

- **M2 §3.1 (public surface for `_resolve_chat_tool_provider`)** —
  still parked.  The toolless-chat fallback is described
  prominently in `getting-started.md` ("Toolless chat — no MCP
  plumbing needed"), but the underlying helper stays
  underscore-prefixed.  Any decision to elevate the symbol is
  post-M5 work tracked under `project_pre_public_polish.md`.
- **M3 §3.3 (`_AgentWrapper.__call__` dual-personality)** —
  resolved by *document, don't fix*.  The trade-off
  (typo'd-kwarg mypy-passes-but-runtime-fails) is now captured
  in `docs/glow/limitations.md` under the
  "`_AgentWrapper.__call__` dual-personality" bullet.
- **M4 §3.1 (`_LLMRoutedComposer` per-invocation Client cost)** —
  resolved on disk in commit `4375412` (a post-M4 fix that
  switched to per-invocation `Client + Agent` for concurrency
  safety).  M5 mentions the cost model in the tutorial's
  dict-router section but does not re-open the design.

### 3.3 — Pre-existing ruff hits in `tests/`

`uv run ruff check src/ tests/` reports 6 errors in
`tests/test_cli/`, `tests/test_core/`, and `tests/test_memory/`
that pre-date M2 (already documented in
`project_pre_public_polish.md` per the M2 dev-log §3.4).  M5
did **not** touch any of these files and did **not** add any
new error — `uv run ruff check` scoped to M5-touched paths
(`src/t0_agents/glow/`, `src/t0_agents/guardrails/`,
`tests/test_glow/`, `tests/test_guardrails/`, `examples/glow/`,
`docs/`) reports "All checks passed!".  No action.

### 3.4 — Retrospective: candidate v1.1 items

The branch is now PR-ready, but the user-manual checklist
(`5-starry-spring.md`) calls out items that may shape a future
roadmap:

- **Per-agent MCP subsets** — every glow agent on a given App
  sees every mount.  A first-class visibility filter (on the
  spec or on `@app.agent`) would close the limitation.
- **Streaming JSON** — depends on TensorZero exposing
  `output.parsed` mid-stream (or a `StreamingOutput[JsonOutput]`
  composition).  Out of glow's hands until the upstream
  primitive lands.
- **`AsyncIterator[str]` tools** — the plan flagged this as a
  future addition; `@app.tool` today only handles single-value
  returns.  Surfaces would be a new event type (`ToolProgress`)
  plus a tool-side `yield`-aware adapter.
- **`_LLMRoutedComposer` public elevation** — power users may
  want to construct the composer with a pre-built classifier
  (cheaper model, custom system prompt).  Currently
  underscore-prefixed.
- **`07_agent_with_tools.py`** — close the example gap noted in
  `limitations.md` (the agent-uses-tool path is exercised by
  `03_streaming.py` but a dedicated DI-emphasising example would
  make the value-style `Depends` story self-contained).

### 3.5 — Retrospective: what went well, what was harder

- **Went well:**
  - The 5 prior dev-logs are a load-bearing input.  M5's plan was
    written entirely from the dev-log carry-forwards plus the
    spec; no fresh exploration was needed for the polish
    artefacts.
  - The user-confirmed D1/D2/D3 decisions before the first edit
    matched the M4 §4 recommendations precisely, so no mid-flight
    course-correction was needed.
  - The code-only LOC methodology was straightforward to
    implement and the headroom (42.1% vs the 40% bar) is not
    tight enough to be fragile.
- **Harder than expected:**
  - The CHANGELOG insertion point for the v1.0 `### Added`
    sub-section required reading 200+ lines of existing entries
    to confirm the right home (no existing per-factory
    sub-section under v1.0 Guardrails).  Picked a new
    `#### Guardrails — regex content blocker` sub-section
    inserted between `#### Composer` and `#### Documentation` —
    parallel to the existing structure.
  - The `Text` selector defaults to `TextEndEvent` (a completion
    marker, not text content) per the M2 D4 decision.  The
    streaming example honours that default but distinguishes the
    yielded event types so the printed output is informative;
    documented the token-by-token alternative
    (`Selector(TextDeltaEvent).map(...)`) in both the example's
    docstring and the tutorial.

## §4 Carry-forward notes

### For the post-M5 PR

- **Final `git diff src/t0_agents/mcp/` audit** for the whole
  `experimental/glow` branch.  Spot-checked at each commit
  during M5; the M0-through-M4 audits were already clean per
  prior dev-logs.  The PR reviewer should re-verify before
  merging to `main`.
- **The v1.0 core change in commit 1 should be reviewed
  separately from the experimental glow additions.**  It's a
  strict superset of the prior contract, but it's the only v1.0
  behaviour change in the M5 PR — surface it explicitly in the
  PR description (analogous to how the M2 D5 toolless-chat
  unlock was surfaced in the M2 PR).
- **The polish-backlog race-coherence items** (4 references in
  `docs/composition.md`, top-level `README.md`,
  `docs/migration-v0.12-to-v1.0.md`) tracked in
  `project_pre_public_polish.md` remain outside the M5 PR.  The
  user may decide to roll them into the M5 PR or split them off
  — out-of-scope for this dev-log.

### Post-M5

- **CHANGELOG promotion / version bump** — out of M5 scope.  The
  decision whether to consolidate the "Experimental —
  t0_agents.glow" sub-section into a formal v1.1-or-later
  release sub-section (with a `pyproject.toml` version bump) is
  the user's call once the PR lands.  Out of M5 scope but
  recorded here for completeness.
- **`07_agent_with_tools.py` example** — closes the
  `05_di_tools.py` agent-uses-tool gap noted in
  `limitations.md`.  Not blocking; a natural first post-M5
  contribution.

### For `project_pre_public_polish.md`

No new items.  The 4 race-coherence references from M0 §3
remain outstanding; M5 did not touch them.

## §5 Verification

Captured commands and outcomes (run at the end of M5
implementation, prior to writing this dev-log):

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/glow/ src/t0_agents/guardrails/ \
    tests/test_glow/ tests/test_guardrails/ examples/glow/ docs/
All checks passed!

$ uv run ruff check src/ tests/
# 6 pre-existing errors in tests/test_cli/, tests/test_core/,
# tests/test_memory/ (all pre-date M2; tracked in
# project_pre_public_polish.md).  No new errors from M5.

$ uv run mypy src/ tests/test_glow/
Success: no issues found in 94 source files

$ uv run pytest tests/test_glow/ -q -m "not integration"
.........................................................................[ 66%]
.....................................                                    [100%]
109 passed, 6 deselected in 0.23s

$ uv run pytest tests/test_guardrails/test_regex.py -q
..........                                                               [100%]
10 passed in 0.10s

$ uv run pytest -m "not integration" -q
[...]
1003 passed, 2 skipped, 63 deselected in 4.78s
# Pre-M5 baseline (end of M4 post-fix-ups): 1002 passing.  M5 net delta
# is +1 (10 v1.0 tests added; 9 glow tests removed — the migration is
# a wash except for the new precompiled-pattern path).

$ uv run pytest tests/test_glow/test_integration.py --collect-only -q
tests/test_glow/test_integration.py::test_hello_world_chat_agent_runs_end_to_end
tests/test_glow/test_integration.py::test_structured_output_returns_pydantic_instance
tests/test_glow/test_integration.py::test_orchestrator_with_glow_sub_agents
tests/test_glow/test_integration.py::test_llm_routed_dict_composer_dispatches_to_picked_route
tests/test_glow/test_integration.py::test_llm_routed_dict_composer_supports_composer_routes
tests/test_glow/test_integration.py::test_regex_guardrail_on_return_annotation_blocks_response

6 tests collected in 0.16s
# Integration coverage unchanged from M4.  Live execution against the
# gateway is the user's pre-push step (the two `_llm_routed_*` tests
# require a model that honours `tool_choice="required"` per M4 §3.3).

$ git diff src/t0_agents/mcp/ | wc -l
0
# (mcp/ standalone-submodule invariant preserved per
# feedback_mcp_standalone)

$ git log --oneline experimental/glow ^main | head -10
6d606bc chore(glow): document M5 in CHANGELOG
7359db5 docs(readme): point at experimental glow tutorial
41902f0 docs(glow): line-count comparison vs base API
ed5731a docs(glow): known limitations — limitations.md
58bc526 docs(glow): user-facing tutorial — getting-started.md
f7eda37 feat(glow): 03_streaming example with event selectors
a2a553b refactor(glow): re-export migrated regex_guardrail from v1.0
0afcf0e feat(guardrails): regex helper for input/output validation
2cb2f00 Merge pull request #41 from sproutfi/glow/m4-composition-mcp-guardrails
4375412 fix(glow): per-invocation Client + Agent for _LLMRoutedComposer concurrency
# The 8 M5 commits (1-8) all with the conventional-commit
# (glow) / (guardrails) / (readme) scopes.  This dev-log commit
# will bring the total to 9.

$ git diff --stat origin/main..HEAD | tail -1
 68 files changed, 10484 insertions(+), 116 deletions(-)
# Touched paths confined to:
# - src/t0_agents/glow/** + tests/test_glow/** + examples/glow/**
# - docs/internals/glow/** + docs/glow/** + CHANGELOG.md
# - pyproject.toml + uv.lock (from M2)
# - README.md (M5-only)
# - src/t0_agents/guardrails/** + tests/test_guardrails/** (M5 D1)
# - src/t0_agents/core/agent.py + tests/test_core/test_agent.py (M2 D5)
# No new collateral damage in M5.

$ wc -l examples/02_agent_class/basic.py examples/glow/01_hello.py
   65 examples/02_agent_class/basic.py
   49 examples/glow/01_hello.py
  114 total
# Raw ratio: 49/65 = 75.4% → 24.6% reduction (below spec target).
# Code-only ratio (after stripping blanks/#/leading-docstring):
# 22/38 = 57.9% → 42.1% reduction.  Clears the ≥40% target.
# See docs/internals/glow/comparison.md for the methodology
# snippet and the verbatim expected output.

$ uv run python -c "
from t0_agents.glow import (
    App, Depends, Done, Error, FluentRun, MCPServer, Selector,
    Text, Thought, ToolCall, ToolResult, TurnEnd, compose, regex_guardrail,
)
from t0_agents.guardrails import regex_guardrail as v1_regex
assert regex_guardrail is v1_regex
print('glow surface OK: 14 symbols imported.')
print('regex_guardrail re-export identity:', regex_guardrail is v1_regex)
"
glow surface OK: 14 symbols imported.
regex_guardrail re-export identity: True
# End-to-end smoke confirms every public glow symbol still
# imports cleanly, and that the M5 migration preserves identity
# (the glow re-export IS the v1.0 callable, not a wrapper).
```

Live integration runs (`docker compose -f
examples/glow/docker-compose.yml --env-file .env up -d` then
`uv run pytest -m integration -q`) are the user's pre-push step,
per the user-manual checklist in `5-starry-spring.md` and per
`feedback_full_test_suite_before_push`.  The 6 integration tests
collect cleanly; live execution requires a model that honours
`tool_choice="required"` for the two `_llm_routed_*` cases
(`openai/gpt-oss-20b` does not — pick Claude Sonnet/Opus, GPT-4o,
GPT-5, or Gemini-2.5-pro and update
`examples/glow/config/tensorzero.toml` before the run).  The
example smoke (`for ex in examples/glow/*.py; do uv run python
"$ex"; done`) is the same pre-push step.

The full pre-push `uv run pytest -q` (no marker filter) is the
user's responsibility per `feedback_full_test_suite_before_push`.
