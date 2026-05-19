# Milestone 2 dev-log — `compose-removal/delete-and-examples`

> **Status:** Landed on `compose-removal/m2-delete-and-examples` (12 commits).
> **Date:** 2026-05-19.
> **Topic branch:** `compose-removal/m2-delete-and-examples` off `experimental/glow`.
> **Plan file:** `~/.claude/plans/compose-removal-plan.md`.
> **Spec:** `docs/internals/compose-removal/milestones/milestone-2-delete-and-examples.md`.
> **Prior dev-log read in full:** `milestone-1-dev-log.md`
> (8 commits, 7 carry-forward items — all resolved here; see §3).

## §1 Summary — what was done

The topology composer layer in `src/t0_agents/compose/` and glow's
`compose()` were removed in full.  Six canonical multi-agent pattern
examples — Sequential, Parallel, Orchestrator under both the legacy
`Agent` API and the `@app.agent` decorator API — replaced the deleted
demos.  CHANGELOG records the deletion under
`## [1.0.0] — Unreleased`.

### Files added / moved / deleted

**Created (examples + dev-log):**

- `examples/12_concurrency/01_sequential.py` (90 LOC) — toolless
  `summarizer → translator` pipeline + ad-hoc truncate.
- `examples/12_concurrency/02_parallel.py` (100 LOC) — three toolless
  agents via `asyncio.gather` + Python reducer.
- `examples/12_concurrency/03_orchestrator.py` (130 LOC) — toolless
  leaves + orchestrator with `AgentConfig(sub_agents=[…])`; explicit
  `max_turns` anti-recursion note in the module docstring.
- `examples/glow/04_sequential.py` (84 LOC) — `@app.agent` counterpart
  of `12_concurrency/01_sequential.py`.
- `examples/glow/05_parallel.py` (98 LOC) — `@app.agent` counterpart
  of `12_concurrency/02_parallel.py`.
- `examples/glow/06_orchestrator.py` (96 LOC) — `Depends(<wrapper>)`
  orchestrator, mirrors `tests/test_glow/test_integration.py:99-146`.
- `docs/internals/compose-removal/logs/milestone-2-dev-log.md` (this
  file).

**Renamed (via `git mv`, history preserved):**

- `examples/glow/05_di_tools.py` → `examples/glow/07_di_tools.py`
- `examples/glow/06_mcp_typing.py` → `examples/glow/08_mcp_typing.py`

**Deleted (source + tests + docs + obsolete examples):**

- `src/t0_agents/compose/{__init__,builtins,_runtime,topology}.py`
  + the directory itself (2,283 LOC across 4 files).
- `src/t0_agents/glow/compose.py` (449 LOC).
- 11 topology test files in `tests/test_compose/v1/`
  (`test_{sequential,parallel,race,conditional,consensus,map_reduce,saga,retry_chain,circuit_breaker,nested_composers,cancellation}.py`,
  ~65 tests, 1,037 LOC) + `_helpers.py` (167 LOC) +
  `v1/__init__.py` + `__init__.py` + the `tests/test_compose/`
  directory itself.
- `tests/test_glow/test_compose.py` (343 LOC, 18 tests).
- Two integration tests in
  `tests/test_glow/test_integration.py`:
  `test_llm_routed_dict_composer_dispatches_to_picked_route`
  (lines 148-186 pre-trim),
  `test_llm_routed_dict_composer_supports_composer_routes`
  (lines 189-226 pre-trim).
- `examples/12_concurrency/01_parallel_tools.py` (104 LOC).
- `examples/12_concurrency/02_parallel_subagents.py` (158 LOC).
- `examples/12_concurrency/server.py` (56 LOC),
  `research_server.py` (27 LOC), `writing_server.py` (28 LOC) —
  orphaned by the toolless-leaves design of the new
  `03_orchestrator.py` (see §2 D5).
- `examples/glow/04_composition.py` (97 LOC).
- `docs/composition.md` (606 LOC).

**Modified:**

- `src/t0_agents/__init__.py` — 12 topology imports removed from the
  import block; 12 corresponding entries removed from `__all__`.
- `src/t0_agents/glow/__init__.py` — `compose` import + `__all__`
  entry removed.
- `tests/test_glow/test_integration.py` — `compose` symbol dropped
  from the `from t0_agents.glow import …` line (24).
- `examples/glow/{07,08}` — internal docstring path references
  updated post-rename.
- `examples/glow/README.md` — table rebuilt for the new 01-08
  numbering (Milestone column dropped — provenance lives in
  `docs/internals/glow/logs/`); function-usage paragraph updated to
  reference the new examples.
- `examples/12_concurrency/README.md` — full rewrite (mocked
  narrative → gateway-backed canonical patterns; shared glow
  gateway boot instructions; nest-and-compose narrative at the
  bottom).
- `examples/08_multi_agent/config/tensorzero.toml:10` —
  `t0_agents.compose.install_sub_agents` →
  `t0_agents.delegation.install_sub_agents` (M1 §4 D4
  carry-forward).
- `docs/glow/limitations.md` — "LLM-routed dict-composer model
  capability" section deleted; race-deprecation section rewritten
  to drop the `compose({…})` cross-reference (the v1.0
  composition.md link too); `05_di_tools.py` → `07_di_tools.py`
  in the path reference of the surviving "MCP-only — no LLM
  round-trip" section.
- `docs/glow/getting-started.md` — "Composers and agents are the
  same thing" + "Native composition — `compose(spec, …)`" + "Sets
  and merging" + "Race is deprecated" sections (~90 LOC) replaced
  with a brief "Multi-agent patterns" pointer table.  Intro line
  drops "Composer" from the v1.0 primitives.
- `docs/multi-agent.md` — top-of-page "v1.0 Composers — orthogonal
  path" callout rewritten as a brief pointer to the new example
  files; line 117 `t0_agents.compose` →
  `t0_agents.delegation`.
- `docs/api-reference.md` — entire `## Composers *(v1.0)*` section
  removed (~62 LOC: 9 builders + protocol table + module constant
  + public-surface example).  `Run` section's "third-party
  Composer authors" reference reworded.
- `docs/middleware.md` — three `composition.md` cross-references
  retired; companion-reading block trimmed; SubAgentMiddleware
  audit blurb retained without composition.md cite; Related list
  swapped `composition.md` for `multi-agent.md`.
- `docs/run.md` — opening callout + Related entries: composition.md
  → multi-agent.md.
- `docs/agent.md` — top callout updated from "four pages" to
  "three pages"; Composition link replaced with Multi-Agent +
  examples pointer.
- `docs/migration-v0.12-to-v1.0.md` — companion-reading block
  trims composition.md; `### Composer — topology primitive`
  subsection (16 LOC) rewritten as a historical note (the v1.0
  Composer was added then removed before public release);
  checklist item 4 drops `Composer`.
- `docs/concurrency.md` — single dangling reference to the deleted
  `01_parallel_tools.py` rewritten as a generic CI probe note.
- `docs/index.md` — reading-path entry 6 (Composition) removed;
  remaining entries renumbered 6-14; Multi-Agent bullet expanded
  to cite the new examples.
- `README.md` — drops `Composer` from the v1.0 primitives list
  ("four" → "three"), removes the `# 4. Composer` code block,
  removes the "Topology composers" feature row, rewrites the
  glow surface paragraph to drop `compose([…])` /
  `compose({a, b}, …)`.  Reading-path bullet for Composition
  deleted; trailing glow example count 6 → 8.
- `CHANGELOG.md` — `### Removed` block under
  `## [1.0.0] — Unreleased` enumerates the 9 topology factories,
  the 4-symbol protocol layer, glow's `compose()`, and the M1
  sub-agent relocation.  No migration guide (no external users).

### Public API delta

`src/t0_agents/__init__.py.__all__`: **12 symbols removed**
(`Composable`, `Composer`, `ComposerError`, `circuit_breaker`,
`conditional`, `consensus`, `map_reduce`, `parallel`, `race`,
`retry_chain`, `saga`, `sequential`).  Net `__all__` post-M2: 47
symbols (was 59 post-M1).

`src/t0_agents/glow/__init__.py.__all__`: **1 symbol removed**
(`compose`).  Net `__all__` post-M2: 13 symbols (was 14 post-M1).

### Tests inventory delta

- Topology tests (M2 C4): 11 files / ~65 tests gone.
- Glow compose tests (M2 C5a): 1 file / 18 tests gone.
- Glow dict-router integration tests (M2 C5b): 2 tests gone.

Pre-M2 baseline on `experimental/glow` (post-M1): 1001 offline
tests passing.  Post-M2: **918 offline tests passing**, 2 skipped,
61 deselected (integration).  Net offline delta: −83
(−65 topology − 18 glow compose = −83 ✓).  Integration delta:
−2 (6 → 4 in `test_glow/test_integration.py`).

### Docs delta

- 1 file deleted entirely (`docs/composition.md`, 606 LOC).
- 1 new file (this dev-log).
- 12 files edited (`docs/glow/{limitations,getting-started}.md`,
  `docs/{multi-agent,api-reference,middleware,run,agent,
  migration-v0.12-to-v1.0,concurrency,index}.md`, `README.md`,
  `examples/glow/README.md`, `examples/12_concurrency/README.md`,
  `examples/08_multi_agent/config/tensorzero.toml`).

### Commit SHAs (in landing order)

| C | SHA | Message |
|--:|:----|:--------|
| 1  | `55abb1c` | `refactor(t0_agents): remove compose topology primitives from public API` |
| 2  | `f9d24a4` | `refactor(glow): remove compose() from public API` |
| 3  | `1f5edf4` | `chore(compose-removal): remove compose/ directory and glow/compose.py` |
| 4  | `6a57ea4` | `test(compose-removal): remove topology tests` |
| 5  | `bc9e8cc` | `test(glow): remove dict-router composer tests` |
| 6  | `2790a36` | `chore(examples): remove obsolete composition examples` |
| 7  | `cd8183b` | `feat(examples): canonical sequential pattern (Agent legacy)` |
| 8  | `483190f` | `feat(examples): canonical parallel pattern (Agent legacy)` |
| 9  | `3ec1fc7` | `feat(examples): canonical orchestrator pattern (Agent legacy)` |
| 10 | `49575d3` | `feat(examples): canonical patterns for glow (sequential, parallel, orchestrator)` |
| 11 | `c380d21` | `docs(compose-removal): rewrite affected docs and prune orphan example servers` |
| 12 | `fbe0af3` | `chore(compose-removal): CHANGELOG → ### Removed` |

(The dev-log commit lands on top — its own SHA goes here once the
file is committed.)

## §2 Deviations from the milestone spec

### D1 — `docs/api-reference.md` and `docs/agent.md:9` added to the docs sweep

**What changed:** the M2 spec's §Docs deletion/update enumeration named
`docs/composition.md`, `docs/glow/limitations.md`, `docs/glow/
getting-started.md`, `docs/multi-agent.md`, `README.md`, and
`docs/migration-v0.12-to-v1.0.md`.  The Phase-1 audit found two
additional load-bearing references: `docs/api-reference.md:757-790`
(an entire `## Composers *(v1.0)*` section reproducing the symbol
table and a code example) and `docs/agent.md:9` (a "four pages"
v1.0-surface callout linking to `composition.md`).  Both were folded
into commit 11.

**Why:** the spec's list is a checklist, not an exhaustive catalogue;
the operational invariant in the M2 acceptance criteria
(`git grep -nE "from t0_agents\.compose|import.*\bcompose\b" src/
tests/` empty) is necessary but not sufficient — the docs grep
(`git grep -n "composition.md" docs/ README.md`) reveals the broader
leakage surface.  User approved this expansion during the planning
interview (decision §3 of the plan).

**Impact on subsequent milestones:** none — this is the final
milestone of the workstream.

### D2 — additional `composition.md` references in `docs/middleware.md`, `docs/run.md`, `docs/index.md`

**What changed:** during the docs-sweep verification (final
`git grep -n "composition.md"`), three more live links surfaced:
`docs/middleware.md:21,219,472`, `docs/run.md:19,397`, and
`docs/index.md:28,34`.  All five files were edited in commit 11 to
swap their composition.md cross-references for multi-agent.md
pointers + example-file citations.  `docs/index.md` also required a
renumbering pass (entries 6-14) after the Composition reading-path
bullet was deleted.

**Why:** same as D1 — the spec list under-enumerated, the grep gate
caught everything.

**Impact on subsequent milestones:** none.

### D3 — `examples/12_concurrency/03_orchestrator.py` uses toolless leaves, not the existing `research_server.py` / `writing_server.py`

**What changed:** the plan's "Files to change" section described
`03_orchestrator.py` as reusing `research_server.py` and
`writing_server.py` for the leaf sub-agents (the `search_papers` /
`format_outline` tools).  An initial draft was written that way and
smoke-ran end-to-end — but it exposed an oversight: the legacy
`Agent` path has no per-agent system prompt mechanism unless the
caller passes `system=…` on `Agent.invoke(...)`, and `SubAgentMiddleware`
invokes sub-agents internally without surfacing `system=` to the
caller.  Without a system prompt, the leaf LLMs ignored their MCP
tools (or used them inconsistently), reducing the example's
clarity.

The final version uses three empty FastMCP servers — one for the
orchestrator (where `researcher_ask` / `writer_ask` get installed)
and one each for the leaves as inert tool providers.  The leaves
run toolless chat; the orchestrator's system prompt is the sole
controller of leaf behaviour.  This is honest pedagogy (one thing
per example) and matches the M3 glow integration test pattern
which also uses toolless leaves (the glow path delivers system
prompts via docstrings, but the demonstration shape is the same).

**Why:** clarity over feature density.  The canonical orchestrator
pattern is "parent LLM picks among `_ask` tools"; the per-leaf
tool-use is orthogonal and demonstrated separately by
`examples/glow/03_streaming.py` (an agent calling a `@app.tool`).

**Impact on subsequent milestones:** the previously-reused
`research_server.py`, `writing_server.py`, and the orphaned
`server.py` (consumer was the deleted `01_parallel_tools.py`) are
deleted in commit 11 alongside the README rewrite.  All three
were already pure code with no other consumers in the repo.

### D4 — `examples/12_concurrency/02_parallel.py` and `glow/05_parallel.py` use three agents (optimist/pessimist/pragmatist), not two

**What changed:** the M2 spec's Reference structure for
`02_parallel.py` showed a 3-agent example (`optimist`,
`pessimist`, `pragmatist`); the plan abstracted it to "multiple
Agents".  Both the legacy and glow versions kept the three-agent
variant verbatim — it makes the `\n---\n` reducer's "drop failures"
behaviour visible without contriving a failure path.

**Why:** the three-agent version is what the spec sketched and what
the user implicitly approved during planning.  No surprise here.

**Impact on subsequent milestones:** none.

### D5 — `examples/glow/06_orchestrator.py` is a known gpt-oss-20b flake

**What changed:** the second-pass smoke run of `06_orchestrator.py`
in the verification block (§5) hit
`tensorzero.types.TensorZeroError: All variants failed … groq: …
Failed to parse tool call arguments as JSON` once before completing
cleanly on the immediate retry.  The failing run was the orchestrator
LLM emitting a >4 kB `writer_ask` tool-call argument that broke
Groq's JSON parser.

**Why:** documented gpt-oss-20b behaviour, not an M2 regression.
The orchestrator pattern itself is sound — the M3 glow integration
test (`tests/test_glow/test_integration.py:test_orchestrator_with_
glow_sub_agents`) exercises the same shape end-to-end and passes
reliably; the example uses the same pattern.  Adding the entry to
`[[project_pre_public_polish]]` is the standing remedy
(`project_glow.md` already lists "2 gpt-oss-20b integration-test
flakies discovered 2026-05-18").

**Impact on subsequent milestones:** none — this is the final
milestone.

## §3 Open questions

**Resolved from prior dev-logs:**

- **Q from M1 §4 (D4 carry-forward):** `examples/08_multi_agent/
  config/tensorzero.toml:10` references
  `:func:\`t0_agents.compose.install_sub_agents\``.  **Resolved** in
  M2 commit 11 — the comment now points at
  `t0_agents.delegation.install_sub_agents`.

- **Q from M1 §4 (D6 carry-forward):** `src/t0_agents/__init__.py`
  still has `from t0_agents.compose import …` (12 topology symbols).
  **Resolved** in M2 commit 1 — entire import block removed.

- **Q from M1 §4 (`delegation/` stability invariant):** `src/t0_agents/
  delegation/` must stay byte-identical from M1's C8 (`04003bc`)
  through M2.  **Resolved** — verified empty `git diff
  src/t0_agents/delegation/ experimental/glow..HEAD` at the M2
  verification block.

- **Q from M1 §4 (`tests/test_compose/` is ready for deletion):**
  what remained were `__init__.py` (empty), `v1/__init__.py` (empty),
  `v1/_helpers.py` (5,460 bytes).  **Resolved** in M2 commit 4 —
  all three removed plus the 11 topology test files; both
  `tests/test_compose/v1/` and `tests/test_compose/` directories
  deleted.

- **Q from M1 §4 (`docs/multi-agent.md` verification):** M1 did not
  touch this file; M2 needed to verify references match
  `delegation/`.  **Resolved** in M2 commit 11 — line 117 updated
  to `t0_agents.delegation` and the "v1.0 Composers — orthogonal
  path" callout rewritten.

- **Q from M1 §4 (`C5 picked up ruff-sort cleanup …`):** M2 commits
  expected similar ruff sort churn.  **Resolved** — M2 commit 1
  required one `uv run ruff check --fix` run on
  `src/t0_agents/__init__.py` after the 12-symbol import block was
  removed (the truncation caused trailing `__all__` ordering churn
  that ruff auto-fixed in a single pass).  All later commits left
  ruff clean without intervention.

- **Q from M1 §3 (6 pre-existing ruff baseline errors):** M2 should
  not include incidental cleanups of these.  **Confirmed** — the
  same 6 baseline errors remain on this branch:
  `tests/test_cli/test_elicit_tui.py:326` (RUF012),
  `tests/test_core/test_agent_integration.py:16` (F401
  `tests.server.mcp`),
  `tests/test_core/test_agent_parity.py:15` (F401 `fastmcp.Client`),
  `tests/test_memory/test_agent_memory.py:112` (F841 `r2`),
  `tests/test_memory/test_agent_memory_integration.py:16,17` (F401 x2).
  None introduced or fixed by M2; all remain on
  `[[project_pre_public_polish]]`.

**New questions:**

- **gpt-oss-20b orchestrator tool-call JSON flake (D5 above).**
  Tracked on `[[project_pre_public_polish]]` (per `project_glow.md`,
  the polish entry already records 2 gpt-oss-20b integration-test
  flakies); adding this one rounds out a third.  No M2-blocking
  issue — the example completes cleanly on retry.

- **Two gateways cannot coexist (port 3000 conflict).**
  `examples/glow/docker-compose.yml` and the repo-root
  `docker-compose.yml` both bind `3000:3000`.  Integration tests
  collected under `tests/` need the repo-root toml (10 functions);
  the 4 surviving glow integration tests need the glow toml (3
  functions including `orchestrator`).  No single `docker compose
  up` boots both.  Inherited from `experimental/glow`; predates the
  compose-removal workstream.  Worth filing on
  `[[project_pre_public_polish]]` if not already there.

## §4 Carry-forward notes

`No carry-forward notes for subsequent milestones.`  This is the
final milestone of the compose-removal workstream.

**Handoff to the user (outside the workstream's scope):** the
`experimental/glow` branch is now 73 commits ahead of `main` (61
from the glow roadmap + 12 from compose-removal M2; M1 already
merged into `experimental/glow` via PR #43 contributed 8 more
commits that are part of the 61/12 split — verify with `git log
--oneline main..experimental/glow | wc -l` after M2 lands).  The
merge-to-main decision is a release-strategy call (whether to
consolidate the experimental subsection of CHANGELOG into a formal
1.0 release with a `pyproject.toml` version bump, or to keep glow
labelled experimental indefinitely).

## §5 Verification

Commands run on `compose-removal/m2-delete-and-examples` HEAD
(`fbe0af3`, pre-dev-log) after all 12 commits landed:

```bash
$ git branch --show-current
compose-removal/m2-delete-and-examples

$ git log --oneline experimental/glow..HEAD | wc -l
12

$ git diff --stat src/t0_agents/delegation/ experimental/glow..HEAD
(empty — delegation/ is byte-identical to its M1 end-state)

$ git diff --stat src/t0_agents/mcp/ experimental/glow..HEAD
(empty — feedback_mcp_standalone invariant holds across M1+M2)

$ git grep -nE "from t0_agents\.compose|import.*\bcompose\b" src/ tests/
(no matches — exit 1)

$ git grep -n "composition.md" docs/ README.md | grep -v "docs/internals"
(no matches — only historical references in docs/internals/{glow,compose-removal}/)

$ uv run ruff check src/ tests/
Found 6 errors.
[*] 4 fixable with the `--fix` option.
(all 6 are the M1-documented baseline — pre-existing on experimental/glow)

$ uv run mypy src/
Success: no issues found in 79 source files
(was 84 pre-M2; the −5 = 4 compose/* + glow/compose.py)

$ uv run pytest -m "not integration" -q
918 passed, 2 skipped, 61 deselected in 2.96s
(pre-M2: 1001 offline.  Delta: −83 = 65 topology + 18 glow compose.)

$ uv run python -c "
import t0_agents
removed = {'Composable', 'Composer', 'ComposerError',
           'circuit_breaker', 'conditional', 'consensus', 'map_reduce',
           'parallel', 'race', 'retry_chain', 'saga', 'sequential'}
overlap = removed & set(t0_agents.__all__)
assert not overlap, f'still exported: {overlap}'
import t0_agents.glow
assert 'compose' not in t0_agents.glow.__all__, 'glow still exports compose'
print('OK — public API clean')
"
OK — public API clean
```

**Integration suite — full run requires both gateways (pre-existing
on `experimental/glow`):**

```bash
# Repo-root gateway (10 functions, covers core/mcp + 3 of 4 glow tests):
$ docker compose --env-file .env up -d && \
    until curl -sf http://localhost:3000/health > /dev/null; do sleep 1; done && \
    uv run pytest -m integration -q
1 failed, 60 passed, 920 deselected in 76.27s (0:01:16)
# Failure: tests/test_glow/test_integration.py::test_orchestrator_with_glow_sub_agents
# — needs the `orchestrator` function only declared in the glow toml.

# Glow gateway (3 functions including `orchestrator`):
$ docker compose --env-file .env down && \
    docker compose -f examples/glow/docker-compose.yml --env-file .env up -d && \
    uv run pytest tests/test_glow/test_integration.py -m integration -q
1 failed, 3 passed in 4.72s
# Failure: test_regex_guardrail_on_return_annotation_blocks_response
# — known gpt-oss-20b flake (project_pre_public_polish); the model
# declined to emit the placeholder SSN this run.  Not an M2
# regression — exists pre-M2 on experimental/glow.
```

**Manual smoke runs of the 6 new examples (against the glow
gateway):**

| Example                                                  | Outcome |
|----------------------------------------------------------|---------|
| `examples/12_concurrency/01_sequential.py`               | ✓ 2-step summarize → translate (Spanish output) |
| `examples/12_concurrency/02_parallel.py`                 | ✓ 3-agent fan-out, reducer joins survivors |
| `examples/12_concurrency/03_orchestrator.py`             | ✓ Markdown bullet via orchestrator → researcher_ask → writer_ask |
| `examples/glow/04_sequential.py`                         | ✓ same as 12_conc/01 via `@app.agent` |
| `examples/glow/05_parallel.py`                           | ✓ same as 12_conc/02 via `@app.agent` |
| `examples/glow/06_orchestrator.py`                       | ✓ on retry (one flake — D5) |

CHANGELOG verification:

```bash
$ grep -A 20 "^### Removed" CHANGELOG.md | head -25
### Removed

- **`compose` topology primitives** — the 9 factories
  (`sequential`, `parallel`, `race`, `conditional`, `consensus`,
  `map_reduce`, `saga`, `retry_chain`, `circuit_breaker`) and the
  `Composable` / `Composer` / `CompositeRun` / `ComposerError`
  protocol layer.  Use raw `asyncio.gather` + `await
  agent.invoke().result()` chaining for Sequential/Parallel, and
  `AgentConfig(sub_agents=[...])` with a routing system prompt for
  the dispatch case.  See
  `examples/12_concurrency/{01_sequential,02_parallel,03_orchestrator}.py`
  and `examples/glow/{04_sequential,05_parallel,06_orchestrator}.py`.
- **`t0_agents.glow.compose`** — removed alongside the underlying
  topology layer.
- **Sub-agent orchestration relocated** — `agent_as_tool`,
  `install_sub_agents`, and friends now live in
  `t0_agents.delegation` (top-level re-exports unchanged).
```

End of milestone.  The compose-removal workstream is complete.
