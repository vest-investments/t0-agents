# M0 — `race-deprecation` — dev-log

> **Milestone:** `milestone-0-race-deprecation` ([spec](../milestones/milestone-0-race-deprecation.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-16.

## §1 Summary — what was done

The low-level `race(*nodes)` composer is now deprecated in v1.0 with
a `DeprecationWarning` emitted by the factory; `docs/composition.md`
demotes it to an "Advanced / not recommended" appendix; `CHANGELOG.md`
carries a v1.0 `### Deprecated` subsection. The existing test suite
asserts the warning fires positively and the two non-`test_race`
callsites are rewritten to `parallel(..., error_policy=...)` to keep
the wider suite green without smearing `pytest.warns` shims across
unrelated cancellation / nesting tests. Glow itself ships zero new
code in M0 — that begins at M1.

### Files touched

| Path | Change | Lines |
|---|---|---|
| `src/t0_agents/compose/builtins.py` | Added `import warnings`; emit `DeprecationWarning` from `race()` factory; "Deprecated since v1.0" header in the docstring. | +17 / -0 |
| `tests/test_compose/v1/test_race.py` | All 5 tests wrap each `race(...)` invocation in `with pytest.warns(DeprecationWarning, match="deprecated"):` so the warning is positively asserted. | +14 / -5 |
| `tests/test_compose/v1/test_nested_composers.py` | Dropped `race` from imports; rewrote two callsites (line 31 `sequential(race(fast, slow), router)`; line 59 `race(MockComposable("a"), MockComposable("b"))`) to `parallel(..., error_policy="continue")`. | +8 / -4 |
| `tests/test_compose/v1/test_cancellation.py` | Dropped `race` from imports; rewrote `sequential(race(inner_a, inner_b), outer_c)` to `sequential(parallel(inner_a, inner_b, error_policy="fail_fast"), outer_c)` preserving the cancellation-cascade assertions. | +6 / -4 |
| `docs/composition.md` | Replaced front-of-page `sequential(race(a, b), …)` nesting illustration with `parallel(a, b)`; added a Deprecation Notice block; removed `race` row from the patterns table + pointer below it; removed the `### race` subsection from the middle of the file; added new `## Advanced / not recommended` appendix at the bottom with rationale, replacement guidance, and the original snippet (prefixed deprecated). | +61 / -18 |
| `CHANGELOG.md` | New `### Deprecated` subsection inserted under `## [1.0.0] — Unreleased`, between `### Changed` and `### Removed` (Keep-a-Changelog canonical order). | +10 / -0 |

### Public glow API delta

`t0_agents.glow.__all__` is still `[]`. M0 is intentionally a
low-level deprecation only; no glow symbol is introduced. M1
begins the additive surface.

### Commits

The milestone landed as three commits on `experimental/glow`, in
order:

| # | SHA | Subject |
|---|---|---|
| 1 | `8f452f2` | `deprecate(compose): emit DeprecationWarning from race() and assert it in tests` |
| 2 | `95db857` | `docs(compose): demote race to advanced / not-recommended appendix` |
| 3 | `0071177` | `chore(compose): document race deprecation in CHANGELOG` |

The 14-commit overall glow roadmap labels this as "Commit 0"; the
split into three story-telling commits is captured in §2 below.

### Tests

No new test modules. The three pre-existing files received targeted
edits (counts above). The suite size is unchanged:
`883 passed, 2 skipped, 57 deselected` (offline; identical to the
pre-M0 state because the changes were behavioural-equivalent under
`pytest.warns` and reducer-default semantics).

## §2 Deviations from the milestone spec

Three small deviations, all in service of cleaner downstream
ergonomics. None of them contradict the plan file or the spec's
intent; they refine specifics the spec left open.

### 2.1 — `Commit 0` split into 3 story-telling commits

The spec labelled the milestone as a single "Commit 0:
`deprecate(compose): mark race as deprecated`". The project-wide
convention codified in `CLAUDE.md` and reinforced by
`feedback_multi_commit_prs` ("split push/PR work into 5–10
story-telling commits; never single giant commits") meant collapsing
M0 into one commit would have been a smell. The agreed three-commit
split is documented in `/home/makarenko/.claude/plans/0-elegant-island.md`
("Decisions taken with the user"):

1. `deprecate(compose): emit DeprecationWarning from race() and
   assert it in tests` — atomic factory change + immediate test
   sync (avoids a window where the suite emits unwrapped
   warnings).
2. `docs(compose): demote race to advanced / not-recommended
   appendix`.
3. `chore(compose): document race deprecation in CHANGELOG`.

**Impact on subsequent milestones:** none. The plan file's
"Implementation plan — commits" item 0 remains accurate; the
three-commit split is internal to M0.

### 2.2 — `test_cancellation.py` rewritten to `parallel(error_policy="fail_fast")`, not wrapped in `pytest.warns`

The spec offered two options for the single `race(inner_a, inner_b)`
inside the `test_cancel_nested_composers_cascades_to_leaves` test:
"either wrap with `pytest.warns(DeprecationWarning)` or rewrite to
test cancellation through `parallel` with `error_policy='fail_fast'`
(whichever is closer to the original test intent)".

Picked **rewrite**. The test's assertion (the long-delay inner
children cancel; `outer_c.invocations == []`) is about cancellation
cascade through nested composers, not race-specific semantics — both
`race(inner_a, inner_b)` and `parallel(inner_a, inner_b,
error_policy="fail_fast")` exercise the same invariant in the same
way (the outer sequential's `.cancel()` walks `_sub_runs` and the
inner composer's `_drive_*` doesn't matter because both children
have `delay=10.0` and never resolve). The rewrite keeps the
assertion precise and avoids smearing a `pytest.warns` shim across
a cancellation test.

**Impact on subsequent milestones:** none.

### 2.3 — `test_nested_composers.py` drops `race` at two callsites, not one

The spec's checklist explicitly called out line 31
(`sequential(race(...), ...)`). A grep at planning time found a
second usage at line 59 inside
`test_nested_composer_satisfies_composable_protocol`
(`inner = race(MockComposable("a"), MockComposable("b"))`). That
test is composer-agnostic — the only assertion is
`isinstance(inner, Composer)` / `isinstance(inner, Composable)`,
both of which `ParallelComposer` also satisfies — so rewriting it to
`parallel(..., error_policy="continue")` keeps the test's intent
intact while letting the file drop its `race` import entirely.
Without this second rewrite, the suite would emit an unwrapped
`DeprecationWarning` every time `test_nested_composer_satisfies_composable_protocol`
ran, polluting `-W error::DeprecationWarning` runs.

**Impact on subsequent milestones:** M1 should treat
`parallel(..., error_policy="continue")` as the framework's
canonical fan-out shape; glow's `compose()` `set[Composable]`
dispatch must call this exact factory with the same `error_policy`
so the policy is anchored in both layers. See §4 for the explicit
carry-forward.

## §3 Open questions

The user's confirmed scope (spec-only) means three coherence gaps
remain in the public docs after M0:

- `docs/composition.md:489` — the "Triage router for support
  tickets" production pattern uses `race(general_a, general_b)`
  inside a `routes={...}` block. The same file otherwise demotes
  race to the appendix, so the in-page example is a coherence gap.
  Logged to the pre-public-polish backlog.
- `README.md:88-93` — flagship "four pillars" demo uses
  `race(writer_fast, writer_thorough)`. Logged.
- `docs/migration-v0.12-to-v1.0.md:324` — v1.0 Composer description
  enumerates nine builders including `race` without flagging the
  deprecation. Logged.

All three are parked in the
`project_pre_public_polish.md` memory entry (under the new "Out-of-scope
`race` references after M0 deprecation" section). They are
intentional out-of-scope items, not unresolved questions per se —
the user confirmed they go to the pre-public-polish backlog.

No new questions surfaced during implementation. The plan file
covered everything; the runtime read of
`src/t0_agents/compose/_runtime.py:192-275` confirmed the default
`parallel` reducer's `[r.value for r in successful]` shape, which
made the `error_policy="continue"` swap in
`test_nested_composers.py:31` mechanical.

## §4 Carry-forward notes

For subsequent milestones implementing the glow surface:

- **M1+: `parallel(*, reducer=..., error_policy="continue")` is the
  canonical fan-out idiom.** The deprecation message itself enshrines
  this — the `race()` factory's `warnings.warn` body literally
  points at this exact factory call. Glow's `compose()` dispatch
  for `set[Composable]` (planned for M4, commit 8 in the 14-commit
  roadmap, file `src/t0_agents/glow/_compose.py`) must use the
  identical signature shape, with the same `error_policy="continue"`
  default. Anything else would introduce a contradiction between
  the deprecation message and the glow API — the very contradiction
  M0 was supposed to prevent. If M4 surfaces a reason to deviate
  (e.g., a `set` literal needs different default error semantics
  than a low-level explicit call), record that in M4's dev-log §2
  and update the deprecation message in `builtins.py:race()`
  accordingly.
- **M1+: the pre-public-polish backlog has 4 race coherence items.**
  Whoever closes them (probably M5 polish work, but no guarantee)
  must propagate the deprecation language consistently: the same
  "use `parallel(*, reducer=..., error_policy='continue')` for
  fan-out + merge; use `asyncio.wait(return_when=FIRST_COMPLETED)`
  for first-finisher-wins" pairing appears in (a) the factory
  warning, (b) the docstring header, (c) the
  `docs/composition.md` intro notice, (d) the
  `docs/composition.md` appendix, (e) the CHANGELOG entry — and
  must appear identically in the README / migration doc fixes.
- **No carry-forward instructions for M2/M3/M4/M5 specifically.**
  M0 touches nothing in the glow source surface; none of the
  later milestones depend on the M0 commits beyond the policy
  anchoring above.

## §5 Verification

Captured commands and outcomes:

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/compose/ tests/test_compose/v1/
All checks passed!

$ uv run mypy src/
Success: no issues found in 71 source files

$ uv run pytest tests/test_compose/v1/test_race.py \
                tests/test_compose/v1/test_nested_composers.py \
                tests/test_compose/v1/test_cancellation.py -q
..............                                                           [100%]
14 passed in 1.12s

$ uv run pytest -m "not integration" -q
[...]
883 passed, 2 skipped, 57 deselected in 4.69s

$ uv run pytest -m "not integration" -q -W error::DeprecationWarning
[...]
883 passed, 2 skipped, 57 deselected in 4.71s
# (no unwrapped DeprecationWarning anywhere in the offline suite)

$ uv run python -c "
import warnings
from t0_agents.compose import race
async def stub(x): return x
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    race(stub, stub)
    assert any(issubclass(item.category, DeprecationWarning) for item in w), w
    print('OK — DeprecationWarning fired:', w[0].message)
"
OK — DeprecationWarning fired: `race(...)` is deprecated and will
be removed in v1.1; use `parallel(*, reducer=...,
error_policy='continue')` for fan-out + merge or
`asyncio.wait(return_when=FIRST_COMPLETED)` in user code for
first-finisher-wins.

$ git log --oneline experimental/glow ^main
0071177 chore(compose): document race deprecation in CHANGELOG
95db857 docs(compose): demote race to advanced / not-recommended appendix
8f452f2 deprecate(compose): emit DeprecationWarning from race() and assert it in tests
7fde060 docs(glow): add milestone specs under docs/internals/glow
768643d chore(glow): scaffold experimental sub-package
# 3 M0 commits + 2 pre-existing scaffold commits = 5 total on the branch.

$ git diff --stat 7fde060..HEAD   # diff from immediately-prior glow scaffold
 CHANGELOG.md                                   | 10 ++++++++++
 docs/composition.md                            | 79 ++++++++++++++++++++++----
 src/t0_agents/compose/builtins.py              | 17 ++++++
 tests/test_compose/v1/test_cancellation.py     | 10 ++++----
 tests/test_compose/v1/test_nested_composers.py | 12 +++++++----
 tests/test_compose/v1/test_race.py             | 19 +++++++++++++-----
 6 files changed, 116 insertions(+), 31 deletions(-)
# Only the M0-allowed paths plus the spec's M0 exception (compose/_runtime.py
# unchanged — the cancellation cascade implementation didn't need to move).

$ git diff src/t0_agents/mcp/
# (empty — mcp/ submodule contract preserved per feedback_mcp_standalone)

$ grep -rn "race(" examples/
# (empty — examples never used race; spec invariant holds)
```

The full pre-push `uv run pytest -q` (no marker filter) is the
user's responsibility per `feedback_full_test_suite_before_push`;
docker compose up is required for the integration marker. M0
introduces no new integration tests, and the existing ones depend
on TZ gateway state, so the offline regression above is the
strongest signal Claude can produce without docker.
