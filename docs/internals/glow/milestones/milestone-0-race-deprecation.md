# M0 — `race-deprecation`

> **Status:** Not started.
> **Branch:** `experimental/glow`.
> **Dependencies:** none.
> **Blocks:** none (independent). Recommended *before* M1 so the
> design philosophy is anchored in committed code.

## Purpose

Deprecate the low-level `race` composer so glow can confidently
treat `set[Composable]` as **fan-out + merge** rather than
first-finisher-cancels-the-rest. Cancelling sibling agents that are
still working is a footgun: results are thrown away, costs are
already paid (the inferences kicked off, the tokens burned), and
the cancellation cascade is best-effort (orphans get logged but
leaked). Landing the `DeprecationWarning` in v1.0 and demoting
`race` in the public docs keeps the framework story coherent
between low-level and glow.

This milestone is intentionally tiny and standalone — it touches
the low-level package only, so glow contributors and downstream
users see the policy reflected in committed code before glow's
`compose()` ships.

## Scope

### In

- `DeprecationWarning` raised by `race()` factory pointing at
  `parallel(*nodes, reducer=..., error_policy="continue")` as the
  recommended replacement.
- Docs demotion: rewrite `docs/composition.md` to move `race` from
  the recommended-patterns table to an "Advanced / not
  recommended" appendix.
- `CHANGELOG.md` — add a `Deprecated` subsection under the v1.0
  Unreleased block flagging the warning.
- Existing tests in `tests/test_compose/v1/` that exercise `race`
  are wrapped in `pytest.warns(DeprecationWarning)` so the suite
  stays green and the warning is positively asserted.

### Out

- Actual removal of `race` (slated for v1.1).
- Any change to `parallel`/`consensus`/`map_reduce`/etc. semantics.
- Touching `src/t0_agents/__init__.py` (the public re-export of
  `race` stays in v1.0 so we don't hard-break consumers).
- Any glow code — that's M1 onwards.

## Checklist

- [ ] **Commit 0: deprecate(compose): mark `race` as deprecated**
  - [ ] Add `warnings.warn(DeprecationWarning(...), stacklevel=2)`
        at the top of the `race()` factory in
        `src/t0_agents/compose/builtins.py` (currently lines
        259–277). Message: "`race(...)` is deprecated and will be
        removed in v1.1; use `parallel(*, reducer=..., error_policy='continue')`
        for fan-out + merge or `asyncio.wait(return_when=FIRST_COMPLETED)`
        in user code for first-finisher-wins."
  - [ ] Update `tests/test_compose/v1/test_race.py` (5 tests):
        wrap each `race(...)` call in
        `with pytest.warns(DeprecationWarning):` to keep the suite
        green and assert the warning fires.
  - [ ] Update `tests/test_compose/v1/test_nested_composers.py:31`:
        replace the `sequential(race(...), ...)` line with
        `sequential(parallel(..., error_policy='continue'), ...)`
        so the deprecation warning doesn't pollute that test.
  - [ ] Update `tests/test_compose/v1/test_cancellation.py`:
        either wrap with `pytest.warns(DeprecationWarning)` or
        rewrite to test cancellation through `parallel` with
        `error_policy='fail_fast'` (whichever is closer to the
        original test intent).
  - [ ] Rewrite `docs/composition.md`:
        - Move the `race` row in the patterns table (around line
          66) into an "Advanced / not recommended" appendix at the
          bottom of the file.
        - Rewrite the "Race" section (around lines 119–131) to lead
          with the rationale for deprecation and point at
          `parallel(reducer=..., error_policy="continue")` as the
          replacement.
        - Add a one-line note to the file's intro mentioning the
          deprecation and linking to the appendix.
  - [ ] Add to `CHANGELOG.md` under v1.0 Unreleased, a new
        `### Deprecated` subsection:
        ```markdown
        ### Deprecated

        - **`race(*nodes)` composer** — emits `DeprecationWarning`;
          removal scheduled for v1.1. Use `parallel(*nodes,
          reducer=..., error_policy="continue")` for fan-out +
          merge, or write `asyncio.wait(return_when=FIRST_COMPLETED)`
          in user code if you need first-finisher-wins.
        ```
  - [ ] Run validation locally:
        ```bash
        uv run ruff check src/ tests/
        uv run mypy src/
        uv run pytest -m "not integration"
        ```

## Acceptance criteria

- `uv run pytest -m "not integration"` is green with no new
  warnings beyond the `DeprecationWarning`s explicitly asserted by
  `pytest.warns(...)`.
- Calling `race(a, b)` in a fresh REPL emits exactly one
  `DeprecationWarning` whose message points at the `parallel(...)`
  replacement.
- `grep -r "race(" examples/` returns zero matches (confirmed: no
  example uses `race` today).
- `docs/composition.md` no longer recommends `race` in its
  primary patterns table.
- `CHANGELOG.md` documents the deprecation under v1.0 Unreleased.

## Verification commands

```bash
# In the repo root, on experimental/glow:
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest -m "not integration"

# Quick REPL check (optional):
uv run python -c "
import warnings
from t0_agents.compose import race
async def stub(x): return x
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    race(stub, stub)
    assert any(issubclass(item.category, DeprecationWarning) for item in w), w
    print('OK — DeprecationWarning fired')
"
```

## Cross-references

- Plan file §"Implementation plan — commits" item 0 and §"Touched
  in low-level (commit 0, deprecation only — no behavior change)"
  at `~/.claude/plans/la-idea-es-tener-glowing-gem.md`.
- v1.0 source: `src/t0_agents/compose/builtins.py:259-277` (the
  `race()` factory currently in production); `src/t0_agents/compose/_runtime.py:277-350`
  (the cancellation cascade implementation that motivates the
  deprecation).
- Replacement primitive: `src/t0_agents/compose/builtins.py:189-217`
  (`parallel(*nodes, reducer, error_policy)`).
- Existing test suite to amend: `tests/test_compose/v1/test_race.py`,
  `tests/test_compose/v1/test_nested_composers.py`,
  `tests/test_compose/v1/test_cancellation.py`.

## Dev-log

When the milestone completes, drop `logs/milestone-0-dev-log.md`
with the five-section template (Summary · Deviations · Open
questions · Carry-forward · Verification).
