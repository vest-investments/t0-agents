# M1 — `foundations` — dev-log

> **Milestone:** `milestone-1-foundations` ([spec](../milestones/milestone-1-foundations.md)).
> **Branch:** `experimental/glow`.
> **Completed:** 2026-05-16.

## §1 Summary — what was done

M1 lands the two unit-testable foundations the headline DX in M2 will
compile to:

1. **Return-type → `OutputMode` inference** — `output_inference.py`
   exposes `output_mode_from_return_type(fn)` and is fed by six private
   helpers in `_typing.py` (`_strip_annotated`, `_unwrap_optional`,
   `_is_dict_alias`, `_is_pydantic_model`, `_is_list_of_pydantic`,
   `_wrap_list_model`).  Forward-ref failures raise
   `ConfigurationError` at registration time so M2's `@app.agent` can
   surface them before the first inference.
2. **`FluentRun[T]` lazy wrapper** (`handle.py`) — owns a deferred
   build closure and exposes `.text()`, `.json()`, `.json(M)`,
   `.as_(mode)`, `.events(*selectors)`, `__await__` (returns the
   unwrapped `T`), and `.result()` (escape hatch for the full
   `AgentResult[T]`).

Glow itself does not touch any v1.0 source files; every behaviour is
exercised against in-process stubs (`_FakeRun` in the test file).

### Files touched

| Path | Change | Lines |
|---|---|---|
| `src/t0_agents/glow/_typing.py` | New — six private helpers for type introspection (Annotated peeling, Optional unwrap, dict-alias detection, Pydantic / list-of-Pydantic predicates, synthetic-wrapper builder). | +86 / -0 |
| `src/t0_agents/glow/output_inference.py` | New — `output_mode_from_return_type(fn) -> OutputMode[Any]`. Raises `ConfigurationError` on forward-ref failure; DEBUG-logs missing return annotation. | +93 / -0 |
| `src/t0_agents/glow/handle.py` | New — `FluentRun[T]` lazy wrapper. Per-call overrides return fresh instances; second override on an already-overridden FluentRun logs DEBUG (`.json(A).json(B)` last-wins). `.events()` is sync so the build closure fires at call time. | +155 / -0 |
| `src/t0_agents/glow/__init__.py` | Edit — `__all__ = ["FluentRun"]` + the import. M1 is the first additive symbol. | +2 / -1 |
| `tests/test_glow/test_output_inference.py` | New — ten cases (str / bare-dict / parameterised-dict / Pydantic model / `list[Pydantic]` wrapper / `Optional[T]` / PEP-604 `T \| None` / forward-ref-resolves / forward-ref-missing → `ConfigurationError` / no-annotation → `ChatOutput` + DEBUG log). | +141 / -0 |
| `tests/test_glow/test_fluent_run.py` | New — seven cases (plain `await`, `.json()` / `.json(M)`, `.json(A).json(B)` last-wins + DEBUG, `.text().json(M)` swap, `.events()` eager build + `await` reuses, `.result()` full AgentResult). Inline `_FakeRun` duck-types the slice of the `Run` contract `FluentRun` consumes. | +188 / -0 |
| `CHANGELOG.md` | Edit — appended two bullets under the existing `#### Experimental — t0_agents.glow` subsection; tightened the scaffold bullet (the `__all__ = []` line was now stale). | +16 / -3 |

### Public glow API delta

`t0_agents.glow.__all__` grew from `[]` to `["FluentRun"]`.
`output_mode_from_return_type` stays **internal** — its caller is M2's
`@app.agent`, not end-users.

### Commits

The milestone landed as four commits on `experimental/glow`, in order:

| # | SHA | Subject |
|---|---|---|
| 1 | `1b22598` | `feat(glow): output-mode inference from return-type hints` |
| 2 | `344d06d` | `feat(glow): FluentRun[T] lazy invocation wrapper` |
| 3 | `3dd52e7` | `chore(glow): document M1 foundations under Experimental — t0_agents.glow` |
| 4 | (this commit) | `docs(glow): milestone-1 dev-log` |

In the 14-commit roadmap from the plan file these are commits 2, 3, plus
a CHANGELOG chore and the always-write log — the doc commit that
introduces the milestone specs (`7fde060`) had already happened during
M0.

### Tests

| File | New tests |
|---|---|
| `tests/test_glow/test_output_inference.py` | 10 |
| `tests/test_glow/test_fluent_run.py` | 7 |

Suite size grew from 883 (M0) → 900 offline tests (`-m "not
integration"`). The 17 new glow tests run in ~0.11 s and need no docker.

## §2 Deviations from the milestone spec

Three small deviations forced by toolchain strictness; none of them
contradict the plan file or the spec's intent.

### 2.1 — Two casts needed for mypy strict to accept ``StructuredOutput`` as ``OutputMode[Any]``

The spec's pseudocode reads literally:

```python
if _is_pydantic_model(rt):        return StructuredOutput(rt)
if _is_list_of_pydantic(rt):      return StructuredOutput(_wrap_list_model(rt))
```

mypy strict rejects this with:

```
error: Incompatible return value type (got "StructuredOutput[Any]",
  expected "OutputMode[Any]")  [return-value]
note: Protocol member OutputMode.name expected settable variable, got
  read-only attribute
```

`StructuredOutput.name` is a `@property` (it formats `structured/<Model>`
from `self.model.__name__`); the `OutputMode` Protocol declares
`name: str` as a settable class attribute.  Other v1.0 sites avoid the
clash by letting the conversion happen at the user's call-site
(`AgentConfig(output=StructuredOutput(M))`), outside `src/`.  Inside
`src/t0_agents/glow/output_inference.py` we now `cast` explicitly:

```python
if _is_pydantic_model(rt):
    return cast(OutputMode[Any], StructuredOutput(rt))
if _is_list_of_pydantic(rt):
    return cast(OutputMode[Any], StructuredOutput(_wrap_list_model(rt)))
```

The cast is widening, not lying — `StructuredOutput` satisfies every
runtime contract of `OutputMode`.

`handle.py:json(model)` needs the same cast inside the `_spawn` call:

```python
return self._spawn(cast(OutputMode[Any], StructuredOutput(model)))
```

**Impact on subsequent milestones:** none directly.  If a follow-up
milestone wants to remove the casts, the v1.0 `OutputMode` Protocol
would need to declare `name` as a `@property` (or use
`typing.Final[str]`) — that is a v1.0 change, out of scope for glow.

### 2.2 — `_wrap_list_model` needs ``# type: ignore[valid-type]`` on ``list[inner]``

`_wrap_list_model(tp)` extracts `inner = get_args(tp)[0]`, where mypy
sees `inner: Any`.  `pydantic.create_model(..., items=(list[inner],
...))` then trips:

```
error: Variable "inner" is not valid as a type  [valid-type]
```

mypy cannot use a runtime variable as a static type expression.  The
runtime semantics are correct (`list[_Item]` is a perfectly valid
`types.GenericAlias` Pydantic v2 understands), so we silence the check
on the single line.

**Impact on subsequent milestones:** none.  If M4 ends up needing
similar dynamic field construction (e.g. for guardrail wrapper models),
the same pattern applies.

### 2.3 — Test-side adjustments to clear ruff strict-style rules

The spec's checklist phrasing — `def fn() -> "_Other":` and
`def fn() -> "MissingType":` — uses explicit string forward refs.  Ruff
`UP037` fires on those (`from __future__ import annotations` is on at
the top of the test module, so every annotation is already a string at
runtime).  Removing the quotes leaves the test semantically identical:
the `get_type_hints` call still receives a deferred string annotation
and the resolution / failure path is exercised the same way.

Similarly the `Optional[_Item]` spec test trips `UP045` ("use `X |
None`").  Keeping the explicit `Optional[T]` form is load-bearing — it
exercises the `typing.Union` origin in `_unwrap_optional`, while the
sibling PEP-604 test covers `types.UnionType`.  Suppressed with `# noqa:
UP045` plus a one-line explanation.

The no-annotation helper `_no_annotation_fn()` lives at module level so
mypy strict's `disallow_untyped_defs` can be silenced via a single
`# type: ignore[no-untyped-def]` on that helper rather than smearing it
across multiple test bodies.

**Impact on subsequent milestones:** M4's guardrail work will see
similar `Annotated[T, Guardrail(...)]` shapes in tests — keep the
`noqa` / `type: ignore` pattern local to where the shape is the test
subject, rather than relaxing the ruff or mypy config globally.

## §3 Open questions

### 3.1 — `_strip_annotated` / `_unwrap_optional` peel order under nested ``Annotated[Optional[T], meta]``

The spec literal — `rt = _strip_annotated(_unwrap_optional(rt_raw))`
— peels Optional first, then Annotated.  This handles
`Optional[Annotated[T, meta]]` cleanly:

```text
Optional[Annotated[T, meta]]
  → _unwrap_optional → Annotated[T, meta]
  → _strip_annotated → T            ✓
```

…but not the mirror case `Annotated[Optional[T], meta]`:

```text
Annotated[Optional[T], meta]
  → _unwrap_optional → unchanged (origin is Annotated, not Union)
  → _strip_annotated → Optional[T]
  → _is_pydantic_model(Optional[T]) → False
  → falls through to ChatOutput     ✗
```

M1 follows the spec literally because none of its test cases nest
`Annotated` and `Optional` — that combination only matters once M4's
guardrail-on-the-return-type pattern lands.  When M4 implements
`Annotated[T, Guardrail(...)]` extraction, the helpers will need either
(a) an iterative `while changed: rt = strip(unwrap(rt))` loop or
(b) inverted order with a second peel of Optional after stripping
Annotated.  Either is a small refactor.  Park here.

### 3.2 — `localns=fn.__globals__` kept per spec; functional

The spec's `get_type_hints(fn, include_extras=True,
localns=fn.__globals__)` is unusual — the conventional kwarg for
forward-ref scoping is `globalns`, and `get_type_hints` already pulls
`fn.__globals__` automatically when `globalns` is unset.  Passing it as
`localns` is redundant but not harmful.

The forward-ref-resolves test (`def fn() -> _Other` with `_Other`
defined at module level) passes against this exact call, so the helper
is functionally correct.  No need to swap `localns` → `globalns` in M1.
If a future milestone introduces functions whose forward refs live in
nested scopes (rare), this may revisit.

No new questions surfaced from the M0 dev-log's §3 / §4 — M0's
carry-forwards explicitly say "No carry-forward instructions for
M2/M3/M4/M5 specifically," so M1 inherits a clean slate from M0.

## §4 Carry-forward notes

For subsequent milestones:

- **M2 — call `output_mode_from_return_type(fn)` from `@app.agent` at
  *decoration* time** (not at first invocation).  Forward-ref failures
  must surface as `ConfigurationError` to the user as soon as the
  decorator fires, matching FastAPI's "fail fast on registration"
  ergonomics.  The helper already raises `ConfigurationError` from
  `t0_agents.core.exceptions`; no wrapping needed.
- **M2 — wire the inferred mode into the `FluentRun` build closure as
  `lambda mode: agent.invoke(message, output=mode, eager=False)`.**
  The `eager=False` is load-bearing: it keeps the Run object created
  without firing the drive task, so per-call overrides (`.json(M)`) can
  swap the OutputMode before the user's `await` lands.
- **M2 — for `list[BaseModel]` returns, unwrap `result.value.items`
  inside `@app.agent`'s wrapper coroutine** so the user-visible value
  matches the original `list[T]` annotation.  M1's
  `_wrap_list_model` ships a `StructuredOutput` over an
  `<inner>List(BaseModel)` class whose single field is
  `items: list[T]`; that wrapper is glow internal-only and must not
  leak.
- **M2 — implement selector wrapping for `FluentRun.events(*selectors)`.**
  M1 ignores `*selectors`; M2 should layer the
  `Thought` / `ToolCall` / `Text` / `ToolResult` / `TurnEnd` / `Done` /
  `Error` filter-map-where surface (`docs/.../milestone-2-core-decorator.md`)
  on top of the iterator returned by `FluentRun.events()`.  No change
  to `handle.py` is required if the selectors are applied externally.
- **M2 — `pyproject.toml` `[glow]` extra.**  M1 deferred adding the
  extra because no public-import path required it (`pydantic` is
  imported by `_typing.py`, but the dev env already has pydantic via
  the existing `[pydantic]` extra).  M2 lands the `App` /
  `@app.agent` user-facing surface; that is the natural moment to
  introduce `glow = ["pydantic>=2.0"]` (FastAPI precedent).
- **M2+ — keep the two `cast(OutputMode[Any], StructuredOutput(...))`
  call-sites in mind.**  If the v1.0 `OutputMode` Protocol's `name`
  attribute is ever upgraded to a `@property`, both casts can be
  dropped.  Until then, any new site that returns `StructuredOutput`
  through an `OutputMode[*]` annotation will need the same cast.
- **M4 — `Annotated[Optional[T], meta]` peel order.**  See §3.1; M1
  follows the spec literally, M4's guardrail metadata work will need
  to refactor the order (or move to an iterative peel).

## §5 Verification

Captured commands and outcomes:

```text
$ git branch --show-current
experimental/glow

$ uv run ruff check src/t0_agents/glow/ tests/test_glow/
All checks passed!

$ uv run mypy src/t0_agents/glow/
Success: no issues found in 4 source files

$ uv run mypy src/
Success: no issues found in 74 source files

$ uv run pytest tests/test_glow/ -q
.................                                                        [100%]
17 passed in 0.11s

$ uv run pytest -m "not integration" -q
[...]
900 passed, 2 skipped, 57 deselected in 4.75s
# Pre-M1: 883 passed; M1 adds 17 new tests; total = 900. No regressions.

$ git diff src/t0_agents/mcp/ | wc -l
0
# (mcp/ submodule contract preserved per feedback_mcp_standalone)

$ git log --oneline experimental/glow ^main | wc -l
11
# 2 scaffold + 4 M0 (3 commits + dev-log) + 2 M0-PR-merge artefacts
# (bc98258 phrasing alignment + 3d0b2a9 merge) + 3 M1 code/CHANGELOG
# commits. The dev-log commit (this one) will bring the total to 12.

$ git log --oneline experimental/glow ^main | head -3
3dd52e7 chore(glow): document M1 foundations under Experimental — t0_agents.glow
344d06d feat(glow): FluentRun[T] lazy invocation wrapper
1b22598 feat(glow): output-mode inference from return-type hints
# The three M1 code/CHANGELOG commits, all with the (glow) scope.

$ git diff --stat origin/main..HEAD | tail -1
 24 files changed, 2343 insertions(+), 31 deletions(-)
# All changes confined to the allowed M1 paths (src/t0_agents/glow/**,
# tests/test_glow/**, docs/internals/glow/**, CHANGELOG.md) plus the
# M0-scope paths that landed via the M0 PR merge.  No collateral
# damage outside the glow surface.

$ uv run python -c "
from t0_agents.glow import FluentRun
print('FluentRun OK:', FluentRun)

from t0_agents.glow.output_inference import output_mode_from_return_type
def fn() -> str: ...
print('infer str:', output_mode_from_return_type(fn).name)

def fn2() -> dict: ...
print('infer dict:', output_mode_from_return_type(fn2).name)

from pydantic import BaseModel
class M(BaseModel):
    x: int
def fn3() -> M: ...
print('infer M:', output_mode_from_return_type(fn3).name)
"
FluentRun OK: <class 't0_agents.glow.handle.FluentRun'>
infer str: chat
infer dict: json
infer M: structured/M
# End-to-end manual check passes; matches the plan file's expected
# output exactly.
```

The full pre-push `uv run pytest -q` (no marker filter) is the user's
responsibility per `feedback_full_test_suite_before_push`; M1
introduces no integration tests so the offline regression above is the
strongest signal Claude can produce without docker.
