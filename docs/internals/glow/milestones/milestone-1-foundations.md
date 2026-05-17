# M1 — `foundations`

> **Status:** Commit 1 (scaffold) ✅ done — `768643d`. Commits 2–3
> pending.
> **Branch:** `experimental/glow`.
> **Dependencies:** none (M0 is recommended-but-not-required first).
> **Blocks:** M2 needs `output_inference` and `FluentRun`.

## Purpose

Build the unit-testable foundations of glow — the empty package
scaffold (already done), the type-hint → `OutputMode` inference, and
the `FluentRun[T]` lazy wrapper. **Nothing in this milestone needs
a TensorZero gateway.** Everything runs in-process against mocked
`Agent` and `Run` objects, which gives us a fast iteration loop and
fully covered code before any integration work begins.

The milestone also lands the **milestone specs themselves** in
`docs/internals/glow/` (this file and its siblings), so subsequent
milestones can reference them.

## Scope

### In

- ✅ Empty scaffold: `src/t0_agents/glow/__init__.py`,
  `py.typed`, `tests/test_glow/__init__.py`,
  `examples/glow/README.md`, CHANGELOG entry (commit 1 = `768643d`).
- The milestone documentation in `docs/internals/glow/` (this file
  + 5 siblings + README + empty `logs/`).
- `src/t0_agents/glow/output_inference.py` — pure function
  `output_mode_from_return_type(fn) -> OutputMode[Any]`.
- `src/t0_agents/glow/handle.py` — `FluentRun[T]` lazy wrapper
  with `.text()`, `.json()`, `.json(M)`, `.as_(mode)`, `.events()`,
  `__await__`, `result()`.
- `src/t0_agents/glow/_typing.py` — internal helpers
  (`_strip_annotated`, `_unwrap_optional`, `_is_dict_alias`,
  `_is_pydantic_model`, `_is_list_of_pydantic`,
  `_wrap_list_model`).
- Unit tests in `tests/test_glow/test_output_inference.py` and
  `tests/test_glow/test_fluent_run.py`.

### Out

- Anything that touches `Agent.invoke()` — that's M2.
- Public `App` class — M2.
- Decorators (`@app.agent`, `@app.tool`) — M2/M3.
- Pydantic optional-import dance — pydantic is a hard dep of the
  `[glow]` extra; in tests we install it.

## Checklist

- [ ] **Commit 2: feat(glow): `output_inference` — return-type → OutputMode**
  - [ ] Implement `_typing.py` helpers:
        - [ ] `_strip_annotated(tp)` — peel `Annotated[T, ...]`.
        - [ ] `_unwrap_optional(tp)` — convert `Optional[T]` and
              `T | None` to `T`.
        - [ ] `_is_dict_alias(tp)` — match `dict`, `dict[str, Any]`,
              `Mapping[str, Any]`.
        - [ ] `_is_pydantic_model(tp)` — `isinstance(tp, type) and
              issubclass(tp, pydantic.BaseModel)`.
        - [ ] `_is_list_of_pydantic(tp)` — origin is `list` and
              args[0] is a Pydantic model.
        - [ ] `_wrap_list_model(tp)` — synthesize a Pydantic
              wrapper model with a single `items: list[T]` field
              (use `pydantic.create_model`).
  - [ ] Implement `output_inference.py`:
        ```python
        def output_mode_from_return_type(fn) -> OutputMode[Any]:
            hints = get_type_hints(fn, include_extras=True, localns=fn.__globals__)
            rt = _strip_annotated(_unwrap_optional(hints.get("return", str)))
            if rt is str or rt is type(None): return ChatOutput()
            if _is_dict_alias(rt):            return JsonOutput()
            if _is_pydantic_model(rt):        return StructuredOutput(rt)
            if _is_list_of_pydantic(rt):      return StructuredOutput(_wrap_list_model(rt))
            return ChatOutput()
        ```
  - [ ] Surface `ConfigurationError` (re-use the v1.0 exception
        from `t0_agents.core.exceptions`) when `get_type_hints`
        fails (forward-ref resolution problems). Fail at
        *registration* time, not invocation.
  - [ ] Unit tests in `tests/test_glow/test_output_inference.py`:
        - [ ] `def fn() -> str:` → `ChatOutput`
        - [ ] `def fn() -> dict:` → `JsonOutput`
        - [ ] `def fn() -> dict[str, Any]:` → `JsonOutput`
        - [ ] `def fn() -> MyModel:` → `StructuredOutput(MyModel)`
        - [ ] `def fn() -> list[MyModel]:` → `StructuredOutput`
              over a synthetic wrapper model
        - [ ] `def fn() -> Optional[MyModel]:` → as `MyModel`
        - [ ] `def fn() -> MyModel | None:` → as `MyModel`
        - [ ] `def fn() -> "ForwardRef":` (with the class defined
              in the same module) — resolves correctly
        - [ ] `def fn() -> "MissingType":` — raises
              `ConfigurationError` at *registration* time
        - [ ] `def fn():` (no annotation) → `ChatOutput` with a
              DEBUG log entry

- [ ] **Commit 3: feat(glow): `FluentRun[T]` lazy wrapper**
  - [ ] Implement `handle.py`:
        - [ ] `FluentRun(build: Callable[[OutputMode], Run], default_output: OutputMode[T])`.
        - [ ] `.text()` → new `FluentRun` with `ChatOutput()`.
        - [ ] `.json(model=None)` → new `FluentRun` with
              `JsonOutput()` or `StructuredOutput(model)`.
        - [ ] `.as_(mode)` → new `FluentRun` with the supplied
              `OutputMode`.
        - [ ] `.events(*selectors)` → builds `Run` immediately
              and returns an async iterator (selectors are
              implemented in M2; for M1, stub to yield every
              event).
        - [ ] `__await__` → calls `build(chosen_output)`,
              `await run.result()`, returns
              `result.value` (the unwrapped `T`).
        - [ ] `async def result()` → calls `build()` and returns
              the full `AgentResult[T]`.
        - [ ] `.json(A).json(B)` semantics: last-wins; log at
              DEBUG when overridden.
  - [ ] Unit tests in `tests/test_glow/test_fluent_run.py`:
        - [ ] Stub `Agent` whose `invoke(...)` returns a stub
              `Run` with an `AgentResult(value="hi")` — verify
              `await FluentRun(...)` returns `"hi"`.
        - [ ] `.json()` chains: `.json().json(Model)` uses
              `StructuredOutput(Model)`.
        - [ ] `.json(A).json(B)` — last-wins (B is used).
        - [ ] `.text().json(M)` — output mode swaps freely.
        - [ ] `.events()` builds the `Run` immediately; awaiting
              after `.events()` is well-defined (matches existing
              `Run` contract — second consumer joins the cached
              buffer).
        - [ ] `await fluent_run.result()` returns the full
              `AgentResult` (with `episode_id`, etc.).

- [ ] **Commit M1 milestone docs: docs(glow): milestone specs**
  - [ ] This file and its 5 siblings already exist on disk after
        the milestone-doc commit; check with `git status` that
        every file in `docs/internals/glow/` is tracked.
  - [ ] `docs/internals/glow/README.md` links to all six
        milestone specs.

- [ ] **Run validation:**
  ```bash
  uv run ruff check src/ tests/
  uv run mypy src/
  uv run pytest -m "not integration"
  ```

## Acceptance criteria

- Every case in `test_output_inference.py` passes; the
  forward-ref-failure case raises `ConfigurationError` at decoration
  time (verified by constructing a function that references a name
  not in its module's globals and asserting the call to
  `output_mode_from_return_type` raises).
- `test_fluent_run.py` passes with a mock `Agent` (no TZ gateway).
  Specifically: `await FluentRun(...)` returns the unwrapped value
  (not `AgentResult`); `await fluent_run.result()` returns the full
  `AgentResult`; the `.json(A).json(B)` last-wins rule holds.
- `mypy --strict` is clean on every new file.
- `ruff` is clean.

## Verification commands

```bash
uv run ruff check src/t0_agents/glow/ tests/test_glow/
uv run mypy src/t0_agents/glow/
uv run pytest tests/test_glow/ -q
uv run pytest -m "not integration" -q     # full offline regression
```

## Cross-references

- Plan §"Technical architecture" → "Type-hint → `OutputMode` mapping"
  and "`FluentRun[T]` — lazy wrapper, not subclass" at
  `~/.claude/plans/la-idea-es-tener-glowing-gem.md`.
- v1.0 source to re-use (NOT modify):
  `src/t0_agents/core/agent.py:227-293` (`Agent.invoke(eager=False, output=...)`),
  `src/t0_agents/core/run.py:79` (`Run[T]` contract),
  `src/t0_agents/core/outputs/json.py:22` (`JsonOutput`),
  `src/t0_agents/core/outputs/structured.py:47` (`StructuredOutput`),
  `src/t0_agents/core/outputs/chat.py` (`ChatOutput`),
  `src/t0_agents/core/exceptions.py` (`ConfigurationError`).
- Pydantic: `create_model` for synthetic wrapper models
  (`_wrap_list_model`).

## Dev-log

When the milestone completes, drop `logs/milestone-1-dev-log.md`
with the five-section template.
