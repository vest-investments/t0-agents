# M2 — `delete-and-examples`

> **Status:** Not started.
> **Branch:** `experimental/glow` (topic branch: `compose-removal/m2-delete-and-examples`).
> **Dependencies:** M1 must have landed on `experimental/glow`
> (sub-agent code lives in `delegation/`).
> **Blocks:** none. This is the final milestone of the workstream.

## Purpose

The big deletion **plus** the 6 canonical pattern examples. After M2:

- `src/t0_agents/compose/` no longer exists.
- `src/t0_agents/glow/compose.py` no longer exists; `glow` no longer
  exposes `compose`.
- 13 topology symbols removed from `t0_agents/__init__.py`.
- Topology tests deleted (~85 tests), `tests/test_glow/test_compose.py`
  deleted (18 tests), 2 dict-router integration tests deleted.
- 6 new examples added — Sequential / Parallel / Orchestrator × Agent
  legacy + glow APIs — each demonstrating where ad-hoc code can be
  inserted and how the pattern nests.
- CHANGELOG has a `### Removed` block under the next-release section.

The examples are the **positive deliverable** of this workstream:
without them, the deletion just feels like a regression. With them,
the message to internal developers is clear — "yes, you can build
these patterns; here's how; you can drop custom code anywhere in the
middle because no abstraction is forcing a shape; and they nest
trivially."

## Scope

### In

#### Code deletion

- Delete `src/t0_agents/compose/builtins.py` (~729 LOC).
- Delete `src/t0_agents/compose/_runtime.py` (~735 LOC).
- Delete `src/t0_agents/compose/topology.py` (~306 LOC).
- Delete `src/t0_agents/compose/__init__.py` (~87 LOC).
- Remove the `src/t0_agents/compose/` directory itself.
- Delete `src/t0_agents/glow/compose.py` (~450 LOC).
- Update `src/t0_agents/glow/__init__.py`: remove the import
  `from t0_agents.glow.compose import compose` (around line 21) and the
  `"compose"` entry in `__all__` (around line 50).
- Update `src/t0_agents/__init__.py`: remove the 13 topology symbols
  from the import block (lines ~7-21) and from `__all__`
  (`Composable`, `Composer`, `ComposerError`, `CompositeRun` if
  exported, `circuit_breaker`, `conditional`, `consensus`,
  `map_reduce`, `parallel`, `race`, `retry_chain`, `saga`,
  `sequential`). Note: `agent_as_tool` already lives in `delegation/`
  post-M1 — no changes for it.

#### Tests deletion

- Delete the 11 topology test files in `tests/test_compose/`:
  - `test_sequential.py`
  - `test_parallel.py`
  - `test_race.py`
  - `test_conditional.py`
  - `test_consensus.py`
  - `test_map_reduce.py`
  - `test_saga.py`
  - `test_retry_chain.py`
  - `test_circuit_breaker.py`
  - `test_cancellation.py`
  - `test_nested_composers.py`
- After the deletions, remove the `tests/test_compose/` directory itself
  (it should be empty: M1 moved out the 4 sub-agent test files, and M2
  deletes the 11 topology test files).
- Delete `tests/test_glow/test_compose.py` (18 tests).
- Edit `tests/test_glow/test_integration.py`: delete the two tests
  `test_llm_routed_dict_composer_dispatches_to_picked_route` (lines
  148-186) and `test_llm_routed_dict_composer_supports_composer_routes`
  (lines 189-226). Also delete any imports they were the sole consumer
  of (typically `from t0_agents.glow import compose`).

#### Examples deletion

- Delete `examples/12_concurrency/01_parallel_tools.py`.
- Delete `examples/12_concurrency/02_parallel_subagents.py`.
- Delete `examples/glow/04_composition.py`.

#### Examples — 6 new canonical files

**Premise common to all 6 examples:**

- Runnable end-to-end against the TZ gateway with Groq's
  `openai/gpt-oss-20b` (no `tool_choice="required"` needed anywhere).
- Top-of-file docstring explains the pattern + when to use it.
- The body **explicitly shows at least one point where ad-hoc code
  can be inserted** (a transform, a validation, a branch, a log).
  Comment it: `# Ad-hoc step — anything you want; no abstraction
  forces a shape here.`
- Closing comment explains how the pattern nests: "call from another
  `async def`, expose as MCP tool via `fastmcp.Tool.from_function`,
  or embed inside an orchestrator's `sub_agents=[...]`."
- `if __name__ == "__main__": asyncio.run(main())` entry point that
  takes a configurable input and prints the output.

##### `examples/12_concurrency/01_sequential.py` (legacy Agent API)

Pipeline of 2-3 Agents in series. Output of one → input of the next.
Ad-hoc step in the middle (truncate, normalize, branch on a condition).

Reference structure:

```python
async def summarize_then_translate(query: str) -> str:
    summary = (await summarizer.invoke(query).result()).response
    # Ad-hoc step — anything you want; no abstraction forces a shape here.
    if len(summary) > 500:
        summary = summary[:500] + "..."
    translation = (await translator.invoke(summary).result()).response
    return translation
```

Closing comment: nesting is just calling `await summarize_then_translate(query)`
from another async function.

##### `examples/12_concurrency/02_parallel.py` (legacy Agent API)

`asyncio.gather` over multiple `Agent.invoke()` calls with
`return_exceptions=True`. A reducer function combines the results
(concat, majority vote, drop errors).

Reference structure:

```python
results = await asyncio.gather(
    optimist.invoke(query).result(),
    pessimist.invoke(query).result(),
    pragmatist.invoke(query).result(),
    return_exceptions=True,
)
successful = [r.response for r in results if not isinstance(r, BaseException)]
# Ad-hoc reducer: pick one, concatenate, vote, etc.
merged = "\n---\n".join(successful)
```

##### `examples/12_concurrency/03_orchestrator.py` (legacy Agent API)

`AgentConfig(sub_agents=[a, b, c])` with the parent LLM driving
dispatch. Replaces the `compose({"a": ..., "b": ...})` case. System
prompt: "Route the user query to the right sub-agent and return its
output verbatim." Demonstrates reuse — the same sub-agents can be
embedded in multiple orchestrators. Includes an **explicit note** on
`max_turns` as the anti-recursion guardrail.

##### `examples/glow/04_sequential.py` (glow API)

Same pattern as `12_concurrency/01_sequential.py` but with `@app.agent`
decorators. Note on file numbering: `04_composition.py` is deleted in
this milestone; this new `04_sequential.py` takes its slot. If the
existing `05_di_tools.py` and `06_mcp_typing.py` need to remain at
their numbers, slot the three glow pattern examples at 04/05/06 and
shift the existing ones to 07/08 — **confirm renumbering with the user
during M2** before doing it.

##### `examples/glow/05_parallel.py` (glow API)

Same pattern as `12_concurrency/02_parallel.py` but using
`@app.agent`-decorated callables awaited via `asyncio.gather`.

##### `examples/glow/06_orchestrator.py` (glow API)

Same pattern as `12_concurrency/03_orchestrator.py` but with
`@app.agent` and `sub_agents=[a, b, c]` on the spec. Verify in
`src/t0_agents/glow/decorators.py` that `sub_agents` is exposed in the
spec passthrough before relying on it; if not, this example uses the
escape hatch (`agent._agent.config.sub_agents = [...]` is the
documented escape).

#### Docs deletion / update

- Delete `docs/composition.md` (606 lines).
- Edit `docs/glow/limitations.md`: delete the "LLM-routed dict-composer
  model capability" section (lines 53-67). If that was the only section
  in the file, delete the whole file.
- Edit `README.md`: remove mentions of "Composers" / "topology
  primitives" from the feature list. Replace with a brief mention of
  the canonical patterns (Sequential / Parallel / Orchestrator) and
  pointers to `examples/12_concurrency/` and `examples/glow/`.
- Edit `docs/glow/getting-started.md`: rewrite the "Native composition"
  section to point at the new examples instead of describing
  `compose()`.
- Edit `docs/multi-agent.md`: confirm references match `delegation/`
  post-M1 (M1 already did this; verify).
- Edit `docs/migration-v0.12-to-v1.0.md` if it references compose
  primitives — strike or rephrase.

#### CHANGELOG

Under the next-release section (currently `## Unreleased` or whichever
header is in use), add:

```markdown
### Removed

- **`compose` topology primitives** — the 9 factories
  (`sequential`, `parallel`, `race`, `conditional`, `consensus`,
  `map_reduce`, `saga`, `retry_chain`, `circuit_breaker`) and the
  `Composable` / `Composer` / `CompositeRun` / `ComposerError`
  protocol layer. Use raw `asyncio.gather` + `await agent.invoke().result()`
  chaining for Sequential/Parallel, and `AgentConfig(sub_agents=[...])`
  with a routing system prompt for the dispatch case. See
  `examples/12_concurrency/{01_sequential,02_parallel,03_orchestrator}.py`
  and `examples/glow/{04_sequential,05_parallel,06_orchestrator}.py`.
- **`t0_agents.glow.compose`** — removed alongside the underlying
  topology layer.
- **Sub-agent orchestration relocated** — `agent_as_tool`,
  `install_sub_agents`, and friends now live in
  `t0_agents.delegation` (top-level re-exports unchanged).
```

No separate migration guide. The commit history is the documentation.

### Out

- No new helpers in `delegation/` (`composite_agent`, `func_as_tool`
  are explicitly **anti-features** per the plan file).
- No changes to `src/t0_agents/mcp/`.
- No changes to `src/t0_agents/delegation/` (stable since M1; verify
  with `git diff src/t0_agents/delegation/`).
- No deprecation warnings (no external users to warn).

## Checklist

- [ ] **Commit 1: `refactor(t0_agents): remove compose topology primitives from public API`**
  - [ ] Edit `src/t0_agents/__init__.py`: delete the 13 topology
        imports and the 13 corresponding `__all__` entries.
  - [ ] Verify `from t0_agents import agent_as_tool, SubAgentMiddleware,
        SubAgentError` still works.
  - [ ] Verify a clean `import t0_agents` from a REPL.

- [ ] **Commit 2: `refactor(glow): remove compose() from public API`**
  - [ ] Edit `src/t0_agents/glow/__init__.py`: delete the
        `from t0_agents.glow.compose import compose` line and the
        `"compose"` entry from `__all__`.
  - [ ] Verify the other 13 glow exports (`App`, `Depends`, `Done`,
        `Error`, `FluentRun`, `MCPServer`, `Selector`, `Text`,
        `Thought`, `ToolCall`, `ToolResult`, `TurnEnd`,
        `regex_guardrail`) still work.

- [ ] **Commit 3: `chore(compose-removal): remove compose/ directory`**
  - [ ] Delete `src/t0_agents/compose/builtins.py`.
  - [ ] Delete `src/t0_agents/compose/_runtime.py`.
  - [ ] Delete `src/t0_agents/compose/topology.py`.
  - [ ] Delete `src/t0_agents/compose/__init__.py`.
  - [ ] `rmdir src/t0_agents/compose/`.
  - [ ] Delete `src/t0_agents/glow/compose.py`.
  - [ ] Verify `uv run mypy src/` still passes.

- [ ] **Commit 4: `test(compose-removal): remove topology tests`**
  - [ ] Delete the 11 topology test files in `tests/test_compose/`
        (see Scope list above).
  - [ ] `rmdir tests/test_compose/` (must be empty post-M1's moves +
        these deletes).
  - [ ] Verify `uv run pytest -m "not integration"` is green.

- [ ] **Commit 5: `test(glow): remove dict-router integration tests`**
  - [ ] Delete `tests/test_glow/test_compose.py`.
  - [ ] Edit `tests/test_glow/test_integration.py`: delete the two
        `test_llm_routed_dict_composer_*` tests at lines 148-186 and
        189-226 (and any imports they were the sole consumer of).
  - [ ] Verify `uv run pytest -m "not integration"` is green.
  - [ ] Verify `uv run pytest -m integration` (after
        `docker compose up -d`) is green — the 2 Groq-failing tests
        are now gone.

- [ ] **Commit 6: `chore(examples): remove obsolete composition examples`**
  - [ ] Delete `examples/12_concurrency/01_parallel_tools.py`.
  - [ ] Delete `examples/12_concurrency/02_parallel_subagents.py`.
  - [ ] Delete `examples/glow/04_composition.py`.

- [ ] **Commit 7: `feat(examples): canonical sequential pattern (Agent legacy)`**
  - [ ] Write `examples/12_concurrency/01_sequential.py` (~80-120 LOC).
  - [ ] Smoke-run end-to-end against the gateway with Groq.

- [ ] **Commit 8: `feat(examples): canonical parallel pattern (Agent legacy)`**
  - [ ] Write `examples/12_concurrency/02_parallel.py` (~80-120 LOC).
  - [ ] Smoke-run.

- [ ] **Commit 9: `feat(examples): canonical orchestrator pattern (Agent legacy)`**
  - [ ] Write `examples/12_concurrency/03_orchestrator.py` (~100-150 LOC)
        using `AgentConfig(sub_agents=[...])`.
  - [ ] Smoke-run.

- [ ] **Commit 10: `feat(examples): canonical patterns for glow (sequential, parallel, orchestrator)`**
  - [ ] Write `examples/glow/04_sequential.py`.
  - [ ] Write `examples/glow/05_parallel.py`.
  - [ ] Write `examples/glow/06_orchestrator.py` (using `@app.agent`
        with `sub_agents=[...]`).
  - [ ] Decide example numbering — if the new numbers clash with
        existing files (`05_di_tools.py`, `06_mcp_typing.py`), confirm
        renumbering with the user before doing it.
  - [ ] Smoke-run all three.

- [ ] **Commit 11: `docs(compose-removal): rewrite affected docs`**
  - [ ] Delete `docs/composition.md`.
  - [ ] Trim `docs/glow/limitations.md` (or delete entirely if empty
        after trim).
  - [ ] Update `README.md` feature list.
  - [ ] Rewrite the "Native composition" section in
        `docs/glow/getting-started.md`.
  - [ ] Verify `docs/multi-agent.md` is consistent post-M1.
  - [ ] Update `docs/migration-v0.12-to-v1.0.md` if it referenced compose.

- [ ] **Commit 12: `chore(compose-removal): CHANGELOG → ### Removed`**
  - [ ] Add the `### Removed` block to `CHANGELOG.md` under the
        next-release section (see Scope for exact wording).
  - [ ] Run all verification commands (see "Verification commands"
        below) and capture outputs for the dev-log.

## Acceptance criteria

- `git grep -nE "from t0_agents\.compose|import.*\bcompose\b" src/ tests/`
  returns no live import matches (docstrings/strings/historical
  comments in docs are OK).
- `uv run pytest -m "not integration"` is green.
- `docker compose up -d` + `uv run pytest -m integration` is green
  (the 2 dict-router tests no longer exist; the remaining glow
  integration tests pass).
- All 6 new examples run end-to-end against the gateway with Groq and
  produce sensible output (manual verification, captured in dev-log §5).
- `uv run mypy src/` strict passes.
- `uv run ruff check src/` passes.
- `git diff --stat src/t0_agents/mcp/` is empty.
- `git diff --stat src/t0_agents/delegation/` is empty vs the post-M1
  state.
- `src/t0_agents/__init__.py.__all__` no longer lists `Composable`,
  `Composer`, `ComposerError`, `CompositeRun` (if it was there),
  `circuit_breaker`, `conditional`, `consensus`, `map_reduce`,
  `parallel`, `race`, `retry_chain`, `saga`, `sequential`.
- `src/t0_agents/glow/__init__.py.__all__` no longer lists `compose`.
- `CHANGELOG.md` has a `### Removed` block listing the removed symbols.

## Verification commands

```bash
# On the topic branch compose-removal/m2-delete-and-examples off
# experimental/glow (with M1 already merged into experimental/glow):

# 1. Final grep — no live compose imports anywhere in src/ or tests/:
git grep -nE "from t0_agents\.compose|import.*\bcompose\b" src/ tests/

# 2. Verify the public API delta:
uv run python -c "
import t0_agents
removed = {'Composable', 'Composer', 'ComposerError', 'CompositeRun',
           'circuit_breaker', 'conditional', 'consensus', 'map_reduce',
           'parallel', 'race', 'retry_chain', 'saga', 'sequential'}
overlap = removed & set(t0_agents.__all__)
assert not overlap, f'still exported: {overlap}'
import t0_agents.glow
assert 'compose' not in t0_agents.glow.__all__, 'glow still exports compose'
print('OK — public API clean')
"

# 3. Verify delegation/ is unchanged from M1:
git diff src/t0_agents/delegation/   # must be empty

# 4. Verify mcp/ invariant:
git diff src/t0_agents/mcp/   # must be empty

# 5. Lint / type / offline tests:
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest -m "not integration"

# 6. Integration tests (requires docker + GROQ_API_KEY in .env):
docker compose up -d
uv run pytest -m integration

# 7. Smoke-run each of the 6 new examples manually:
uv run python examples/12_concurrency/01_sequential.py
uv run python examples/12_concurrency/02_parallel.py
uv run python examples/12_concurrency/03_orchestrator.py
uv run python examples/glow/04_sequential.py
uv run python examples/glow/05_parallel.py
uv run python examples/glow/06_orchestrator.py

# 8. Commit count:
git log --oneline experimental/glow..HEAD | wc -l   # should be in [8,12]
```

## Cross-references

- **Plan file:** `~/.claude/plans/compose-removal-plan.md` §"What's
  being removed vs preserved" and §"Implementation plan — milestones".
- **Investigation report:**
  `~/.claude/plans/quiero-quitar-compose-la-drifting-nygaard.md` §3
  catalogues every file being deleted.
- **M1 dev-log:**
  `docs/internals/compose-removal/logs/milestone-1-dev-log.md` — must
  be read first; §4 Carry-forward identifies the exact M1 end-state
  this milestone builds on.
- **Source files deleted:**
  - `src/t0_agents/compose/builtins.py` (~729 LOC).
  - `src/t0_agents/compose/_runtime.py` (~735 LOC).
  - `src/t0_agents/compose/topology.py` (~306 LOC).
  - `src/t0_agents/compose/__init__.py` (~87 LOC).
  - `src/t0_agents/glow/compose.py` (~450 LOC).
- **Examples deleted:**
  - `examples/12_concurrency/01_parallel_tools.py`.
  - `examples/12_concurrency/02_parallel_subagents.py`.
  - `examples/glow/04_composition.py`.
- **Tests deleted (~85 + 18 + 2 = ~105 tests total):**
  - 11 `tests/test_compose/test_*.py` topology tests.
  - `tests/test_glow/test_compose.py` (18 tests).
  - 2 tests in `tests/test_glow/test_integration.py:148-226`.
- **Docs deleted/edited:**
  - `docs/composition.md` (deleted).
  - `docs/glow/limitations.md` §53-67 (deleted or whole file removed).
  - `README.md`, `docs/glow/getting-started.md`,
    `docs/migration-v0.12-to-v1.0.md` (edits).
- **CHANGELOG:** `CHANGELOG.md` — new `### Removed` block under the
  next-release section.

## Dev-log

When the milestone completes, drop
`docs/internals/compose-removal/logs/milestone-2-dev-log.md` with the
five-section template (Summary · Deviations · Open questions ·
Carry-forward · Verification).

This is the **final milestone of the workstream**. §4 Carry-forward
should either be empty (`No carry-forward notes for subsequent
milestones.`) or note an explicit handoff for a future merge-to-main
decision (e.g., "`experimental/glow` is ready to merge to `main`
pending the user's release-strategy decision").
