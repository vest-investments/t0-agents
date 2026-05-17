# M5 — `polish`

> **Status:** Not started.
> **Branch:** `experimental/glow`.
> **Dependencies:** M1–M4 all done.
> **Blocks:** PR readiness.

## Purpose

Make glow demoable. Write the user-facing tutorial and cookbook
under `docs/glow/`, finalize the six runnable examples under
`examples/glow/`, and run the side-by-side line-count comparison
against `examples/02_agent_class/basic.py` to verify the **≥40%
line reduction** target. Document known limitations so users don't
discover them in production.

After M5, the branch is PR-ready. The user can decide whether to
open a draft PR for review or keep dogfooding before exposing it.

## Scope

### In

- `docs/glow/getting-started.md` — the cookbook walking through
  the plan's 10 examples (1 → 10) with explanation, code, and
  output excerpts.
- `docs/glow/limitations.md` — known limitations (streaming JSON,
  `AsyncIterator` tools, per-agent MCP subsets, etc.).
- A prominent section in `getting-started.md` titled "Composers
  and agents are the same thing" — explaining the `Composable`
  contract and showing the composer-in-composer pattern.
- All six runnable examples:
  - `examples/glow/01_hello.py` (already done in M2 — confirm)
  - `examples/glow/02_structured.py` (already done in M2 — confirm)
  - `examples/glow/03_streaming.py` (new — uses event selectors)
  - `examples/glow/04_composition.py` (already done in M4 — confirm)
  - `examples/glow/05_di_tools.py` (already done in M3 — confirm)
  - `examples/glow/06_mcp_typing.py` (already done in M4 — confirm)
- A line-count comparison script or note at
  `docs/internals/glow/comparison.md` documenting the side-by-side
  result.
- Top-level `README.md` of the repo: add a one-paragraph blurb
  pointing at `docs/glow/getting-started.md` (clearly marked
  experimental).

### Out

- Anything functional. M5 is documentation + examples + polish.

## Checklist

- [ ] **Commit 13: docs(glow): tutorial + cookbook**
  - [ ] `docs/glow/getting-started.md`:
        - [ ] Intro paragraph + 4-line hello-world.
        - [ ] Section "Structured outputs by return type".
        - [ ] Section "Per-call overrides with `.json()` / `.text()`".
        - [ ] Section "Streaming and event selectors".
        - [ ] Section **"Composers and agents are the same thing"** —
              walks through the `Composable` contract; shows
              composer-in-composer; reassures the reader that
              every layer is interchangeable.
        - [ ] Section "`Depends` — tools and sub-agents".
        - [ ] Section "Typed MCP".
        - [ ] Section "Guardrails as type metadata".
        - [ ] Section "Sets and merging" — explicitly explains
              the default `_default_merge` and when users need to
              pass `merge=` (the type-fragility risk from the
              plan).
        - [ ] Section "Race is deprecated — use parallel + reducer"
              cross-linking the M0 deprecation.
        - [ ] Footer linking to `limitations.md` and the
              `experimental/glow` branch.
  - [ ] `docs/glow/limitations.md`:
        - [ ] No streaming JSON.
        - [ ] No `AsyncIterator` tools yet.
        - [ ] No per-agent MCP subsets (one App = one composed
              client view).
        - [ ] Pydantic is a hard dep of the `[glow]` extra.
        - [ ] mypy strict mode supported; runtime type
              introspection only via `get_type_hints`.

- [ ] **Commit 14: chore(glow): finalize examples directory**
  - [ ] Confirm/refresh `01_hello.py`, `02_structured.py`,
        `04_composition.py`, `05_di_tools.py`, `06_mcp_typing.py`
        (mostly already in place from earlier milestones).
  - [ ] **Add `examples/glow/03_streaming.py`**:
        - [ ] Uses `@app.agent` with a streaming-capable tool.
        - [ ] Iterates via `agent("...").events(Thought,
              ToolCall.map(lambda tc: f"calling {tc.name}"))`.
        - [ ] Prints the live stream to stdout.
  - [ ] Each example prints a success banner at the end (matches
        the convention of `examples/02_agent_class/basic.py`).

- [ ] **Line-count comparison:**
  - [ ] Pick `examples/02_agent_class/basic.py` as the baseline.
  - [ ] Write `examples/glow/01_hello.py` as the equivalent in
        glow.
  - [ ] `wc -l` both. Document in
        `docs/internals/glow/comparison.md`:
        - Baseline LOC (excluding blank/comment lines)
        - Glow LOC (same exclusion)
        - Ratio. Target: glow ≤ 60% of baseline (i.e. ≥40%
          reduction).
  - [ ] If the target isn't met, note it as a deviation in the M5
        dev-log and either (a) tighten the example or (b) explain
        why the gap exists (e.g., glow surfaces features the
        baseline didn't use).

- [ ] **Top-level README update:**
  - [ ] One-paragraph blurb at the bottom (or in a dedicated
        section) flagging glow as experimental, pointing at
        `docs/glow/getting-started.md`.

- [ ] **Run validation (full pre-PR gate):**
  ```bash
  uv run ruff check src/ tests/
  uv run mypy src/

  docker compose up -d
  uv run pytest                            # full suite incl. integration
  for ex in examples/glow/*.py; do
      uv run python "$ex"
  done
  docker compose down
  ```

## Acceptance criteria

- All six examples in `examples/glow/` run cold against a live
  gateway and print success banners.
- The full pytest suite (including integration) is green.
- `docs/glow/getting-started.md` covers every public symbol glow
  exports.
- `docs/glow/limitations.md` enumerates the deferred items.
- `docs/internals/glow/comparison.md` documents the LOC reduction
  vs. baseline; the ≥40% target is hit or its deviation
  explained.
- The top-level `README.md` mentions glow as experimental and
  links to the tutorial.

## Verification commands

```bash
uv run ruff check src/ tests/
uv run mypy src/

docker compose up -d
uv run pytest                              # full suite
for ex in examples/glow/*.py; do uv run python "$ex"; done
docker compose down

# LOC comparison:
echo "Baseline:"; wc -l examples/02_agent_class/basic.py
echo "Glow:";     wc -l examples/glow/01_hello.py
```

## Cross-references

- Plan §"Verification" and §"End-to-end smoke" at
  `~/.claude/plans/la-idea-es-tener-glowing-gem.md`.
- Plan §"Anti-features" — `docs/glow/limitations.md` should mirror
  the deferred items called out as out-of-scope.
- M0 dev-log: cross-link the deprecation policy.

## Dev-log

`logs/milestone-5-dev-log.md` at end. This is the final dev-log;
it should include a brief retrospective: what went well, what was
harder than expected, what to consider for v1.1 (per-agent MCP
subsets, streaming JSON, `AsyncIterator` tools).
