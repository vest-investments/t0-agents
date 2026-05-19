# `compose-removal` — internal milestone tracker

This directory tracks the implementation of the `compose-removal`
workstream: removing the topology composer layer from t0-agents while
preserving (and relocating) the sub-agent orchestration code to
`src/t0_agents/delegation/`.

These docs are **internal**. They are the source of truth for *how* the
work is split into milestones, what each milestone owns, and how we
know it's done. There is no user-facing documentation for this
workstream — the workstream is a deletion, and the CHANGELOG
`### Removed` entry that lands in M2 is the only user-facing artifact.

## Where the design lives

The full design and rationale live outside the repo at:

```
~/.claude/plans/compose-removal-plan.md                       (canonical plan)
~/.claude/plans/quiero-quitar-compose-la-drifting-nygaard.md  (investigation report)
```

The milestone specs in `milestones/` are **executable instructions**
that assume the plan has been read. They focus on what to build (or
delete), in what order, and how to verify.

## Milestones

Execute in order. M2 depends on M1.

| # | Spec | Purpose |
|---|---|---|
| **M1** | [milestone-1-relocate.md](milestones/milestone-1-relocate.md) | Migrate internal callers off `compose/`. Move sub-agent code to `src/t0_agents/delegation/`. **No deletions yet.** |
| **M2** | [milestone-2-delete-and-examples.md](milestones/milestone-2-delete-and-examples.md) | Delete the topology layer + glow's `compose()` + add 6 canonical pattern examples (Sequential / Parallel / Orchestrator × Agent legacy + glow APIs). |

## Doc/log discipline

Always-write log policy: at the end of each milestone, write
`logs/milestone-<N>-dev-log.md` with five sections:

- **§1 Summary** — what landed; commit SHAs; file/test/doc inventory.
- **§2 Deviations** — where implementation diverged from the spec,
  and why.
- **§3 Open questions** — anything that surfaced but is parked.
- **§4 Carry-forward notes** — load-bearing handoff to the next
  milestone. Never skip this.
- **§5 Verification** — exact commands run and a summary of their
  output.

The `logs/` directory is currently empty; populate it as work
progresses. The dev-log template lives in the skill command at
`.claude/commands/compose-removal-milestone.md`.

## Branch

All work lives on `experimental/glow` (the branch glow's M0–M5
established; see the sibling tracker at `docs/internals/glow/`).
Per-milestone topic branches:

- `compose-removal/m1-relocate`
- `compose-removal/m2-delete-and-examples`

Each topic branch merges back to `experimental/glow` via PR.

**Do not merge `experimental/glow` to `main`** until the user
explicitly opts in after M2 completes. The decision to merge to main
is independent of this workstream's completion — the user owns it.

## Invariant

`src/t0_agents/mcp/` must stay byte-identical to `main` throughout
both milestones. Verify with `git diff src/t0_agents/mcp/` (must be
empty). This is the `feedback_mcp_standalone` invariant inherited from
the glow workstream.

## Why this workstream exists

The `compose/` submodule is a third API surface alongside legacy
`run_agent` and v1.0 `Agent`/`@app.agent`. It has its own protocol
(`Composable`), base class (`Composer`), error type (`ComposerError`),
and `Run` subtype (`CompositeRun`) — ~2,300 LOC + 13 public symbols
that mostly duplicate what raw `asyncio` + native `sub_agents` already
provide. Removing it simplifies the framework's mental model, kills
the single FastMCP hack in the codebase, and unblocks the 2 Groq
integration tests that fail because of `tool_choice="required"`.

The investigation that motivated the workstream is the canonical
reference for "why". See
`~/.claude/plans/quiero-quitar-compose-la-drifting-nygaard.md`.
