# `glow` — internal milestone tracker

This directory tracks the implementation of `t0_agents.glow`, the
experimental Pythonic high-level layer over t0-agents v1.0.

These docs are **internal**. They are the source of truth for *how*
the work is split into milestones, what each milestone owns, and how
we know it's done. User-facing documentation lives elsewhere
(`docs/glow/` after Milestone 5 ships).

## Where the design lives

The full design and rationale — comparative analysis, architectural
decisions, anti-features, risks — live in the plan file outside the
repo at:

```
~/.claude/plans/la-idea-es-tener-glowing-gem.md
```

The milestone specs in `milestones/` are **executable
instructions** that assume the plan has been read. They focus on
what to build, in what order, and how to verify.

## Milestones

Execute in order. Each milestone has its own spec with checklist
and acceptance criteria.

| # | Spec | Purpose |
|---|---|---|
| **M0** | [milestone-0-race-deprecation.md](milestones/milestone-0-race-deprecation.md) | Deprecate `race` in low-level; demote in docs. |
| **M1** | [milestone-1-foundations.md](milestones/milestone-1-foundations.md) | Scaffold (done) + `output_inference` + `FluentRun[T]`. No TZ gateway needed. |
| **M2** | [milestone-2-core-decorator.md](milestones/milestone-2-core-decorator.md) | `App` + `@app.agent` + event selectors. First milestone that needs TZ. |
| **M3** | [milestone-3-di-and-tools.md](milestones/milestone-3-di-and-tools.md) | `Depends` + `@app.tool` + sub-agents-via-`Depends(agent)`. |
| **M4** | [milestone-4-composition-mcp-guardrails.md](milestones/milestone-4-composition-mcp-guardrails.md) | `compose(spec)` native types + LLM router + typed MCP + `Annotated` guardrails. |
| **M5** | [milestone-5-polish.md](milestones/milestone-5-polish.md) | Tutorial + cookbook + 6 runnable examples. PR-ready. |

## Doc/log discipline

Following the always-write log policy from the structured-outputs
roadmap: at the end of each milestone, write
`logs/milestone-<N>-dev-log.md` with five sections:

- **§1 Summary** — what landed; commit SHAs.
- **§2 Deviations** — where the implementation diverged from the
  spec, and why.
- **§3 Open questions** — anything that surfaced but is parked.
- **§4 Carry-forward notes** — load-bearing handoff to the next
  milestone. Never skip this.
- **§5 Verification** — exact commands run and a summary of their
  output (paste the tail of the test output if it fits).

The `logs/` directory is currently empty; populate it as you go.

## Branch

All work lives on `experimental/glow`. The scaffold commit is
`768643d`. Until M5 lands and the user opts in, **do not merge to
main**.
