# 01 — Implementation Plan

> Master roadmap for the `t0-agents` Agent abstraction. Describes the
> **phase ordering**, the **milestones** that stitch phases into
> user-visible shipments, and the **cross-cutting checklist** every
> phase applies.
>
> Read [`00-integration-contract.md`](00-integration-contract.md) first —
> the invariants there bound everything below.

**Related documents**

- [`00-integration-contract.md`](00-integration-contract.md) — stable
  architectural contract.
- [`02-open-questions.md`](02-open-questions.md) — open questions
  grouped by upstream lib.
- [`phases/`](phases/) — one detailed spec per phase.
- [`logs/`](logs/) — per-phase implementation log, written while
  executing.

---

## 1. Roadmap at a glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Problem # (from problemas.md) → Phase letter                             │
│                                                                           │
│      P1 → A     P3 → B     P6 → C     P5 → D     P2 → E                   │
│      P8 → F     P4 → G     P7 → H                                         │
└──────────────────────────────────────────────────────────────────────────┘

Phase:      A ──▶ B ──▶ C ──▶ D ──▶ E ──▶ F ──▶ G ──▶ H
            │     │     │     │     │     │     │     │
Milestone: M1    M2    M3    M4    M5    M6    M7    M8

Legend:
  A — Agent core
  B — Memory
  C — Guardrail abstractions
  D — Serving (non-streaming)
  E — Streaming
  F — TUI demo
  G — Multi-agent composition
  H — Agent testing framework
```

---

## 2. Phase table

| Phase | Name | Problem | Phase deps | Spec | Log |
| :---: | --- | :---: | --- | --- | --- |
| **A** | `Agent` core | P1 | — | [`phases/phase-a-agent-core.md`](phases/phase-a-agent-core.md) | `logs/phase-a-implementation-log.md` |
| **B** | Memory (repository) | P3 | A | [`phases/phase-b-memory.md`](phases/phase-b-memory.md) | `logs/phase-b-implementation-log.md` |
| **C** | Guardrails | P6 | A | [`phases/phase-c-guardrails.md`](phases/phase-c-guardrails.md) | `logs/phase-c-implementation-log.md` |
| **D** | Serving (HTTP) | P5 | A, B, C | [`phases/phase-d-serving.md`](phases/phase-d-serving.md) | `logs/phase-d-implementation-log.md` |
| **E** | Streaming | P2 | A, D | [`phases/phase-e-streaming.md`](phases/phase-e-streaming.md) | `logs/phase-e-implementation-log.md` |
| **F** | TUI demo (Typer + Rich) | P8 | A, C, E | [`phases/phase-f-tui.md`](phases/phase-f-tui.md) | `logs/phase-f-implementation-log.md` |
| **G** | Multi-agent composition | P4 | A–F | [`phases/phase-g-multi-agent.md`](phases/phase-g-multi-agent.md) | `logs/phase-g-implementation-log.md` |
| **H** | Agent testing framework | P7 | A–G | [`phases/phase-h-testing-framework.md`](phases/phase-h-testing-framework.md) | `logs/phase-h-implementation-log.md` |

All open questions answered (see [§4](#4-dependencies-between-phases)
below and [`02-open-questions.md`](02-open-questions.md)). Each phase's
"blockers" are now only its phase-deps from this column.

### 2.1 Why this ordering

Priorities came from the user's inline notes in `problemas.md`. The
rationale is restated here so phase authors don't have to reread the
original file:

- **A before anything.** P1 is the foundation. Every other phase
  extends `Agent`/`AgentConfig`.
- **B before C.** Memory changes what the LLM sees; guardrails decide
  what the LLM can do. Building context first keeps guardrails simple.
- **C before D.** The serving phase pipes elicitation through HTTP. If
  guardrails define the in-process contract first, the serving phase
  just exposes it over the wire.
- **D before E.** A non-streaming HTTP agent is immediately useful for
  request/response apps. Streaming is additive and non-trivial; shipping
  non-streaming first halves D's surface area.
- **F before G.** The TUI demo validates streaming + elicitation
  outside a browser. Multi-agent composition is a bigger design surface
  that benefits from a mature single-agent story.
- **H last.** Testing patterns solidify after they've been mocked in
  anger throughout A–G.

---

## 3. Milestones

Milestones are user-visible shipments. A milestone is "done" when its
target phase(s) are complete **and** it passes the cross-cutting
checklist (§5).

| # | Name | Ships after | Public artefact | Validation |
| --- | --- | :---: | --- | --- |
| **M1** | Declarative Agent | A | `from t0_agents import Agent, AgentConfig` works; all existing `run_agent` usage unchanged. | `examples/08_agent_class/` runs end-to-end against the live gateway. |
| **M2** | Stateful Agent | A + B | `Agent(config, memory=SqliteMemory("./m.db"))` — state persists across process restarts. | `examples/09_memory/` shows a two-session conversation recalling prior context. |
| **M3** | Trusted Agent | A + B + C | Declarative guardrails (`require_approval`, `max_cost`) compose with `Agent`. | `examples/10_guardrails/` shows an approval gate and a cost limit halting a run. |
| **M4** | Hosted Agent | A–D | `serve.create_app(agent)` returns a FastAPI app; `POST /chat` works. | `examples/11_http_agent/` serves an agent on `:8080`; `curl` exchange round-trips. |
| **M5** | Live Agent | A–E | `GET /chat/stream` (SSE) + WS elicitation channel. | `examples/12_streaming_http/` shows a browser client receiving deltas. |
| **M6** | TUI Agent | A–F | `t0-agents run <name>` launches a Rich TUI. | `examples/13_tui/` demonstrates streaming + elicitation in a terminal. |
| **M7** | Composed Agents | A–G | `Agent(config, sub_agents=[researcher, writer])`. | `examples/14_multi_agent/` orchestrator delegates across two sub-agents. |
| **M8** | Testable Agents | A–H | `from t0_agents.testing import MockInference, assert_tool_called`. | `examples/15_testing/` shows a green CI run with zero Docker dependency. |

### 3.1 "Ships" means

A milestone is a shipment, not an internal checkpoint. For each
milestone:

1. Example lands in `examples/` and runs end-to-end.
2. Public doc page lands in `docs/` describing the feature.
3. `CHANGELOG.md` entry with the extras needed to use it (e.g.
   `[memory-sqlite]` for M2).
4. An announcement snippet in the phase log, suitable for release notes.

---

## 4. Dependencies between phases

Restated as a DAG:

```
A ──┬──▶ B ──┐
    │        │
    └──▶ C ──┴──▶ D ──▶ E ──┐
                            │
                   ┌────────┤
                   │        │
                   │        └──▶ F ──▶ G ──▶ H
```

Independent work that can happen in parallel once A is done:

- B and C are independent of each other. They can be developed in
  parallel by two contributors if the team has the bandwidth, as long
  as both merge before D starts.
- H is the final synthesis and must wait for A–G.

Open-question status (resolved 2026-04-15 — see
[`02-open-questions.md`](02-open-questions.md)):

| Question | Phase(s) it informed | Resolution summary |
| --- | --- | --- |
| T0 Q1 | E | Per-block delta chunks with stable IDs → real-time `AgentEvent` emission feasible. |
| T0 Q2 | G | Per-call `additional_tools` is gateway-safe; LLM-side schema-change risk on us → ship stable list per run. |
| T0 Q3 | D | `namespace` and `tags` are orthogonal; namespace also auto-stored as `tensorzero::namespace` tag → ship both. |
| T0 Q4 | B | Both `system_template` and `prepended_message` (via `RawText`) supported → ship both behind `memory_strategy`. |
| T0 Q5 | C | Metrics must be pre-declared; framework does NOT auto-emit → document the manual `t0.feedback(...)` path. |
| T0 Q6 | B, D | No close API → memory upserts per-run; serving GCs sessions on disconnect + idle timeout. |
| T0 Q7 | H | `build_embedded` IS available → ship MockGateway **and** an embedded helper. |
| FastMCP Q1 | G | `add_tool(Tool.from_function(...))` is the runtime path. |
| FastMCP Q2 | D, E | No FastMCP abstraction → build `core/elicitation_forwarder.py`. |
| FastMCP Q3 | D | `call_tool` reentrant; elicitation handler is shared per Client → per-session Client. |
| FastMCP Q4 | B | Resources viable for RAG-flavoured memory → document as opt-in alternative. |
| FastMCP Q5 | F | Schema guarantees are stronger than draft-07 (flat object, primitives + enum + const) → renderer can be confident. |

**All phases are unblocked.** Each phase spec under [`phases/`](phases/)
restates only its phase-specific dependencies (Phase deps from §2 above).

---

## 5. Cross-cutting checklist

Applied to every phase before it merges to `main`. Lives verbatim in
each phase spec so the author doesn't have to context-switch.

### 5.1 Code

- [ ] New public API is type-annotated; `uv run mypy src/` passes in
      `strict` mode.
- [ ] `uv run ruff check src/` passes (no `--fix` needed at merge time).
- [ ] New module is re-exported from the appropriate `__init__.py`
      (`core/__init__.py` or `<submodule>/__init__.py`).
- [ ] No new dependency in base `pyproject.toml` — all third-party deps
      live under an extra.
- [ ] ImportError guard pattern used when a submodule depends on an
      extra:

      ```python
      try:
          import aiosqlite
      except ImportError as exc:
          raise T0AgentsError(
              "SqliteMemory requires the 'memory-sqlite' extra. "
              "Install with: pip install 't0-agents[memory-sqlite]'"
          ) from exc
      ```

### 5.2 Tests

- [ ] Offline unit tests under `tests/test_<submodule>/` cover the
      happy path and every error branch.
- [ ] At least one integration test marked
      `@pytest.mark.integration` exercises the feature against the
      live gateway + real FastMCP server.
- [ ] `uv run pytest -m "not integration"` is green.
- [ ] `docker compose up -d && uv run pytest -m integration` is green
      (run locally, not in CI unless T0 Q7 unblocks it).
- [ ] If the phase exposes an async API, multi-concurrent-call test
      included.

### 5.3 Examples

- [ ] One example under `examples/NN_<feature>/` demonstrating the
      phase's minimum-viable usage.
- [ ] Example boots against `docker compose up -d` with the
      repo's existing `config/tensorzero.toml` (extended if needed,
      never renamed).

### 5.4 Documentation

- [ ] One public page under `docs/<feature>.md` (or update an existing
      page) — not internal docs under `docs/internal/`.
- [ ] Update [`developer-guide.md`](developer-guide.md) §4 (Public API
      surface) if a new symbol is exported.
- [ ] Update [`developer-guide.md`](developer-guide.md) §12 (Customization
      knobs) if a new knob is added.
- [ ] `CHANGELOG.md` entry in the format used by earlier releases.

### 5.5 Implementation log

- [ ] Phase log written to
      `docs/internal/agentic-feature/logs/phase-<X>-implementation-log.md`
      during the phase, recording:
      - Divergences from the phase spec and why.
      - Answers received to open questions during the phase.
      - Any contract amendments (with a pointer to the amended section
        in `00-integration-contract.md`).
      - Release-notes snippet for the milestone.

### 5.6 Review ritual

- [ ] One reviewer confirms the contract is unchanged *or* that the
      amendment is documented.
- [ ] One reviewer confirms `mcp/` public API is unchanged.
- [ ] One reviewer runs the new example locally.

---

## 6. Definition of done for the whole roadmap

The Agent abstraction work is complete when:

- All eight milestones (M1–M8) have shipped.
- A new user can install `t0-agents[all]` and:
  - Import `Agent` and call `.run(...)`.
  - Attach memory.
  - Attach guardrails.
  - Serve it over HTTP / WebSocket with streaming.
  - Run it in a TUI.
  - Compose it with other agents.
  - Unit-test it without Docker.
- The `mcp/` submodule still works standalone without any of the above
  installed.
- The developer guide and public docs describe every new capability
  with at least one example reference.

---

## 7. How to work this plan

1. Pick a phase whose blockers are resolved (dependencies in §2, open
   questions in §4).
2. Read its spec in `phases/phase-<X>.md`.
3. Open its log at `logs/phase-<X>-implementation-log.md` — even if
   empty, this is where you record everything as you go.
4. Work the checklist from top to bottom.
5. Before merging, run the cross-cutting checklist (§5).
6. Ship the milestone if this phase closes one.

Parallelism note: when multiple phases are unblocked and staffed, record
the order-of-merge in each log to keep the history legible.
