# Structured Outputs — Implementation Plan

> Roadmap mapping the five milestones to their dependencies and the
> cross-cutting checklist that every milestone applies before merge.
> Authoritative source for the `/structured-milestone <N>` command's
> phase map.

## §1 Goal

Ship JSON-mode / structured-output support in five small, independently
mergeable milestones.  Every milestone leaves `main` green for the
chat-function path (contract §2.4) and adds a coherent slice of the
JSON-function path.

## §2 Milestone table

| `$1` | Milestone | Spec file | Log file (always-written) | Deps |
|:----:|-----------|-----------|---------------------------|:----:|
| `1`  | JSON-inference foundations | `milestones/milestone-1-foundations.md` | `logs/milestone-1-dev-log.md` | — |
| `2`  | `Agent.run` JSON dispatch (non-streaming) | `milestones/milestone-2-agent-dispatch.md` | `logs/milestone-2-dev-log.md` | 1 |
| `3`  | Permissive parsing & retry-on-invalid-json | `milestones/milestone-3-permissive-parsing-and-retry.md` | `logs/milestone-3-dev-log.md` | 1, 2 |
| `4`  | Provider safety, streaming guard, quirks | `milestones/milestone-4-safety-and-quirks.md` | `logs/milestone-4-dev-log.md` | 2 |
| `5`  | Testing helpers, examples, polish | `milestones/milestone-5-testing-and-polish.md` | `logs/milestone-5-dev-log.md` | 1, 2, 3, 4 |

> Dependency arrows are **advisory**.  The user may merge out of order,
> but the implementer reads every prior dev-log before planning (see
> the `/structured-milestone` command).

## §3 Milestone summaries

Detailed specs live in the per-milestone files.  This section gives a
one-paragraph description of each so the reader can scan the
roadmap.

### M1 — JSON-inference foundations
Introduces `core/structured.py` with `run_structured_inference()`, the
`StructuredOutput` dataclass, and the three M1 `AgentConfig` fields
(`function_kind`, `output_schema`, `json_mode`) plus the two M1
`AgentResult` fields (`response_kind`, `parsed`).  Ships a single-shot
extractor example.  No changes to `Agent`, `Agent.run`, `Agent.stream`,
or the existing dispatch logic — the helper is callable directly.

### M2 — `Agent.run` JSON dispatch
Wires `Agent.run` to dispatch to `run_structured_inference` when
`config.function_kind == "json"`.  Adds the per-call kwargs
(`output_schema`, `json_mode`).  Defines the guardrail / memory /
sub-agent behaviour for the JSON path documented in contract §3.5
and §3.6.  Ships the "Structured Agent" example and completes
`docs/structured-outputs.md` for the non-retry happy path.

### M3 — Permissive parsing & retry-on-invalid-json
The value-add layer that justifies the framework's existence on top
of raw TensorZero.  Adds `permissive_parse()`, the
`permissive_json_parsing` and `max_json_retries` config dials, the
`parse_strategy` and `json_parse_attempts` result fields, and the
retry orchestration in `Agent.run`.  Ships a recovery-demo example
exercising all three layers (TZ → permissive → retry) and the
`docs/structured-outputs.md` "Robust JSON parsing" section.

### M4 — Provider safety, streaming guard, quirks
Adds the configuration-error surface for incompatible combinations
(`json_mode = "tool"` + tools, `Agent.stream` + JSON, `function_kind`
+ sub-agents) and the documentation for provider-specific quirks
(Anthropic extended-thinking, Bedrock/Vertex degradation).  Ships the
opt-in `require_parsed()` guardrail.

### M5 — Testing helpers, examples, polish
`MockGateway` extensions for JSON-function shapes, new assertion
helpers, the NER `extract_entities` end-to-end example with dynamic
schema override and `json_mode` swap, doc cross-links, and the
five `CHANGELOG.md` entries (one per milestone).

## §4 Dependency DAG

```
            ┌───────────────────────────┐
            │  M1 Foundations           │
            │  (core/structured.py,     │
            │   AgentConfig fields,     │
            │   AgentResult fields)     │
            └────────────┬──────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐         ┌────────────────────────────┐
│  M2 Dispatch     │         │  M4 Safety & quirks        │
│  (Agent.run wiring,│       │  (config errors, stream    │
│   per-call kwargs) │       │   guard, log warnings,     │
└────────┬─────────┘         │   require_parsed)          │
         │                   └──────────┬─────────────────┘
         ▼                              │
┌──────────────────────┐                │
│  M3 Permissive +     │                │
│  retry (parse        │                │
│  strategy, retry     │                │
│  loop, observability)│                │
└────────┬─────────────┘                │
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  M5 Testing + polish  │
        │  (mocks, asserts,     │
        │   example, docs,      │
        │   CHANGELOG)          │
        └───────────────────────┘
```

M2 and M4 can run in parallel after M1.  M3 needs M2 to land first so
the retry loop has a place to live.  M5 closes the roadmap.

## §5 Cross-cutting checklist

Every milestone applies every item below before merge.  The
`/structured-milestone` command's "Dual checklist" includes these
verbatim alongside the milestone-specific items in §8 of the spec.

### §5.1 Code

- New third-party dependencies: **none allowed**.  TensorZero, FastMCP,
  and `pydantic` (already pulled by `[serve]`) cover every need.  If a
  milestone genuinely needs a new dep, escalate to the contract.
- `mcp/` is untouched (contract §2.1).  Touching it is a contract
  amendment and must be recorded in the dev-log.
- Async functions use `async def` / `await`, not bare callbacks.
- mypy strict mode passes (`uv run mypy src/`).
- ruff passes (`uv run ruff check src/`); imports auto-sorted.
- Line length ≤ 100.

### §5.2 Tests

- Every new public symbol has at least one unit test.
- Every new branch in `Agent.run` / `Agent.stream` has at least one
  unit test exercising both sides.
- `uv run pytest -m "not integration"` is green on a fresh clone with
  no docker running.
- Integration tests are marked `@pytest.mark.integration` and run
  green when `docker compose up -d` is up with `GROQ_API_KEY`
  configured.  They are not required for CI default, but the
  milestone's acceptance criteria call them out explicitly.
- The shape-contract test
  (`tests/test_testing/test_mock_shape.py`) keeps passing.  When
  `MockGateway` learns new shapes (M5), the contract test grows.

### §5.3 Documentation

- `docs/structured-outputs.md` ships its slice each milestone (M1 = stub
  + matrix; M2 = Agent section; M3 = robust-parsing section; M4 = quirks
  + safety; M5 = cross-links).
- `docs/agent.md` is updated when a new `AgentConfig` field or
  `Agent.run` kwarg lands.
- `docs/concepts.md` gets a one-paragraph "Chat vs JSON functions"
  primer at M1.

### §5.4 Examples

- M1 ships `examples/17_structured/extract_customer_info.py`
  (single-shot helper).
- M2 ships `examples/17_structured/structured_agent.py` (Agent
  dispatch).
- M3 ships `examples/17_structured/recovery_demo.py` (3 layers in
  action).
- M5 ships `examples/17_structured/ner_dynamic_schema.py` (per-call
  schema override and `json_mode` swap).
- Each example folder carries its own `README.md` and tensorzero.toml
  snippet.

### §5.5 Implementation log (always-written)

`logs/milestone-<N>-dev-log.md` is written **at the end** of every
milestone, regardless of whether the implementation diverged from the
spec.  The template lives in §3 of the `/structured-milestone` command
(§1 Summary · §2 Deviations · §3 Open questions · §4 Carry-forward ·
§5 Verification).

This is the **only** difference from the agentic-feature roadmap's
log policy (which was divergence-only).  The reason: every milestone's
implementation surfaces small decisions that the next milestone needs
to know about — there is no "silent clean phase".

### §5.6 CHANGELOG entry

Every milestone appends an `### Added` / `### Changed` block to a new
top-level "## [0.10.0-M<N>] — Unreleased" section under the existing
v0.9.0 entry.  Final version-bump consolidation happens at M5 (collapse
all M1–M5 sub-sections into a single `## [0.10.0] — Unreleased`).

### §5.7 Review ritual

Before the milestone is declared done:

1. Run `uv run ruff check src/`.
2. Run `uv run mypy src/`.
3. Run `uv run pytest -m "not integration"`.
4. If applicable, run `uv run pytest -m integration` (after
   `docker compose up -d`) — note the result in the dev-log §5.
5. Re-read the touched docs in a Markdown previewer to catch broken
   internal links.
6. Read the prior dev-log(s) one more time to ensure no carry-forward
   item from §4 was missed.

## §6 Estimated surface

Rough size estimate for planning purposes only — not an acceptance
gate.

| Milestone | New files | Modified files | Tests added | Docs added |
|:---------:|:---------:|:--------------:|:-----------:|:----------:|
| M1 | 2 (structured.py, example dir) | 3 (config.py, types.py, __init__.py) | ~8 unit, 2 integration | 1 (stub of structured-outputs.md, concepts.md primer) |
| M2 | 1 (example file) | 1 (agent.py) | ~6 unit, 2 integration | 1 (Agent section in structured-outputs.md) |
| M3 | 1 (example file) | 2 (structured.py, agent.py, config.py) | ~10 unit, 1 integration, 1 property-based | 1 (Robust JSON parsing section) |
| M4 | 0 | 2 (agent.py, guardrails/builtins.py) | ~5 unit | 1 (Safety + quirks section) |
| M5 | 1 (example file) | 4 (testing/*, mock_shape test, every docs/*.md cross-link) | ~6 unit | cross-links across `docs/` |

Total: ~5 new files, ~12 modified files (some touched by multiple
milestones), ~30+ new tests, ~5 docs sections.
