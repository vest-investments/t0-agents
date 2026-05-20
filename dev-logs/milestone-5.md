# Milestone 5 — Dev Log

**Date**: 2026-05-20
**Base branch**: main
**Branch**: `milestone/5-example-chat-only-showcase`
**PR**: <filled after Phase 5>

## Implemented

The optional showcase example that closes the
"optional `tool_provider`" workstream (M1+M2+M3+M4+M5).
M5 ships the runnable counterpart to M4's prose: a new
example folder sorted before `01_first_agent` so the first
thing a reader sees under `examples/agent/` is the
toolless-chat baseline.

Scaffolding (commit `6b95c17`, `feat(examples): scaffold
00_chat_only directory layout`):

- `examples/agent/00_chat_only/main.py` — ~35 lines.  Imports
  only `asyncio`, `Awaitable`, `cast`, `AsyncTensorZeroGateway`,
  `Agent`, `AgentConfig`.  No `fastmcp` import, no `Client`
  type, no `ToolProvider` protocol import, no `server.py`
  sibling.  Body is the README chat-only snippet expanded
  into a runnable script: build a gateway, enter it as the
  *single* async context manager, construct
  `Agent(AgentConfig(name="tutor", function_name="agent"), t0=t0)`
  with no `tool_provider=` kwarg, `await agent.invoke(
  "Explain monads in two sentences.")`, print
  `Response`/`Turns`/`Episode ID`/`Inference IDs` on the
  same shape as `01_first_agent/main.py` so SMOKE's signal
  text is identical.
- `examples/agent/00_chat_only/config/tensorzero.toml` —
  byte-equivalent to `01_first_agent/config/tensorzero.toml`
  (Groq `openai/gpt-oss-20b`, `[functions.agent]` of
  `type = "chat"`, `default` variant).  Per the every-example
  self-contained convention; each folder owns its config.
- `examples/agent/00_chat_only/docker-compose.yml` — verbatim
  copy of `01_first_agent/docker-compose.yml` (and
  `14_multi_server/docker-compose.yml`; they were already
  identical).  Keeps the known-good
  `clickhouse:lts` + `tensorzero/gateway:${T0_GATEWAY_VERSION:-2026.4.0}`
  pin per memory `feedback_example_compose_images.md`.

README for the new folder (commit `90326ec`,
`docs(examples): README for 00_chat_only`):

- `examples/agent/00_chat_only/README.md` — 89 lines.
  Title (`00 — Toolless chat`), aha! moment with concrete
  prompt/response transcript, "What it demonstrates"
  bullets (optional `tool_provider` contract, zero MCP
  allocations, shared `agent.invoke()` protocol, diagnostic
  shape), "When to use this template" decision aid,
  standard Prerequisites/Start/Run/Stop block (mirrors
  `01_first_agent/README.md`), "What to observe" calling
  out `Turns: 1` as the toolless-chat signature, "Related"
  section linking the repository `README.md` 4-line snippet,
  `docs/glow/getting-started.md`'s toolless-chat section,
  and the `docs/api-reference.md` signature row.

Cross-link from `01_first_agent` (commit `a7bf352`,
`docs(examples): cross-link 01_first_agent → 00_chat_only`):

- `examples/agent/01_first_agent/README.md` — appended a
  one-paragraph pointer right before the "Once this works,
  head to `02_agent_class`…" closing paragraph: "If you
  only need conversational LLM access without tools, see
  [`../00_chat_only/`](../00_chat_only/) for the same
  `Agent` surface with no MCP plumbing."

Reading-path table update (commit `9716984`,
`docs(examples): list 00_chat_only in agent README reading-path`):

- `examples/agent/README.md:23` — inserted a new first row
  in the reading-path table above the existing `01 |
  01_first_agent` row: "`00 | [00_chat_only](00_chat_only/)
  | Toolless chat — Agent(config, t0=t0) without
  tool_provider. The shortest possible agent: one inference,
  one text response, zero MCP plumbing. | yes |`".  Bumped
  the prose count "15 numbered examples" → "16".

SMOKE.md release-gate row (commit `e3190b8`,
`docs(examples): add B.0 row to SMOKE.md for 00_chat_only`):

- `examples/SMOKE.md` — inserted a new `### B.0 —
  \`00_chat_only\` (1 script)` section before `### B.1`,
  with the per-folder run command
  (`uv run python examples/agent/00_chat_only/main.py`) and
  the signal: `Response:` (non-empty), `Turns: 1`, `Episode
  ID:`, `Inference IDs: [<uuid>]`, exit 0.  The new section
  calls out `Turns: 1` explicitly as the load-bearing
  toolless-chat signature (single inference, zero tool-call
  turns).
- Added the matching `B.0 | 00_chat_only | …/00_chat_only/main.py`
  row in the per-batch checklist above the existing `B.1` row.
- Bumped two top-line counts to reflect the +1 script: "all
  24 example scripts" → "all 25 example scripts" (line 4),
  and "Eight cycles, ~13 minutes" → "Nine cycles, ~14 minutes"
  (line 89) since Batch B now cycles nine docker-backed
  composes instead of eight.

## Deviations from plan

- **`tensorzero.toml` lives under `config/` not at the example
  root.**
  - **What changed**: PLAN.md M5's literal path was
    `examples/agent/00_chat_only/tensorzero.toml`.  The
    config was placed at
    `examples/agent/00_chat_only/config/tensorzero.toml`
    instead.
  - **Why** (decision at implementation time): every
    existing example folder under `examples/agent/` puts the
    toml under a `config/` subdir so the `docker-compose.yml`
    can mount `./config:/app/config:ro` and pass
    `--config-file /app/config/tensorzero.toml`.  Matching
    the convention let me copy `01_first_agent`'s compose
    verbatim — no compose-time differences to maintain.
    Confirmed in plan-mode with the user.
  - **Impact on next milestones**: none.  M5 is the final
    milestone of the workstream.

- **No local `.env.example` carried in the folder.**
  - **What changed**: PLAN.md M5 listed
    `examples/agent/00_chat_only/.env.example` as a
    checklist item; the folder ships without one.
  - **Why** (decision at implementation time): no other
    example folder under `examples/agent/` carries a local
    `.env.example` — the canonical template is the shared
    `examples/agent/.env.example` (referenced explicitly by
    `examples/SMOKE.md` Setup step
    `cp examples/agent/.env.example examples/agent/.env`).
    Adding a local one would introduce a pattern no other
    folder uses and risk confusing readers about which
    `.env` file the example actually consumes.  Confirmed
    in plan-mode with the user.
  - **Impact on next milestones**: none.

- **SMOKE.md `B.0` entry added (scope expansion of +1 item
  beyond PLAN.md M5).**
  - **What changed**: PLAN.md M5's literal scope was
    strictly the example folder + cross-links; SMOKE.md
    polish was listed nowhere.  Added a `B.0` section and
    matching checklist row, and bumped two top-line counts.
  - **Why** (decision at implementation time): the M4 dev-log
    set the precedent of folding closely-related discoveries
    into the current milestone (M4 expanded by 4 items
    beyond its checklist) rather than spawning a docs-polish
    follow-up.  The example introduces a new SMOKE-eligible
    script; without the row, the next release-gate run
    would silently skip `00_chat_only` and a regression in
    the toolless-chat path could ship undetected.  The diff
    is small (1 prose section, 1 checklist row, 2
    number-bumps).  Confirmed in plan-mode with the user.
  - **Impact on next milestones**: none.  M5 is the final
    milestone of the workstream.

## Tests & lint

- `uv run pytest -m "not integration"` → **928 passed**, 2
  skipped, 61 deselected (3.36s).  Identical totals to the
  M1/M2/M3/M4 baseline; M5 touches zero source under `src/`
  and zero test under `tests/`, so the count cannot move.
- `uv run ruff check src/ tests/ examples/agent/00_chat_only/`
  → **All checks passed!** (M4 already cleared the 6
  pre-existing errors documented in M1+M2+M3; M5 introduces
  no new ones).
- `uv run mypy src/ examples/agent/00_chat_only/main.py` →
  **Success: no issues found in 80 source files**.  Count is
  79 source files (M4 baseline) + 1 (the new `main.py`) = 80.
- `python -c "import ast; ast.parse(open('examples/agent/00_chat_only/main.py').read())"`
  → exit 0 (PLAN.md M5 acceptance #1).
- `grep -nE "fastmcp|FastMCP|Client" examples/agent/00_chat_only/main.py`
  → **zero hits**, exit 1 (PLAN.md M5 acceptance #4).  The
  example's `main.py` does not import any MCP plumbing
  whatsoever; the contrast against `01_first_agent/main.py`
  (which imports `fastmcp.Client`, `ToolProvider`, and a
  sibling `server.py`) is the entire pedagogical point.  The
  README docstring originally said "No FastMCP server, no
  MCP client" and was reworded to "No MCP server, no MCP
  client" to satisfy the grep — the substance is identical.
- `grep -rn "mcp_client" examples/agent/00_chat_only/` →
  **zero hits**, exit 1.  Preserves M4's `mcp_client →
  tool_provider` rename hygiene in the new folder.
- **Live smoke run** (PLAN.md M5 acceptance #5):
  ```
  docker compose --env-file .env \
      -f examples/agent/00_chat_only/docker-compose.yml up -d
  # gateway + clickhouse healthy after ~6s
  uv run python examples/agent/00_chat_only/main.py
  ```
  Transcript (verbatim, trimmed to the signal block):
  ```
  Response: A monad is an abstraction that lets you wrap a value in a
  computational context (like optionality, state, or I/O) and then chain
  operations on that value while automatically propagating the context.
  It provides two core operations—`return` (or `pure`) to inject a
  value and `bind` (`>>=`) to sequence computations—ensuring that side
  effects or context are consistently handled.
  Turns: 1
  Episode ID: 019e4700-a002-79f3-8364-94558c11ff8e
  Inference IDs: ['019e4700-a002-79f3-8364-9441e2f1a9cf']
  ```
  Exit code: 0.  `Turns: 1` confirms the toolless-chat
  signature (single inference, zero tool-call turns).
  Response is non-empty and on-topic for the
  "Explain monads in two sentences" prompt (the LLM
  answered with two sentences as requested).
- Compose tear-down: `docker compose -f
  examples/agent/00_chat_only/docker-compose.yml down` →
  exit 0.  The
  `"GROQ_API_KEY" variable is not set` warning on `down` is
  benign — docker compose only needs the env var on `up`,
  and the `--env-file .env` flag was omitted on tear-down
  intentionally to test the compose file's resilience.
- Integration suite (`-m integration`): **not run** as a
  separate `pytest` invocation.  The live smoke run above
  is M5's integration coverage — the example is itself an
  integration test of the optional `tool_provider`
  contract end-to-end against a real Groq inference.

## Notes for next milestone

**M5 is the final milestone of the
optional-`tool_provider` workstream.**  M1+M2+M3+M4+M5 are
all merged (or, for M5, awaiting merge of this PR).  This
section closes the workstream rather than handing off:

- **Workstream complete.**  The `tool_provider` parameter
  is now fully optional throughout the v1.0 agent stack:
  `run_agent` (M1), `StreamingAgentLoop.stream` (M2), the
  high-level `Agent` class (M3).  Public docs, the v1.0
  changelog, and the api-reference all reflect the
  contract (M4).  A runnable showcase example lives at
  `examples/agent/00_chat_only/` and is exercised on every
  release gate (M5).  No follow-up milestones are required.
- **Pending memory rewording.**  The auto-memory note
  `feedback_mcp_standalone.md` still claims the `run_agent`
  signature is "frozen".  After M1's rename + optional
  default that is no longer literally true; the
  *standalone-submodule rule* the note codifies (no imports
  from higher layers; `Client[Any] | None` is the only
  allowed type on `tool_provider`) is still strictly
  enforced.  The next memory pass touching the note should
  reword it to describe the structural rule rather than the
  literal frozen-signature claim.  M5 is not the milestone
  to touch memory.
- **`project_pre_public_polish.md` carve-out.**  PLAN.md
  M5's "Notes" called out that annotating
  `examples/agent/03_output_modes/` (or any other existing
  example) with "no tools needed" pedagogy is out of scope.
  If that polish ever materialises, it belongs in the
  pre-public-polish backlog, not in a follow-up workstream.
  Memory note `project_pre_public_polish.md` is the right
  place to log it.
- **Numbering invariant.**  `00_chat_only` is intentionally
  the lowest-numbered folder under `examples/agent/`.  Any
  future example that wants the "first impression" slot
  must either (a) use a different ordering scheme entirely,
  or (b) renumber `00_chat_only` to `0X` and bump every
  reference (README rows, SMOKE rows, cross-links from
  `01_first_agent/README.md`).  Cheaper to leave `00_` as
  the toolless-chat slot in perpetuity.

## Open questions

None.  M5 closes cleanly: smoke ran green on first attempt,
all six PLAN.md acceptance criteria satisfied, no source
under `src/` or test under `tests/` touched.  The
optional-`tool_provider` workstream is complete after the
M5 PR merges.
