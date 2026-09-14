# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.9.0] — 2026-09-13

`load_transcript` rebuilt what a person was shown and nothing of what
it was derived from: a round whose input ended on tool results
contributed only its reply, so a UI that turned a result into
something on screen — a citation row under a market-analysis answer —
had no way to draw it again on reload. The results were in the store
the whole time, as the `user` message the gateway appends before the
next inference.

### Changed

- **`TensorZeroMemory.load_transcript` emits a round's tool results.**
  Each round contributes the user message its input ends on — the text
  that opened the turn, or the previous round's results as the gateway
  stored them — followed by its `output`, so a turn reads as its
  opening text and then each round's results and reply in order. A
  two-round turn now yields `user, assistant, user, assistant`, and a
  parallel batch stays one `user` message with its blocks in stored
  order. A consumer that shows only text skips the results as it skips
  any non-text block. `load_history` is untouched.

## [2.8.1] — 2026-09-12

In 2.8.0, `StructuredOutput` does not satisfy `OutputMode` under mypy.
The protocol declares `name` as a settable variable, and
`StructuredOutput` — the only built-in mode whose name is derived from
its model — implements it as a read-only property, which does not
satisfy a settable member. So every public surface taking an
`OutputMode` rejects the one mode that produces typed output,
including the snippet the documentation publishes as the canonical
path. Two `cast()` calls inside `src/` keep `mypy src/` green, which
is why the failure is visible only to callers.

### Fixed

- **`OutputMode.name` is now declared as a read-only property.** The
  relation holds one way only: a member declared that way is satisfied
  by a plain attribute, so the four attribute-backed modes are
  untouched while `StructuredOutput` conforms.
  `AgentConfig(output=...)`, `Agent.invoke(output=...)` and
  `FluentRun.as_(...)` accept it, and `as_()` infers `FluentRun[M]`
  with no annotation on the target — the mismatch used to collapse
  inference to `OutputMode[Never]`, which reads as a variance problem
  and is not one.

### Notes

- Types only. `isinstance(mode, OutputMode)` still holds for every
  built-in mode and `name` returns the same strings: a
  `@runtime_checkable` protocol reduces a data member to `hasattr`,
  which a property and an attribute satisfy alike.
- The two internal casts that hid the defect are gone, so the concrete
  pairing now happens inside `src/`. `mypy src/` — the pre-PR gate in
  `CONTRIBUTING.md` — fails if the declaration regresses.

## [2.8.0] — 2026-09-12

A clean install of 2.7.0 does not import. The package declares
`fastmcp >= 3.2.3` with no ceiling and never declares `mcp` at all,
so a fresh resolution takes fastmcp 4.x, which requires mcp 2.x,
which renamed `McpError` to `MCPError` — and `import t0_agents`
raises `ImportError` before anything runs. The bundled `uv.lock`
pins fastmcp 3.2.4 and mcp 1.27.0, so every workflow that
`CONTRIBUTING.md` documents masks it; only someone resolving from
`pyproject.toml` ever sees it. This release bounds both ranges and
adds the check that would have caught it.

### Fixed

- **`mcp` is now a declared dependency, bounded to `>=1.24.0,<2`.**
  The package imports `McpError`, `mcp.types`, `RequestContext` and
  `mcp.shared._httpx_utils` across eight modules but left fastmcp to
  supply them, so it had no way to state the constraint its own code
  imposes. Supporting mcp 2.x is a port, not a rename: the wire types
  move to a separate `mcp-types` distribution, `MCPError` ships no
  back-compat alias and takes a different constructor, and
  `mcp.shared.context` drops `RequestContext` for `BaseContext`.

- **`fastmcp` is bounded to `>=3.2.3,<4`.** 4.x moves
  `fastmcp.client.tasks` into a separate distribution behind an
  extra, so `TaskNotificationHandler` is missing from a default 4.x
  install for reasons unrelated to the mcp rename. Two independent
  breaks, one ceiling.

- **`t0-agents info` accepts `--agent`.** On an App registering more
  than one agent it failed with "pick one with --agent" — naming an
  option only `run` and `serve` declared, so following the advice
  produced "No such option: --agent". All three subcommands now take
  the flag, reach the shared loader the same way, and describe glow
  App specs identically in `--help`.

### Added

- **A lockfile-free import check in `CONTRIBUTING.md`.** It resolves
  from `pyproject.toml` rather than `uv.lock`, which is what an
  installing consumer does and the only way this class of defect is
  visible from inside the repo.

- **The DeepSeek row the structured-output guide was missing.** Its
  Chat Completions endpoint accepts only `text` and `json_object`,
  so `json_mode = "strict"` warns and downgrades — an endpoint limit
  rather than a provider quirk. `json_object` also requires the word
  "json" and an example in the prompt; without them the model can
  emit unbounded whitespace until `max_tokens`, which is a hang
  rather than a parse failure and so never reaches the recovery
  ladder.

### Notes

- The bounded range resolves to fastmcp 3.4.7 and mcp 1.30.0, one
  minor above the pair in `uv.lock`. The suite runs green on both.

## [2.7.0] — 2026-09-10

Every inference reports its usage and the library adds it up, but the
sum lives only as long as the `Run` that produced it: nothing wrote it
anywhere, and the SDK has no way to read it back afterwards. A
consumer that meters or bills its users had to scrape ClickHouse.
`UsageLedger` is the seam that was missing — a Protocol the consumer
implements over its own store, fed by a middleware that captures each
inference as the gateway reports it and hands over one record per run
when the run ends, whichever way it ends. An agent that does not set
it behaves exactly as it did in 2.6.0.

### Added

- **`UsageLedger`, the persistence seam for usage.** A
  `runtime_checkable` Protocol with one method,
  `record(run: RunUsageRecord)`, called once after every run. The
  record carries one `InferenceUsage` per inference — the gateway's
  `inference_id` and `episode_id`, the variant, the per-inference
  `Usage`, the names of the tool calls it made, its `finish_reason` —
  plus the run's totals, turn count, outcome (`completed`,
  `cancelled` or `failed`), timings and tags. Tags pass through as the
  gateway saw them; the library reads nothing into them.

- **`usage_ledger` on `AgentConfig`, `App(...)` and `@app.agent`.**
  Setting it appends `UsageLedgerMiddleware` to the default chain
  right after `cost_tracking`. The write is best-effort by contract:
  bounded by a timeout (2 s by default), and a ledger that raises or
  hangs is logged and never changes the run's result or raises to
  the caller. A cancelled run records `cancelled` and still cancels.

- **`run_structured_inference(..., usage_ledger=)`** records the
  single inference of a direct JSON-function call the same way.

- **`InMemoryUsageLedger` and `SqliteUsageLedger`** under
  `t0_agents.usage` — the reference backends, the latter behind the
  existing `memory-sqlite` extra. `SqliteUsageLedger` refuses
  `":memory:"` and `""`: it connects per record, so a
  connection-scoped database would lose every row.

### Notes

- Inference ids are captured from the gateway's own responses, never
  from `AgentResult.inference_ids`, so the corrective-retry sentinels
  that list carries never reach a ledger. A stream that ends without
  a usage chunk yields no inference row; its run row carries none.
- A sub-agent's inferences are recorded by the sub-agent's own chain
  when it has a ledger; the parent's record covers the parent's own
  inferences, so rows are never counted twice.

## [2.6.0] — 2026-09-05

A stored episode has two readers that want opposite things from one
block. The model must not be shown a tool call that no result ever
answered — a provider rejects it — so `load_history` drops the call a
turn ended on. A person who watched that call render something needs
it back, and once the conversation continues the one-row read can
never reach it again: the stripped reply is what got persisted
forward. `load_transcript` is the second reader, beside the first
rather than a flag on it.

### Added

- **`TensorZeroMemory.load_transcript(episode_id, *, window=200)`.**
  Reads every chat round of an episode (the `window` most recent,
  oldest first) and emits the user message that opened each turn
  followed by one assistant message per round, built from the round's
  own stored `output` with every tool call kept. Tool results never
  appear, `Thought` blocks are dropped as `load_history` drops them,
  and an empty episode reads as `None`. Rows are ordered by
  `inference_id` — a UUIDv7 — because the server orders by
  `timestamp` at second granularity and two rounds of one turn
  regularly share a second.

### Changed

- The "skipped unsupported content-block kind(s)" warning names the
  reader that emitted it and no longer says the model will see the
  rest of the conversation — the transcript reader's consumer is not
  the model.

### Fixed

- `docs/memory.md` listed `Thought` among the blocks `load_history`
  converts and said tool calls from prior turns are preserved. It
  drops the first, and it drops a call the final round left
  unanswered. The page now says so and describes both readers.

## [2.5.0] — 2026-09-05

Reasoning is not free, and some providers expect the reasoning a model
produced earlier in a turn to come back with the next request while
that turn still carries tools. The loop stripped it unconditionally,
so every round after the first paid to derive its reasoning again —
and on a provider that enforces the expectation rather than tolerating
its absence, the request fails outright. `replay_thoughts` keeps a
round's reasoning in the transcript the next round of the same turn is
built from. It is off by default, and it stops at the turn boundary on
purpose: replaying every past turn's reasoning would grow a long
conversation's prompt without bound.

### Added

- **`replay_thoughts` on `AgentConfig`, and `replay_thoughts=` on
  `@app.agent`.** When `True`, each assistant round keeps its
  `Thought` blocks in the replay transcript instead of having them
  stripped before the next round's request. The default `False` leaves
  an agent behaving exactly as it did in 2.4.0, so this is opt-in per
  agent and costs nothing to ignore.

- The strip-or-keep decision lives in `complete_assistant_turn`, the
  one post-inference boundary both loops already share, so the
  streaming and non-streaming paths cannot come to disagree about what
  a turn replays.

- **The scope is a single turn, deliberately.** `load_history` and the
  stored-output conversion strip reasoning unconditionally and ignore
  the flag, so nothing from an earlier turn is ever replayed. Were a
  provider to enforce the same expectation across turns, the bounded
  answer is the previous turn's reasoning rather than all of it.

### Fixed

- **The streaming loop rebuilt an assistant turn in its own block
  order** — every `Text`, then every `Thought` — while the
  non-streaming path receives the provider's own order, reasoning
  first. Both loops now describe a turn the same way. Nothing could
  observe the difference while thoughts were stripped on the way to
  the replay transcript, and replaying them is what puts the list on
  the wire.

- **A rebuilt `Thought` lost its `signature` and `provider_type`.**
  The stream assembler accumulated reasoning as text alone, so a
  provider that verifies its own reasoning on replay would have been
  handed a block it has to reject, while the non-streaming path
  carried both fields intact. They now survive the rebuild, and are
  picked up from whichever chunk of the block carries them.

## [2.4.0] — 2026-09-05

Some tool calls are the reply rather than a step toward it — a render,
a hand-back to the caller, anything whose payload is what the person
asked for. The loop had no way to know that: it fed every result back
and spent another inference writing a closing line over the top of
work that was already done. `terminal_tools` names those tools, and a
round calling only them ends the turn where it stands. The release
also closes a hole in the older `tools=` filter that could widen it to
the entire tool surface, which is the part to read if you upgrade for
nothing else.

### Added

- **`terminal_tools` on `AgentConfig`, and `terminal_tools=` on
  `@app.agent`.** Entries are literal tool names or `fnmatch` globs,
  the same string grammar `tools=` accepts. When every tool call in a
  round matches one, no result is an error, and the round produced
  text of its own, the loop executes the tools and returns instead of
  running another inference. The round's text is the answer. The
  default `()` never ends a round early, so an agent that does not
  declare it behaves exactly as it did in 2.3.0.

- The rule lives in `core.turn_engine.is_terminal_round` and both the
  streaming and non-streaming loops consult it at the same point, so
  the two cannot drift. `DoneEvent` remains the only turn terminator
  and the event order is unchanged: tool calls, then results, then
  `DoneEvent`, with no synthesised text block.

- **Three rounds deliberately keep going**, and they are the contract
  worth knowing. A round with no text of its own has said nothing yet,
  and ending it would leave the run blank; blank counts as none, by
  the same `strip()` rule the wire's never-blank belt uses, so a model
  that emits only newlines before its tool call does not qualify. A
  round where any result came back an error keeps going so the model
  can see it and fix the call. And a round that batches a terminal
  tool with a data lookup keeps going, because the model asked for
  that data — with `parallel_tools=True` this is a routine shape.

- A `terminal_tools` pattern matching no live tool warns at
  `build()`, the same dead-pattern diagnostic `tools=` emits. The two
  now share one warning, so their wording cannot drift apart.

### Fixed

- **A bare string passed to `tools=` widened the filter instead of
  narrowing it.** `str` is itself a sequence, so `tools="lookup_*"`
  satisfied the annotation at runtime and then `list()` split it into
  its characters. One of them is `*`, and `*` matches every live tool,
  so a filter written to restrict an agent returned the entire
  surface. Because `tools=` is the control that keeps a sensitive tool
  away from an agent whose server stays mounted for others, the
  failure mattered more than the typo that caused it: the only signal
  was a scatter of dead-pattern warnings, one per character, which
  reads like noise rather than a breach. Both `@app.agent(tools=)` and
  `conversation(tools=)` now refuse a bare string where it is written,
  before anything is split. Typed callers were never exposed — every
  declaration site already annotates `list[Any] | None` — so this
  closes the runtime hole beneath the annotation.

### Changed

- **The declared `tensorzero` floor is `>= 2026.6.0`.** The 2.3.0
  notes described this floor and the reasoning behind it, but the
  package still declared `>= 2026.4.0`; the declaration now matches
  the documentation.

### Notes

- `run_agent` dispatches through the same detailed executor the
  streaming loop already used. Behaviour is unchanged — the public
  `execute_tool_calls` and `execute_tool_calls_parallel` are thin
  projections over it and are untouched — but the loop now reads each
  call's typed error flag rather than inspecting the result text for
  an `ERROR:` prefix, which is what makes the terminal rule
  trustworthy.

- A turn ended by its tools stores an assistant message whose tool
  calls have no results, because there is no following inference to
  carry them. That is the same shape `TensorZeroMemory.load_history`
  has dropped since 2.2.3, so the two compose: the reply survives
  replay, the unanswered calls do not, and the next turn runs over the
  reconstructed history. Pinned by a test rather than left to be
  rediscovered.

## [2.3.0] — 2026-08-27

This release moves the library's TensorZero floor from 2026.4.0 to
2026.6.0, the final upstream release before the project was archived,
and it does so with evidence rather than a version edit. Every
gateway behaviour the library leans on was re-run against a real
2026.6.0 gateway before the floor moved: the `list_inferences` reads
behind `TensorZeroMemory`, client-minted UUIDv7 episode ids, the
`finish_reason` enum in ClickHouse, per-provider `extra_body`,
`reasoning_content` round-trips, and `json_mode` degradation. None
of them changed. What did change is the default the gateway writes
with, and that is the reason to read this entry.

### Changed

- **`tensorzero >= 2026.6.0` is the new floor.** The Python client
  surface the library uses is identical between 2026.4.0 and
  2026.6.0 (the only additions are an optional `gateway_http_headers`
  keyword on `inference()` and typing refinements), so no code
  changed. The floor moves because the gateway side did: 2026.5.1
  treats SSE body decoding errors as fatal instead of retrying the
  decode in a loop, and 2026.6.0 closes GHSA-824w-x939-6cmc. Both
  matter to anyone running the gateway the library talks to. The
  upstream repository is archived, so the floor is left open rather
  than capped: no newer upstream release can appear, and a fork that
  publishes under the same name is expected to run this suite.

- **The bundled `docker-compose.yml` and every example pin
  `tensorzero/gateway:2026.6.0`**, and the development gateway
  config sets `observability.async_writes = false` explicitly.
  Gateway 2026.4.1 flipped that default to `true`, which is a
  latency win for a plain proxy and a correctness bug for anyone
  using `TensorZeroMemory`: the next turn's `load_history` can read
  before the previous turn's row exists and start the conversation
  over. Measured in this release; see the Notes.

### Added

- **Two integration tests pin the client-minted episode id
  contract**: `inference(episode_id=str(uuid7()))` is stored under
  that id and found by `list_inferences`, and a UUID that is not v7
  is rejected with HTTP 400. Every conversation glow opens depends on
  the first; nothing in the suite had exercised either against a
  real gateway before.

### Notes

- **`async_writes = true` measured, not described.** With the
  2026.6.0 default, 34 of 50 `load_history` reads issued right
  after the previous inference returned came back empty; the row
  became visible after p50 80 ms / p90 121 ms / max 153 ms. With
  `async_writes = false`, 0 of 50. Set it under
  `[gateway.observability]` wherever episodes span turns;
  `docs/memory.md` § Read-after-write timing carries the numbers.

- **The `weight` field on variants still boots as a deprecation
  warning on 2026.6.0, not an error**, and the legacy static-weights
  path is still compiled in. Migrate to
  `[functions.<name>.experimentation]` at leisure, not under
  pressure.

## [2.2.8] — 2026-08-27

This release is about one question: what should a turn do when it
cannot read the conversation it is supposed to continue? Until now,
answer anyway. `memory_strategy="messages"` exists to thread the prior
conversation through `input.messages` — `AgentConfig` refuses to build
without a provider that can supply it — yet a failed `load_history`
logged a warning and continued, replying as though the conversation
had never happened. That fired three times in seven seconds against a
dead gateway on 2026-08-25; had any of those attempts connected,
someone would have received that reply. A caller can retry an error.
Nobody can un-see an amnesiac answer.

### Changed

- **A failed `load_history` under `memory_strategy="messages"` now
  ends the turn** with the new `MemoryHistoryUnavailableError`, raised
  before any inference is issued. The failure and the empty episode
  had collapsed onto one value; they are now distinct — `load_history`
  returning `None` or `[]` still means "no inferences yet" and a fresh
  session still starts clean. A backend raising its own
  `T0AgentsFatalError` propagates unwrapped, keeping its class as the
  discriminator. `load` and `save` are unchanged and stay best-effort:
  they inform a prompt, they do not constitute it.

  The class name is the whole consumer contract, and it needs no wire
  change — it arrives as a typed `except`, as
  `RunErrorEvent.error_type`, and as `detail["exc_class"]` on the
  terminal `ErrorFrame`, so a re-drive policy can tell "the store
  could not be read" from "the model call failed".

  No opt-out flag ships. A caller who wants the old behaviour wraps
  the provider and returns `None` on failure, which states in code
  that an unreadable history is to be treated as a new conversation.

### Fixed

- **Caller-explicit `prior_messages` now skip the history read rather
  than discarding its result afterwards.** `MemoryMiddleware` has
  always documented that `invoke(prior_messages=…)` and `Run.fork`
  "take precedence and skip the load entirely"; they took precedence,
  but the read ran regardless and what it returned was dropped.
  Harmless while that read was best-effort — not once the change above
  makes it fatal, since a fork holding the entire transcript would
  then abort on a store it needs nothing from. Threaded down as
  `caller_supplied_history` so the skip happens before the I/O, and so
  the strategy dispatch stays in one place.
- **A `T0AgentsFatalError` raised by a memory backend now actually
  propagates.** `docs/memory.md` has documented that escape hatch —
  "the same contract guardrails use for hard aborts" — since the
  strategy shipped, but the broad `except Exception` around `load` and
  around `save` caught the very class the doc promised would escape,
  so wrapping a backend to fail hard did nothing. Both belts re-raise
  it first now, the idiom `mcp/loop.py`, `guardrails/runtime.py` and
  `delegation/provider.py` already use.

- **A `messages`-strategy provider without `load_history` reaching the
  read is reported as the configuration error it is.**
  `AgentConfig.__post_init__` checks for it, but neither override seam
  repeats the check — `invoke(memory=…)` validates only the
  MemoryProvider protocol, and glow's `conversation(memory=…)`
  validates only inside `hydrate`. Such an override arrived as an
  `AttributeError` that the belt above would have reported as a failed
  read, pointing the operator at the gateway for a fault in their own
  call.

### Notes

- **One amnesia shape is out of reach of this error, and it is a
  gateway setting.** A read that *succeeds* against a lagging
  observability store returns an empty history that is by construction
  indistinguishable from a new episode: nothing raises, and the turn
  proceeds with no prior context. Gateway **2026.4.1** flipped
  `observability.async_writes` to `true` by default, widening that
  window from "heavy local load" to "every turn". If your episodes
  span turns, set `async_writes = false` under
  `[gateway.observability]`. See `docs/memory.md` § Read-after-write
  timing.

- Behaviour change for any consumer that relied on the history read
  being best-effort. In practice a turn that used to answer badly now
  fails visibly, which is the point — but it moves a failure count
  rather than a success count.

## [2.2.7] — 2026-08-24

This release is about what a consumer sees at the end of a run. The
public wire makes two promises — `error` is terminal, and every
successful run ends in visible text — and both had holes. A
recoverable artefact (a malformed tool payload, an unknown chunk, a
corrective retry) crossed `project()` as a bare `ErrorFrame` and
rendered mid-run, so a client that seals its view on `error`
truncated an answer that was still streaming. The dominant shape of
that malformed payload turned out to be recoverable in the first
place: a complete argument object followed by a stray closing brace.
And the never-blank belt, which keeps a thought-only run from
arriving empty, could be defeated from either side — by publishing
raw reasoning as the answer on providers that stream it, or by
staying disarmed across a run boundary it failed to recognise.

### Added

- **`with_text_fallback(thought_fallback=…)` lets the consumer choose
  what a thought-sourced answer is worth.** Two provider classes
  reach the stage under the same `response_source == "thought"`:
  summary channels put a publishable summary in that slot, while
  providers that stream unsummarised chain-of-thought put private
  reasoning there. `"emit"` (the default, unchanged) synthesises it
  as the answer; `"suppress"` substitutes the never-blank belt text
  so a monologue is never published as assistant text. The stage
  does not sniff the provider — the consumer knows its variant and
  states the policy. Neither policy touches the `DoneFrame`, so a
  consumer reading `response_source` can still recognise a
  thought-only run and re-drive it.

### Fixed

- **Recoverable stream artefacts no longer render as a public
  `error` event.** `project()` now stamps such frames
  `detail["recoverable"] = True` and `public_allowlist` drops them;
  only failures that end the run (`exc_class`-bearing, or
  consumer-synthesised with an empty `detail`) reach the public
  wire. `with_status` no longer treats a recoverable frame as a run
  boundary, so a tool call keeps the "Thinking…" hint suppressed
  across a mid-run retry. The public golden moved with the
  contract: a recoverable `ErrorEvent` is now a dropped fixture.
- **Tool-call arguments that form a complete object followed only by
  closing braces, brackets or whitespace are salvaged.** One bounded
  rule, applied by both the streaming assembler and the shared
  dispatch executor, so the call executes with its full arguments
  and the wire's `arguments_json` matches the runtime's
  `raw_arguments` exactly. Syntactically valid JSON that the gateway
  zeroed for schema reasons is still refused — the rule recovers
  from trailing garbage, never from a schema verdict — and anything
  else still error-feeds the model without touching the public wire.
- **A bare terminal `ErrorFrame` re-arms the never-blank belt.**
  `with_text_fallback` recognised a run boundary by
  `detail["exc_class"]`, which a consumer-synthesised terminal frame
  does not carry. A run that streamed text and then died on such a
  frame left `saw_text` set, so the next run on the same long-lived
  stream got no fallback at all and a thought-only completion
  arrived blank. The stage now uses the discriminator `with_status`
  already uses, `not detail.get("recoverable")`.

### Notes

- The `thought_fallback` default is `"emit"`: no behaviour change for
  existing consumers.
- The salvage rule is deliberately narrow. `permissive_parse` mines
  JSON out of prose and would accept two concatenated objects; tool
  arguments get the strict rule instead.

## [2.2.6] — 2026-08-21

A small surface release. Stopping an in-flight run was reachable only
by dropping down to the underlying `Run` — and the two places that
documented that detour contradicted each other and the code. The
escape hatch said the fluent handle has no `cancel()`; the
`RunCancelledError` docstring said an `events()` iterator never
raises on cancellation. Both were wrong, the second in the worst
possible place. A missing `cancel()` also invites the one mistake
that matters here: cancelling the task that *consumes* a run leaves
the drive running unobserved, tools in flight included.

### Added

- **`FluentRun.cancel()`.** The fluent handle reaches `Run.cancel()`
  directly, joining `cost()` / `usage()` / `feedback()` / `fork()`.
  Awaiting it stops the *drive*, not merely the consumption: the
  drive task is cancelled and awaited, so on return the run is
  settled at `"cancelled"`, no further tool call can start, and a
  tool blocked mid-dispatch has already observed the cancellation at
  its own await point. The run is reached through the shared lineage,
  so any handle in a fluent chain cancels the one run the family
  stands for. Idempotent.
- **It is a no-op before consumption.** Unlike the other run-level
  delegations, which raise until the run exists, cancelling an
  unconsumed handle succeeds and does nothing — the same answer
  `Run.cancel()` gives for an already-settled run. It does not arm a
  pending cancellation, so a caller racing a cancel against the first
  consumption needs its own "cancel requested" flag; the method
  cannot pre-empt a run that was never built.

### Fixed

- **`RunCancelledError`'s docstring described the opposite
  behaviour.** It claimed `run.events()` consumers see the events up
  to the cancellation point "followed by an end-of-stream — the
  iterator does not raise". The iterator *does* raise, in **both**
  `on_error` modes: unlike a failure, a cancellation is never
  converted into a terminal `RunErrorEvent`. `Run.cancel()`,
  `events()`, `docs/streaming.md` and an existing test all said so
  already; the exception's own docstring was the lone dissenter.

### Notes

- No behaviour change for existing consumers: one new method, two
  corrected docstrings, no control-flow edits.
- The escape-hatch guide no longer lists `cancel()` / `fork()` as
  drop-downs — `fork()` had been on the handle since that guide was
  written. `replay()` is now the only live-handle method that still
  requires the underlying `Run`.
- A cancelled run crossing `wire.project()` lands as a terminal
  `ErrorFrame` carrying `exc_class="RunCancelledError"`, not a raise:
  `as_terminal_safe()` catches `Exception`, and `RunCancelledError`
  is one. Long-standing behaviour, newly pinned by a test.

## [2.2.5] — 2026-08-20

The lifecycle defect [2.2.4] left open, fixed at its real source. A
conversation-scoped elicitation client is pooled once and then
outlives the handle that minted it, but the handler it must reach is
per-handle state. The minting handle's handler was baked into the
client, so an elicitation raised on any later send of the same
conversation was routed at a settled run — dropped as undeliverable
(the `run buffer already closed` warning [2.2.4] added) while the
server went on blocking against its own elicit deadline. The first
send delivered; every send after it was silent.

### Fixed

- **Conversation-scoped pooled clients dispatch elicitation through
  the acquiring handle's current handler.** `_acquire_entry` rebinds a
  `_HandlerDispatch` slot on every lease instead of constructing the
  client around one handler, so a prompt raised on any send reaches
  the handle that made the call — and its answer resolves against a
  forwarder that is still registered. Identity-pool entries stay
  handler-less so a surprise elicit still fails loud. The
  trampoline's documented contract ("the factory is consulted per
  elicitation — never captured") now holds at the pool too.
- **Conversation-scoped pool keys carry the credential fingerprint.**
  A rotated overlay mints a fresh private client instead of riding
  the first send's headers for the conversation's lifetime; headers
  live in the transport and cannot be swapped on a warm client. The
  identity pool has always re-keyed rotations this way.

### Notes

- `Conversation._emit` is unchanged — its warnings stay as the net
  for genuinely undeliverable frames.
- `ClientCounts.conversation` may transiently read one higher for a
  single conversation while a rotated-out client idles out.
- Both fixes are pinned by regression tests that fail against
  [2.2.4]: events surviving a reopened handle, per-send dispatch, and
  the rotation re-key.

## [2.2.4] — 2026-08-19

An observability release. `Conversation._emit` is best-effort by
design: an event whose run has no active buffer, or whose buffer
already closed, is dropped rather than failing the caller. Both
branches were bare returns, so a drop left no trace anywhere — and
for an elicitation that is indistinguishable from a person who
simply never answered, because the MCP server goes on blocking
against its own elicit deadline either way. The drops themselves
are unchanged here; they are now merely visible.

### Added

- **Both `Conversation._emit` drop branches log at `WARNING`.** The
  message names the branch that fired — `Dropping <Event> <event-id> on
  conversation <conversation-id>: no active run buffer` when the run's
  lineage carries no buffer, and `… run buffer already closed` when
  `buffer.put()` raises on a settled run — so a dropped HITL prompt can
  be correlated with the request a server is still blocking on. `_emit`
  stays best-effort and still never fails an elicitation.

### Notes

- Diagnostic only. A dropped event is still dropped, and the second
  branch — a buffer that closed while an elicitation raised during tool
  execution was still resolving — remains an open lifecycle defect.
- Each branch's WARNING is pinned by a test, conversation and event ids
  included; both fail against a bare return.

## [2.2.3] — 2026-08-18

A replay-correctness patch written from a production incident: a turn
that died mid-tool-loop (a provider 429, in the incident) left its
conversation **permanently dead** — every subsequent message failed in
under a second with `data did not match any variant of untagged enum
ToolCallWrapper`, before any inference ran. Measured blast radius at
fix time: 5 of 207 episodes (30 days) sat in that state; one user
retried 11 times into the same wall. The poison was generated at
**replay time** from healthy stored rows, so this release revives those
conversations in place — no data migration.

### Fixed

- `TensorZeroMemory.load_history` replays output-side tool calls with
  the shape the gateway actually serves. The converter read
  `item.arguments` expecting the raw JSON string, but
  `ContentBlockChatOutputToolCall` carries `arguments` already PARSED
  (a dict — `None` when invalid) and the string in `raw_arguments`;
  the dict flowed into `ToolCall.raw_arguments` (a dataclass, no
  runtime validation) and onto the wire, where the input side demands
  `str` in every `ToolCallWrapper` variant. Now the raw string is
  preferred, a str `arguments` (older stored shapes) is accepted, and
  a dict-only shape is `json.dumps`-serialized as a last resort;
  `raw_name` is mirrored the same way — an INVALID stored call has
  `name=None` but always carries `raw_name`.
- `load_history` drops `tool_call` blocks from the appended final
  assistant message (debug-logged). They are unanswered by
  construction — their results would live in the input of a next
  inference that never ran — and OpenAI-compatible providers reject an
  assistant tool call followed directly by a user message. The model
  re-decides the call instead; a normal turn ends in text and is
  untouched.

### Notes

- The replaced test stubs modeled `arguments` as a str with no
  `raw_arguments` — a shape the gateway never returns — which is why
  they passed on broken code. The new tests build the real client
  dataclass (`tensorzero==2026.4.0`) and pin the wire dict; 3/3 fail
  against the previous converter.
- No API change. Only replay-time construction changed; nothing about
  stored data or the write path moved.

## [2.2.2] — 2026-08-10

A one-shape follow-up to 2.2.1, found by verifying 2.2.1 in production
rather than by reading the code. 2.2.1 recovers a session a server
**rejects**, which is what a redeploy produces. It does not recover a
client whose session simply **vanished**, which is what a server's
*absence* produces — scaled to zero, evicted, or crash-looping long
enough for the transport to die. Nothing is rejected in that case, so
neither session predicate matched, `rebuild_if` never fired, and the
handle stayed dead for the lifetime of the process while the server sat
there healthy. Measured: three consecutive turns failed to reach a server
that had been back for minutes; only a restart cleared it.

### Fixed

- `MCPRuntime` now recognises a third session-loss shape — fastmcp's
  `RuntimeError("Client is not connected…")` — and recovers from it on
  both the inventory and the call path. A server that was fully absent
  and came back no longer requires a consumer restart, closing the gap
  2.2.1 left behind for that trigger.
- The call path **re-sends** once on this shape. The safety argument is
  stronger than for the `32600` shape: fastmcp raises from the
  `Client.session` property on attribute access, before any network I/O,
  so the request provably never left the process. (The bare-400 shape
  keeps its no-resend rule — it dies *after* dispatch.)

### Notes

- The predicate matches fastmcp's message, not a bare `RuntimeError`;
  matching the type alone would swallow unrelated bugs and could re-send
  an order write. Pinned by a test.
- No API change. `docs/error-handling.md` § Session loss now documents
  all three shapes in one table.

## [2.2.1] — 2026-08-07

A resilience patch over 2.2.0, written from a production incident: one
mounted MCP server that stopped answering hung every conversation in the
process — silently, with no error and no log. The tool-inventory fan-out is
now bounded and degrades instead of blocking; session-loss recovery reaches
the inventory path, so a server redeploy no longer needs a consumer
restart; a session rejected with a bare HTTP 400 is recovered; and
in-stream elicitation survives derived fluent handles. Nothing breaks — the
only public type change *widens* a Protocol.

### Added

- **`MCPRuntime(tool_list_timeout=)` — a per-server budget for the tool
  inventory.**  Each handle's share of the `list_tools()` fan-out gets
  `tool_list_timeout` seconds (default `10.0`), covering the fetch and any
  session-loss recovery inside the one budget.  Deliberately its own
  constant rather than derived from `MCPServer(timeout=)`: a read window
  sized to clear an elicitation wait is no deadline at all.
- **`AgentHandle.conversation()` carries the identity kwargs.**  The public
  Protocol declared only `id` / `hydrate` / `user` / `tags` while the
  concrete `@app.agent` wrapper had always accepted `auth`,
  `elicitation_handler`, `memory` and `tools` as well.  Consumers typing
  against `AgentHandle` therefore needed `cast(Any, handle)` — switching
  off type-checking on the call that opens every turn — and multi-tenancy
  was unreachable through the public type.  The two signatures now agree.
  Type-surface only; the implementation already satisfied the wider shape,
  so structural conformance is unchanged.  This was one of the original
  `2.2.1` follow-ups, stranded when the release was never cut; it lands
  here.

### Fixed

- **`list_tools()` no longer hangs on an unreachable server.**  The
  inventory fan-out was an `asyncio.gather` with no timeout and no
  `return_exceptions`, so a single server that never answered meant the
  turn never reached the LLM — and because the SDK swallows a read timeout
  on a dead request stream, there was no ceiling short of whatever the
  caller imposed.  Each handle now runs inside its own `tool_list_timeout`
  budget and the fan-out tolerates failures, degrading in order: a dead
  session is rebuilt and re-listed once; failing that, the handle's
  **last-known** inventory is served; a server that has never answered is
  excluded from the merge with a warning.  **The tradeoff to know about:**
  a tool *added* while a server is unreachable stays invisible until that
  server answers again.  The last-known list is held separately from the
  60 s TTL cache so `refresh()`'s invalidate-all cannot wipe the fallback
  exactly when it is needed.  Relatedly, a `connect()` interrupted
  mid-`enter_async_context` now re-mints its client under the lock it
  already holds instead of leaving a half-entered stack for the next caller
  to re-enter — cancelling a wedged handle no longer converts a transient
  outage into a permanent one.
- **Session-loss recovery now covers the inventory path, not just calls.**
  A server that merely *rolled* — its warm session dead, the server itself
  answering — used to poison its handle until the consuming process
  restarted: the fan-out saw the rejection, fell into the degradation arm,
  and served a stale list forever.  `_ServerHandle.tools()` now rebuilds
  and re-lists once on either rejection shape, inside the same
  `tool_list_timeout` budget, with a generation guard that collapses
  concurrent rebuilds into a single one.  Listing is read-only, so unlike a
  tool *call* it retries on both shapes.  **This retires the "restart the
  consumer after every MCP-server deploy" rule.**  The unprefixed-name
  routing lookup goes through the same wrapper and self-heals too.
- **A session rejected with a bare HTTP 400 rebuilds the handle.**  Servers
  whose SDK predates the `404 → 32600` mapping reject a stale session with
  a plain HTTP 400, which the transport raises as `httpx.HTTPStatusError`
  before anything is decoded as JSON-RPC — so the existing
  session-terminated detection never saw it and the handle stayed dead for
  every later call.  The client is now rebuilt on that shape, but the call
  is **never re-sent**: a 400 tears the whole transport down and fails
  every request in flight on that handle, including ones the server already
  dispatched and ran to completion, and the client cannot tell pre- from
  post-dispatch.  The `404 / 32600` arm still re-sends exactly once — that
  error is synthesised per rejected POST and carries the request's own id,
  which is what makes resending provably safe there.  The asymmetry is
  deliberate; see `docs/error-handling.md` § Session loss.
- **Derived fluent runs share their root's consumption state.**
  `Conversation.send()` pinned `_pending` to the original `FluentRun` while
  `.with_(...)` / `.json(...)` / `.text()` / `.as_()` derive a new one, so
  in-stream elicitation events resolved against a run that was never built
  and both HITL frames were dropped.  Every run derived from one root now
  shares a lineage cell, and the conversation resolves its buffer through
  that.  Two symptoms go with it: `ElicitationRequested` /
  `ElicitationResolved` now reach the stream on a derived send, and a
  fully-drained derived send no longer trips the sequential-send guard with
  `ConfigurationError` on the *next* `send()` — a pattern
  `docs/glow/conversations.md` already documented as supported.
  `FluentRun.consumed` (and the post-consumption `cost()` / `usage()` /
  `feedback()` / `fork()` helpers) are family-aware accordingly.  Consumers
  re-pointing `conv._pending` by hand can stop.

## [2.2.0] — 2026-07-21

The fastmcp vertical: mounting MCP servers on a glow `App` is now Pythonic
and efficient for many concurrent conversations in one process. The
composite-proxy remote path is replaced by a native multi-server runtime
(one warm connection per server); identity becomes a per-conversation
dimension over a single long-lived App; the two agent loops share one turn
engine; and timeout truth, structured content, testing DX, and tool-call
observability round it out. Additive over 2.1.2 — the only behavior changes
are the `tool_timeout=None` fix, the streaming no-dispatch parity fixes, and
non-text tool content no longer being dropped (all flagged below). Validated
against fastmcp 3.2.4.

### Added

- **`MCPRuntime` — a native multi-server router behind `ToolProvider`.**
  glow's `App` now routes every `MCPServer` spec (and the local
  `@app.tool` server) through direct, long-lived fastmcp Clients — one
  warm connection per server shared by every agent and conversation —
  instead of a composite FastMCP with `create_proxy` mounts that
  reconnected (~5 HTTP requests) on every remote tool call.  A warm call
  is now a single POST.  Includes: mount-parity namespace prefixing,
  60 s TTL inventory caching with `tools/list_changed` invalidation
  (in-memory servers stay live-linked), `refresh()`, and transparent
  session-terminated recovery (the client is rebuilt and the rejected
  call re-sent exactly once — safe, the server rejected it before
  dispatch; all other failures propagate untouched).
- **`run_agent` accepts `progress_callback`.**  The tool executors have
  always accepted a per-call progress relay, but only the streaming loop
  wired it — server-side ``ctx.report_progress(...)`` notifications were
  silently dropped on non-streaming runs.  ``run_agent`` now forwards an
  optional ``progress_callback`` (same ``ProgressCallback`` shape) to the
  executors on every dispatch turn.
- **`ToolResultEvent.structured`.**  Streaming tool-result events now carry
  the tool's raw structured payload (``CallToolResult.structured_content``) as
  an additive ``structured: dict | None`` field, so consumers get structured
  data without re-parsing the model-visible ``result`` string.  Round-trips
  through the codec with no version bump (additive).
- **First-class in-memory testing for glow MCP mounts.**
  `t0_agents.testing` gains `glow_tools(*servers)` — an async context
  manager that mounts `MCPServer` specs on an in-memory glow `App` and
  yields their tools as a `ToolProvider`, driving the real fastmcp
  client→server→schema→result boundary with no sockets;
  `ScriptedElicitation`, a scripted forwarder-backed `ctx.elicit` handler
  that resolves each request inline for socket-free human-in-the-loop
  round-trips; and `tool_call(name, **arguments)`, which builds a
  `ToolCall` content block from the name and keyword arguments in place of
  the five-field literal.  Documented in the new "Glow MCP servers" chapter
  of `docs/testing.md`.
- **Per-tool-call observability on `ToolResultEvent`.**  Streaming tool-result
  events gain additive `duration_ms: float | None` (the `call_tool`
  round-trip; `None` when the call never dispatched) and `server: str | None`
  (the owning mounted server's namespace; `None` for an unprefixed / single
  local mount) — both computed once in the shared executor so the two agent
  loops agree, observable via `run.events(ToolResult)`.  Additive through the
  codec, no version bump.
- **In-stream elicitation events.**  `ElicitationRequestedEvent` /
  `ElicitationResolvedEvent` join the `AgentEvent` union (selectors
  `ElicitationRequested` / `ElicitationResolved`), carry a shared correlation
  `id`, and project onto the public wire's existing `elicitation_request` /
  `elicitation_resolved` frames — making a human-in-the-loop prompt observable
  in-stream, coexisting with the out-of-band forwarder.  Automatic emission of
  these events lands with per-conversation elicitation clients; the events,
  codec, selectors, and wire projection are in place now.
- **`App.tools()` — the public tool inventory.**  Returns the merged,
  namespace-prefixed inventory of mounted tools (booting the MCP infrastructure
  on first call; `[]` when the App mounts none), replacing the private
  `app._owned_client.list_tools()` idiom.  `refresh=True` drops the per-server
  TTL caches and re-lists.
- **Identity per conversation — multi-tenant apps over ONE long-lived App.**
  `conversation()` gains the identity dimension: `auth=` maps mounted server
  namespaces to per-conversation credentials (a headers mapping, an
  `httpx.Auth`, or the new `ServerCredentials` — exported from
  `t0_agents.glow`); `elicitation_handler=` (an `ElicitationForwarder` or any
  bare handler; rebindable between turns) routes HITL prompts to THIS
  conversation; `memory=` swaps the memory provider per conversation
  (tenant `tag_filter` scoping, hydrate included); `tools=` narrows tool
  visibility per conversation with the `@app.agent(tools=)` grammar,
  intersected with the agent's own filter.  The runtime pools warm clients
  by `(server, identity)` — fingerprinted credentials, idle-TTL eviction,
  a size cap that never tears down an in-flight client — and `elicits=True`
  mounts (new `MCPServer` field) get one private client per conversation,
  the only scoping that cannot misroute a confirmation (the SDK discards
  the elicit↔call correlation id).  A conversation passing none of these
  behaves exactly as before.  Documented in the new
  `docs/glow/multi-tenant.md` chapter (with the one-App-per-tenant
  migration).
- **In-stream elicitation emission is live for conversation-scoped HITL.**
  Passing `conversation(elicitation_handler=<ElicitationForwarder>)` emits
  `ElicitationRequestedEvent` / `ElicitationResolvedEvent` on the run's
  stream with `id` equal to the forwarder's own `elicitation_id` — the id an
  out-of-band resolver answers — via the new optional
  `ElicitationForwarder(observer=...)` seam (rebindable; observers are
  best-effort and can never break a confirm).  Bare handlers emit with a
  glow-minted id.
- **Per-run `Agent.invoke(tool_provider=…, memory=…)` overrides.**  A shared
  `Agent` can run one invocation against a different tool provider (re-wrapped
  with the agent's `SubAgentToolProvider` delegation layer) and a different
  memory provider — the injection seam conversation identity rides, usable
  from the base API too.  `MCPRuntime` gains `for_conversation(...)` views,
  `client_counts()`, and `pool_idle_ttl=` / `pool_max_size=` knobs.
- **`MCPServer(timeout=…)` — per-server transport read timeout.**  Pins the
  httpx read window (maximum silent gap between bytes; any SSE frame the
  server emits resets it) on a `url` mount's connections, replacing the
  previously unreachable SDK default of 300 s.  Size it above an
  elicitation-capable server's elicit wait.  Url-only, like
  `headers=`/`auth=`; validated eagerly.
- **Docs: the timeout stack.**  `error-handling.md` now maps all four timeout
  layers (per-call deadline, session default, httpx read window, proxy inner
  hop), including the source-verified SDK hang mode (a dead request stream
  without a per-call deadline never resolves) and heartbeat guidance.

### Changed

- **Duplicate final tool names across mounts now fail loud.**  Two
  mounts producing the same prefixed name raise `ConfigurationError` at
  build/refresh; the composite silently forwarded duplicates to
  inference.  Value-identical specs registered twice (e.g. `servers=`
  plus a parameter annotation) still fold into one mount, as before.
- The public glow surface is unchanged; `create_proxy` is no longer
  used anywhere in the library.

### Fixed

- **`tool_timeout=None` now disables the per-call timeout — everywhere.**
  `Agent` coerced a `None` config value back to 30.0 s before dispatch (both
  the non-streaming and streaming terminals), so the documented "set
  `tool_timeout=None` for elicitation-capable tools" advice silently did not
  hold on the `Agent`/`AgentConfig`/`@app.agent` surface: any elicitation
  slower than 30 s died with `McpError` while the human deliberated.  `None`
  now flows through to `call_tool(timeout=None)` on every path
  (`run_agent`, `StreamingAgentLoop.stream`, and both executors widened to
  `float | None`).  **Behavior change** only for callers passing an explicit
  `None` — they now get what the docs always promised; the 30.0 s default is
  unchanged.
- **Streaming no longer dispatches tool calls whose JSON arguments failed to
  parse.**  When the LLM streamed invalid (or non-object) JSON arguments, the
  streaming loop rebuilt the call with ``arguments={}`` and executed the tool
  — a tool with all-optional parameters would run on garbage input.  The
  non-streaming loop has always refused to dispatch these (TensorZero marks
  them ``arguments=None``) and fed the model an
  ``ERROR: invalid JSON arguments...`` result instead.  Streaming now follows
  the same contract: no dispatch, same error string, ``ToolResultEvent``
  flagged ``is_error=True``.  The wire event sequence is unchanged.
- **Streaming no longer dispatches hallucinated tool names.**  TensorZero
  validates tool names against the advertised schema on the non-streaming
  path and zeroes unknown ones — the loop answers with
  ``ERROR: tool name ... is not in the schema`` without dispatching.
  Streamed chunks carry the raw name unvalidated, so the streaming loop
  forwarded hallucinated names to fastmcp and surfaced a provider
  ``ToolError`` instead.  The rebuild now applies the same zeroing against
  the advertised tool list, aligning both loops on the schema-error
  contract.
- **Non-text tool content is no longer dropped silently.**  A tool result
  whose only content was image/audio/embedded-resource blocks serialized to
  ``""`` — the model saw nothing.  It now becomes a loud placeholder naming
  the omitted content (with a ``WARNING`` log); mixed text+non-text keeps the
  text and appends a note.  **Behavior change** for tools returning non-text
  content; all-text, ``.data``, and error results stay byte-identical.
- **Streaming `ToolResultEvent.is_error` is typed, not string-sniffed.**  The
  flag came from `result.startswith("ERROR:")`, which mis-flagged a
  *successful* result that happened to render as `"ERROR: …"`.  It now comes
  from the executor's typed outcome (the dispatch error branch or
  `CallToolResult.is_error`).  The model-visible `result` string is unchanged.

## [2.1.2] — 2026-07-17

### Fixed

- **Replayed thought blocks stripped from the streaming loop too.**  `2.1.1`
  fixed the non-streaming `run_agent`, but the streaming loop
  (`stream/runner.py`) still replayed each assistant turn with its `Thought`
  block intact, so thought-less providers (e.g. groq) kept rejecting those
  input thought blocks with a WARN on the streaming path.  The streaming loop
  now strips `Thought` at the same replay-append boundary via the shared
  `_strip_thoughts` helper, so both loops behave identically.  The turn's
  local blocks are untouched, so the returned answer and the
  `response_source == "thought"` fallback are unchanged.

## [2.1.1] — 2026-07-15

### Fixed

- **Replayed thought blocks stripped from the intra-run loop.**  The agent
  loop appended each assistant turn to its replay history with any reasoning
  `Thought` block intact; on the next turn TensorZero rejected those input
  thought blocks for thought-less providers (e.g. groq) and dropped them with
  a WARN.  `run_agent` now strips `Thought` blocks before appending the
  assistant turn, matching the cross-conversation strip already applied by the
  memory subsystem.  Provider-agnostic (no capability signal exists in the
  pinned SDK); the returned answer and the `response_source == "thought"`
  fallback are unchanged.

## [2.1.0] — 2026-07-15

Production-mount ergonomics for glow's App-owned MCP client — the three
seams the v2.0.0 pilot needed, deliberately kept out of the clean 2.0.0
tag and landed here.  All additive; core is untouched.

### Added

- **Human-in-the-loop — `App(elicitation_handler=…)`.**  Binds a FastMCP
  `ElicitationHandler` (e.g. `ElicitationForwarder.handler`) onto the
  App-owned composite `Client`, so a mounted tool's `ctx.elicit(...)`
  reaches a real resolver instead of surfacing "Elicitation not
  supported".  A `Client` binds exactly one handler and the App owns one
  Client, so this is one resolver per App — construct one App per
  session/tenant for isolated HITL.  With no handler the default is
  unchanged (elicitation unsupported).
- **Authenticated remote MCP mounts — `MCPServer(headers=…, auth=…)`.**  A
  `url=` mount carries a static `headers` dict and an `auth`
  (`httpx.Auth`, applied per request); when either is set the mount rides
  an explicit `StreamableHttpTransport` / `SSETransport` (per `transport`)
  so identity reaches the server.  This is the first-class replacement for
  hand-building `create_proxy(MCPConfig(RemoteMCPServer(...)))` and
  wrapping it in a local `MCPServer` subclass.  `headers=`/`auth=` on an
  in-memory `server=` mount raises `ConfigurationError`.
- **`MCPServer` instances in `App(servers=…)`.**  `servers=` accepts
  `MCPServer` instances alongside subclasses, so a per-mount configuration
  needs no module-level subclass.
- **Per-mount prefix opt-out — `MCPServer(prefix=False)`.**  Mounts with
  `FastMCP.mount(namespace=None)` so a single-server mount presents tool
  names verbatim (the legacy-inventory case); `namespace` becomes a
  logical label only.  Multi-server prefixing is unchanged.  Passing an
  unprefixed server into `tools=` raises `ConfigurationError`, directing to
  name/glob selection.

## [2.0.0] — 2026-07-15

Glow v2 — the batteries-included, Pythonic surface for building
TensorZero agents.  Seven approved proposals (P1–P7) plus the versioned
`t0_agents.wire` contract (E2) land together: knob passthrough and
`.with_()`, functionless `model=` agents, per-agent `tools=` filtering, a
streaming production pack with tool progress, boot collapse and a CLI,
and first-class conversations.  The deliberate breaking changes are
confined to glow — core stays back-compatible — and each carries an
inline `Migration:` note.

### Added

- **First-class conversations — `agent.conversation()` / `send()` (glow).**
  A multi-turn handle over one TensorZero episode: `conv =
  agent.conversation()` (or `agent.conversation(id=eid, hydrate=True)` to
  resume), then `await conv.send("…")` — which returns the same `FluentRun[R]`
  a call does, so `.stream_text()` / `.json()` / `.with_()` stay intact.
  `conv.id` **equals the `episode_id`** (a UUIDv7 — the pinned P5 contract);
  `conv.transcript` is the in-process boundary-pair record (each turn's final
  reply, per `Run.transcript`'s limitation).  Sends are memory-strategy-aware:
  `memory_strategy="messages"` passes only `episode_id` and lets the provider
  weave history via `load_history` (also passing `prior_messages` would
  suppress it), while any other agent gets the in-process transcript seeded as
  `prior_messages` so multi-turn works everywhere.  `user=` / `tags=` set the
  send tags **and** the hydrate/list read filter from one dict; a send left
  unconsumed blocks the next.  `FluentRun` gained an additive `on_result` hook
  (fires once on success across every consumption path) and a `consumed`
  property.
- **`AgentHandle.conversation()`.**  The handle protocol now declares
  `conversation(...)`, so a conversation-capable sub-agent handle types
  without importing the private `_AgentWrapper`.
- **Dev-grade conversation listing — `app.conversations(agent, tags=…)` +
  `TensorZeroMemory.list_conversations()`.**  Groups a bounded window of recent
  inferences by `episode_id` client-side (there is no server-side aggregation
  in 2026.4.0), returning newest-first `ConversationSummary` rows; the docstring
  is explicit about the O(window) horizon and that `turn_count` is exact only
  under `memory_strategy="messages"`.  For production scale, implement the new
  `ConversationIndex` protocol (`t0_agents.core.conversation`) and pass it as
  `app.conversations(index=…)` — the seam a real inbox plugs into with no
  library change.
- **`RunErrorEvent` + `Run.events(on_error=)` — the terminal-event
  invariant.**  `Run.events(on_error="event")` converts a fatal
  drive-time failure into a single terminal `RunErrorEvent`
  (`error_type` = exception class name, structured `detail` such as
  `{"max_turns": 20}`, and the original `exception` for in-process
  consumers) instead of raising mid-iteration — so every stream in
  that mode ends with exactly one `DoneEvent | RunErrorEvent`.  The
  core default stays `on_error="raise"`; cancellation is never
  converted.  `ErrorEvent` remains the recoverable inline anomaly.
- **`ToolProgressEvent` — MCP tool progress reaches the stream.**
  `execute_tool_calls` / `execute_tool_calls_parallel` accept a
  `progress_callback`; server-side `ctx.report_progress(...)`
  notifications (previously dropped inside the loop) arrive tagged
  with the emitting `ToolCall` block id, and streaming runs push them
  into the event buffer as `ToolProgressEvent`s interleaved between a
  `ToolCallEvent` and its `ToolResultEvent`.
- **`Run.stream_text(*, fallback=True)` — the reasoning-safe text
  one-liner.**  Yields text deltas as they arrive; a turn that
  streamed no non-blank delta (reasoning models route the answer
  through thought summaries) yields the final `result.response` once
  after the drain.
- **`t0_agents.stream.codec` — versioned event dict codec.**
  `event_to_dict` / `event_from_dict` round-trip every `AgentEvent`
  variant through plain JSON-serialisable dicts (`{"v": 1, "type":
  ..., ...}`); unknown discriminators and version mismatches raise
  `ValueError`.
- **`t0_agents.stream.status` — status-hint synthesis.**
  `synthesize_statuses(events, known_servers=...)` interleaves opaque
  `Status` hints ("Calling {tool}…", "Thinking…") with the
  production-proven dedup/suppression policy; `split_tool_identity`
  decomposes namespaced tool names (longest server prefix first).
- **Glow streaming surface.**  `FluentRun.stream_text(fallback=True)`,
  `FluentRun.statuses()` (namespace-aware via the App's mounted MCP
  servers), and `TextEnd` / `RunError` / `ToolProgress` selectors.
- **Per-agent tool filtering — `@app.agent(tools=[...])`.**  An agent can
  now restrict which of the App's composed tools it sees: entries are
  `@app.tool` callables (matched by name), `MCPServer` classes/instances
  (matched by their `namespace` prefix), and glob strings (e.g.
  `"customers_get_*"`).  Omitting `tools=` is unchanged — the agent sees
  every tool; `tools=[]` is explicitly toolless.  Patterns matching no live
  tool warn at `build()`, turning prefix-guessing bugs into a build-time
  diagnostic; an `MCPServer` used without `namespace=` raises.  Sub-agent
  `<name>_ask` tools are never filtered.  Fully additive — agents without
  `tools=` behave exactly as before.
- **`t0_agents.tooling.FilteredToolProvider`.**  A standalone `ToolProvider`
  wrapper exposing only an allow-listed set of tool names — the mechanism
  behind `@app.agent(tools=...)`, reusable on its own.
- **Functionless agents — `model=`.**  `AgentConfig` (and `@app.agent`)
  accept `model=` — a `{provider}::{model}` shorthand or a `[models.*]`
  name — routing inference through the gateway's built-in passthrough
  with **zero tensorzero.toml**.  Tools, episodes, streaming, and tags
  work unchanged.  `model` XOR `function_name` (neither → the
  back-compat `"agent"`); `variant=` pins a declared function's variant
  and is rejected alongside `model=` (mirroring the gateway); JSON
  output modes are rejected on the `model=` path at build time with a
  ready-to-paste TOML stanza — the passthrough ignores per-call
  `output_schema` (verified against gateway 2026.4.0).  New sibling
  knobs `params=` (merged under output-mode params) and `cache=`
  (TensorZero `cache_options`) forward to every inference, on both
  routing paths.
- **Observability auto-tag.**  Every `Agent`-driven inference now
  carries `t0_agents.agent = <config.name>` (per-call and config tags
  win on collision) — per-agent ClickHouse segmentation on shared
  functions and the only per-agent handle under `tensorzero::default`.
  Behaviour note: `tags` is never `None` on Agent-driven inferences
  anymore.  `TensorZeroMemory(tag_filter={"t0_agents.agent": name})`
  is the documented cover for the functionless memory blind spot.
- **Client-side prompt templates — `.with_(prompt_vars={...})`.**  The
  docstring (or an explicit `system=`) may carry `str.format`
  `{placeholders}` filled per call, entirely client-side.  Mutually
  exclusive with `system_arguments` (TZ server-side templates), which
  in turn is rejected on `model=` agents up front — the passthrough
  has no system template.
- **`t0_agents.wire` — the versioned wire contract.**  A frozen
  15-frame `WireFrame` union (the event union plus consumer-facing
  `Init` / `Status` / `Elicitation*` frames) with a versioned dict
  codec (`frame_to_dict` / `frame_from_dict`, `WIRE_VERSION = 1`,
  additive-only evolution) and pure composable projection stages:
  `project()` (exhaustive event→frame mapping with byte-bounded tool
  payloads), `as_terminal_safe()` (a raised exception becomes one
  terminal `error` frame — same frozen shape as the `RunErrorEvent`
  path, `max_turns_exceeded` token and `detail.exc_class`
  discriminator included), `with_status()` (production status-hint
  policy at frame level), `with_text_fallback()` (reasoning fallback +
  never-blank belt), and `public_allowlist()` (the frozen 8-name
  public projection and privacy guard — tool activity and reasoning
  never reach a public client).  Explicitly not a broker: no transport
  imports; consumers serialise the emitted dicts onto their own bus /
  SSE / WebSocket.  The production byte-compat golden is ported at
  dict level as a library test (`tests/test_wire/test_public_golden.py`)
  and every stage `match` ends in `assert_never`, so upstream event
  churn is absorbed in the same PR that introduces it.  See
  `docs/wire.md`.
- **Boot collapse — `App(gateway=...)`, `App(embedded=...)`, and the
  deferred `App()`.**  `App` no longer needs a pre-built gateway:
  `gateway=` takes an HTTP gateway URL, `embedded=` a
  `tensorzero.toml` path (the docker-free in-process dev mode —
  degraded observability without ClickHouse: no episode queries,
  ephemeral feedback), and a bare `App()` resolves `TENSORZERO_URL` →
  `TENSORZERO_CONFIG` at first use, erroring clearly when neither is
  set and an agent actually needs a gateway.  Resolved gateways are
  App-owned (closed by `shutdown()` / the exit of the boot bracket);
  a caller-supplied `t0=` is never entered or closed.
- **Auto-build on first consumption + `async with app:`.**  Consuming
  an agent on an unbooted App now runs the full infrastructure boot
  (one INFO log): gateway resolution, MCP mounts, client connection,
  and `tools=` filter resolution — eliminating both halves of the old
  consume-before-lifespan trap (silently unfiltered/toolless cached
  agents; opaque disconnected-client errors).  `async with app:` is
  the new explicit boot bracket, byte-equivalent to
  `async with app.lifespan():`.
- **REPL surface — `@app.agent(name, replace=True)` +
  `App.remove_agent(name)`.**  Re-decorating a name in a notebook no
  longer dead-ends on the duplicate-name error (which remains the
  default); replacements registered after the App started still get
  their gateway and `tools=` filter resolved before first use.
- **CLI — glow App specs and one-shot messages.**  `t0-agents run` /
  `serve` / `info` accept a `module:attr` spec that resolves to a glow
  `App` instance (`--agent` picks one of several registered agents;
  the sole agent is the default).  `run` additionally takes an
  optional positional MESSAGE for one-shot invocations:
  `t0-agents run examples.glow.01_hello:app --agent tutor "hi"`.
- **`t0_agents.serve.create_app_from_glow`.**  Async bridge from a
  glow `App` to the serving layer: boots the App, selects an agent,
  and delegates to `create_app` (`share_client=True` default over the
  App-owned client; the caller owns the glow lifecycle).

### Changed

- **Breaking — glow `FluentRun.events()` defaults to
  `on_error="event"`.**  A glow stream never raises mid-iteration: a
  fatal failure ends the stream with one terminal `RunErrorEvent`.
  Core `Run.events()` keeps the raising default.  Migration: consumers
  that relied on `try/except` around a glow event loop should either
  match on `RunError` / `RunErrorEvent` or pass `on_error="raise"`.
- **Breaking — glow `Text` selector retargeted to `TextDeltaEvent`.**
  The old binding selected `TextEndEvent`, a completion marker that
  carries no text.  Migration: consumers that wanted the marker use
  the new `TextEnd` selector; consumers that built a manual
  `Selector(TextDeltaEvent)` can now just use `Text`.
- **Breaking — `ToolProvider.call_tool` grows a `progress_handler`
  keyword.**  Custom providers (and wrapper providers, which must
  forward it) should add `progress_handler: ToolProgressHandler | None
  = None` to their signature.  The agent loop only passes the kwarg
  when progress is actually wired, so providers predating it keep
  working on every progress-free path (`run_agent`, streaming without
  a sink).  All in-library providers — filtered, guarded,
  cost-tracking, delegation — forward it.
- **`ScriptedAgent(raises=...)` now replays its scripted `events=`
  before raising**, modelling a mid-stream abort (previously the
  exception fired before any event was delivered).
- **Breaking — glow configuration errors move from the wrapper call to
  first consumption.**  Calling `tutor("hi")` is now side-effect-free
  (a deferred `App()` hands out handles before any gateway exists);
  the underlying Agent is resolved when the run is first consumed —
  `await`, `.result()`, or the first iteration of `.events()` /
  `.stream_text()` (which now build at first iteration rather than at
  the call; selector validation stays eager).  Migration: code that
  wrapped the *call* in `try/except ConfigurationError` should wrap
  the *consumption*; everything that awaited the handle before
  observes errors exactly as before.

### Removed

- **Docs: the "No per-agent MCP subsets" limitation.**  Resolved by
  `@app.agent(tools=[...])`.

### Fixed

- **Elicitation forwarder `response_type` leaked an unstable class
  name.**  The request envelope exposed FastMCP's generated
  `"ScalarElicitationType"`; it is now derived from the requested schema
  (`bool` / `int` / `float` / `str`), matching
  `ElicitationRequestFrame.response_type`.
- **`ToolProgressFrame` was stricter than its source event.**  `total` /
  `message` now default to `None`, matching `ToolProgressEvent`, so a
  hand-built frame needs only `id` and `progress`.
- **Docs: the escape-hatch middleware recipe.**  It imported a
  non-existent `t0_agents.middleware.TimingMiddleware` and drove the
  private `._agent`; it now uses the real `CostTrackingMiddleware` and
  documents middleware as a first-class glow knob (`App` / `@app.agent`
  / `.with_(...)`).
- **Docs: the getting-started `cost` claim.**  `.result()`'s
  `AgentResult` carries no aggregated `cost`; the guide and the
  `handle.py` docstrings now point at the run handle's `.cost()` /
  `.usage()`.
- **`examples/glow/06_orchestrator.py` crashed before build.**  It poked
  `orchestrator._agent.config.max_turns` while `._agent` was still
  `None`; it now passes `max_turns=` to `@app.agent(...)`.
- **`examples/glow/03_streaming.py` printed block ids as text.**  It now
  leads with `stream_text()` and shows the `Text` / `TextEnd`
  distinction.

## [1.1.0] — 2026-06-22

Live token usage lands as a cohesive, additive feature: the live per-turn
context footprint on `Run`, the episode-level `ConversationUsage`
accumulator, sub-agent billing roll-up, and the serving `usage` block —
plus the public-readiness cleanup previously staged for 1.0.1.  Every
addition is back-compatible; no existing behaviour changes.

### Added

- **Live per-turn context footprint on `Run`.** `Run.context_footprint()`
  returns the latest inference's usage — roughly the current
  full-conversation prompt size — as a `t0_agents.core.types.Usage`,
  distinct from the cumulative `Run.usage()` sum (which double-counts the
  prompt re-sent each turn). This is the signal an outer layer compares
  against the model's context window to decide when to compact or rotate a
  conversation. `Run.usages()` exposes the per-inference breakdown
  (including cache-served zero turns, which the footprint skips), and
  `Run.context_remaining(window)` returns `window -
  footprint.input_tokens` for a caller-supplied window (not clamped — may
  be negative when over-window).
- **`t0_agents.core.types.Usage`** — a per-inference, lossless mirror of
  `tensorzero.Usage` (`input_tokens`, `output_tokens`,
  `provider_cache_read_input_tokens`, `provider_cache_write_input_tokens`,
  `cost`) with a `from_tz(...)` constructor and a `total_tokens` property.
- **`RunUsage`** gains `provider_cache_read_input_tokens`,
  `provider_cache_write_input_tokens`, and `cost_is_priced` (all defaulted,
  so construction stays back-compatible). `cost_is_priced` is `True` only
  when a provider actually reported a price, distinguishing a real `$0.00`
  from missing pricing. Existing `RunUsage` fields and `Run.cost()` /
  `Run.usage()` behaviour are unchanged.
- **`ConversationUsage` — an episode-level usage accumulator.** A conversation
  is the many `invoke()`s that share one `episode_id`; nothing aggregated usage
  across them. `ConversationUsage(episode_id)` records each completed `Run`,
  summing **billing** across the conversation (`.total`, a `RunUsage`) while
  tracking the current context **footprint** as a *level* — the latest Run's
  `context_footprint()`, not a sum. `.remaining(window)` and
  `.fraction_used(window)` answer "how much context is left" against a
  caller-supplied window. A Run with no footprint (a failure before the first
  turn, or a whole-response cache hit) leaves the prior level intact rather than
  blanking the live signal. Exported from the package root.
- **Sub-agent token usage now rolls up into the orchestrator's billing.** When
  an agent delegates to a sub-agent, the child's `RunUsage` is carried home on
  the delegated tool result and folded into the parent `Run.usage()` /
  `Run.cost()`, so billing reflects the whole delegation tree. The parent's
  `context_footprint()` is deliberately **unaffected** — a sub-agent runs in its
  own context window, so only billing rolls up, never the footprint.
- **The serving layer now reports live token usage.** `POST /chat` and the
  SSE / WebSocket streaming `done` frame carry an optional, additive `usage`
  block: a `footprint` (the current prompt size — the compaction signal) and a
  `cumulative` total (summed billing, including delegated sub-agents, with cache
  fields and `cost_is_priced`). Over a WebSocket the block additionally carries a
  per-session running total (`usage.session`) that accumulates across the
  connection's turns, keyed per `episode_id`. The block is purely additive — every
  pre-existing response and `done`-frame key is unchanged, so existing clients are
  unaffected. The model's context window is not knowable in-process, so the block
  reports the raw footprint and the caller computes "remaining" against the window
  it knows.
- **`t0_agents.__version__`** attribute (`str`), kept in sync with
  `pyproject.toml`.  Lets downstream code do
  `import t0_agents; t0_agents.__version__` without falling back to
  `importlib.metadata`.

### Changed

- **README** — Features table now lists `ElicitationForwarder`
  alongside Memory and Observability so external developers can see
  human-in-the-loop forwarding is first-class.  New "Status &
  Limitations" section consolidates the alpha / experimental / known
  caveats narrative previously scattered across the intro and the glow
  docs.

### Removed

- Internal workstream planning files (`PLAN.md`, `dev-logs/`) and
  remaining "milestone", "dev-log", "roadmap", and "experimental
  branch" references in `CHANGELOG.md` and `CLAUDE.md`.  The
  external-developer experience no longer surfaces internal milestone
  vocabulary.

## [1.0.0] — 2026-05-21

The v1.0 four-abstraction refactor lands as one cumulative release.
Four new abstractions — `Run[T]`, `Middleware`, `OutputMode[T]`,
`Composer` — give power users a first-class surface for invocation
handles, cross-cutting concerns, output polymorphism, and topology
orchestration without forking `core/agent.py`.  Two v0.12 surfaces are
removed: `Agent.run()` / `Agent.stream()` (replaced by
`Agent.invoke(...) -> Run[T]` and `run.events()`) and the five
JSON-only `AgentConfig` kwargs (replaced by `output=JsonOutput(...)`).
Both are one-line migrations per call site; the migration walkthrough
is at [`docs/migration-v0.12-to-v1.0.md`](docs/migration-v0.12-to-v1.0.md).
`mcp/` is largely untouched across the refactor — the only signature
change is that `run_agent` (and `StreamingAgentLoop.stream`) rename
their second parameter from `mcp_client` to `tool_provider` and
default it to `None`, so chat-only callers no longer need to allocate
a FastMCP server (see the `Changed` section below).

### Added

#### `Run[T]` — first-class invocation handle

- **`Agent.invoke(user_message, *, ...) -> Run[T]`** is the canonical
  entry point.  Returns a `Run` you can `await` for the final
  `AgentResult`, iterate via `events()` for the streaming
  `AgentEvent` sequence, cancel, replay, fork, or attach feedback to.
  Accepts `prior_messages` (used by `Run.fork`), `middleware`,
  `output`, `eager`, and `context` in addition to every kwarg the
  v0.12 `Agent.run()` accepted.
- **`Run[T]`** (in `t0_agents.core.run`) — public methods: `result()`,
  `events(replay=False)`, `stream_text()`, `__await__`, `cancel()`,
  `feedback()`, `replay()`, `fork()`, `transcript()`, `cost()`,
  `usage()`.  Eager by default; `eager=False` opt-in for queueing.
  Holds a `weakref` to its Agent so a discarded Run does not extend
  gateway lifetime.
- **`RunInput`** — frozen, slotted dataclass capturing the immutable
  snapshot of one `invoke()` call.
- **`RunState`** — `Literal["pending", "running", "succeeded",
  "failed", "cancelled"]` lifecycle alias.
- **`RunCancelledError`** — raised by `await run.result()` after
  `await run.cancel()`.
- **`RunFailedError(cause)`** — defensive fallback raised by
  `await run.result()` when the drive failed without a captured
  exception; the normal failure path re-raises the original
  exception (`MaxTurnsExceededError`, `GuardrailBlockedError`, ...)
  directly.

#### `Middleware` — composable cross-cutting concerns

- **`Middleware` Protocol** — `name: str` + `async __call__(ctx, next_) -> AgentResult`.
- **`RunContext`** — mutable per-Run state threaded through the
  chain.  Methods: `chain_on_inference(cb)` (compose multiple
  inference observers), `replace_tool_provider(tp)` (records prior
  providers on `ctx.extras["_tool_provider_stack"]` for audit).
- **`build_pipeline(middleware, terminal) -> Next`** — right-fold
  composition with `_NextGuard` double-call detection
  (`allow_retry = True` opt-in skips the guard).
- **Six built-in middlewares:** `MemoryMiddleware`,
  `GuardrailsMiddleware`, `CostTrackingMiddleware`,
  `SubAgentMiddleware`, `JsonRetryMiddleware`,
  `ObservabilityMiddleware` (opt-in; accepts any tracer exposing
  `start_as_current_span`).
- **`AgentConfig.middleware: list[Middleware] | None`** — `None`
  (default) auto-materialises the v0.12-compat chain
  (`Memory → SubAgents → Guardrails → CostTracking → JsonRetry → terminal`);
  `[]` is the explicit "no middleware" override; an explicit list
  replaces the default.
- **`AgentConfig.default_middleware_for(...)`** static method — single
  source of truth for the default chain shape, usable in
  power-user-composed chains that sandwich the default.
- **`RunUsage`** — frozen dataclass with `input_tokens`,
  `output_tokens`, `cost` aggregates.  Returned by `Run.usage()`.

#### `OutputMode[T]` — output polymorphism

- **`OutputMode[T]` protocol** — `expects_tool_loop` /
  `supports_streaming` capability flags; `shape_inference(req)`
  mutates the request; `materialize(response, attempts) ->
  AgentResult[T]`; `recover(failure, attempts) -> Recovery` drives
  retry/accept/fail.
- **Five built-in output modes** (`from t0_agents import ChatOutput,
  JsonOutput, StructuredOutput, ToolCallOutput, StreamingOutput`):
  `ChatOutput` (T=str, default — MCP tool loop),
  `JsonOutput(schema, json_mode, permissive, max_retries)` (single-shot
  JSON with permissive parsing + bounded retry-with-hint),
  `StructuredOutput[M](model)` (Pydantic-validated single-shot;
  `result.value` is statically typed as `M`),
  `ToolCallOutput(tool_choice)` (router pattern),
  `StreamingOutput(base=ChatOutput())` (placeholder for v1.1 streaming
  JSON; `base=JsonOutput(...)` raises `ConfigurationError`).
- **`AgentResult[T]` generic** with `value: T` as the canonical field.
  v0.12 read-side properties (`response`, `parsed`, `parse_strategy`,
  `json_parse_attempts`, `function_kind`) remain available as
  `@property` aliases (Tier 0 byte-identity).
- **`OutputDiagnostics`, `InferenceRequest`, `Recovery`,
  `ParseFailure`, `Attempt`** — the retry handshake the
  `JsonRetryMiddleware` uses to drive `OutputMode.recover`.
- **`AgentConfig.output: OutputMode = ChatOutput()`** — canonical
  config-level field.  Per-call override via
  `Agent.invoke(..., output=...)` (overloads narrow `Run[T]`).
- **`[pydantic]` extra** — `pip install 't0-agents[pydantic]'` pulls
  `pydantic>=2.0` for `StructuredOutput`.

#### `Composer` — topology primitive

- **`Composable[Input, T]` Protocol** — `@runtime_checkable`; an
  object whose `.invoke(input)` returns a `Run[T]`.  Both `Agent`
  and `Composer` satisfy it.
- **`Composer` base class** — declarative dataclass-based
  topologies; each concrete subclass exposes its parameters plus
  `name: str` and the narrow `invoke(input, *, eager, context,
  cancel_timeout)` surface (Composer.invoke does NOT accept
  Agent-level kwargs — sub-Runs use their own `AgentConfig`).
- **`ComposerError(node, cause)`** — `T0AgentsError` subclass;
  raised when a sub-Run inside a Composer fails with topology
  context.  `retry_chain` is the documented exception: it
  re-raises the underlying original exception un-wrapped.
- **`CompositeRun(Run[Any])`** — the `Run` subclass produced by
  every Composer.  Public `Run` surface works uniformly;
  `.feedback()`, `.replay()`, `.fork()` raise `NotImplementedError`
  (Agent-runs-only semantics).
- **`DEFAULT_CANCEL_TIMEOUT = 30.0`** module-level constant; every
  topology factory accepts `cancel_timeout=`; per-call override via
  `composer.invoke(input, cancel_timeout=...)` wins.
- **Nine topology builders** (`from t0_agents import ...`):
  `sequential`, `parallel`, `race`, `consensus`, `map_reduce`,
  `conditional`, `saga`, `retry_chain`, `circuit_breaker`.
- **Cancellation cascade** — `await root_run.cancel()` propagates to
  every sub-Run; overshoot logs WARN on `t0_agents.compose.topology`.
- **`parent_episode_id` audit** — every Composer driver threads
  `parent_episode_id` correlation through to children for
  ClickHouse reconstruction of nested runs.

#### Guardrails — regex content blocker

- **`regex_guardrail(pattern, message, *, phase="after_run")`** —
  new built-in factory at `t0_agents.guardrails.regex`, re-exported
  from `t0_agents.guardrails`.  Blocks a run when
  `re.search(pattern, ...)` matches: `phase="after_run"` (default)
  scans `AgentResult.value` (string values directly, everything
  else JSON-encoded via `json.dumps(default=str)`); `phase="before_run"`
  scans `GuardrailContext.user_message`.  Raises
  `GuardrailBlockedError(message)` on a hit, which
  `GuardrailsMiddleware` propagates as a hard abort.  Pre-compiled
  `re.Pattern` accepted in lieu of a source string.  `t0_agents.glow`
  re-exports the same callable for the
  `Annotated[T, Guardrail(...)]` typed-metadata pattern.

#### Runtime system prompt — the `system=` escape hatch

Expose TensorZero's raw-string `input.system` form on every public
inference surface as the escape hatch for prompt-as-a-service flows
where the prompt body itself is runtime data.  The existing
`system_arguments` (template-variable dict) stays the recommended
default; the new `system` kwarg is layered alongside it, with up-front
validation that catches the combinations TensorZero would reject on
the wire.  Backwards compatible: any caller that does not pass
`system=` keeps today's behaviour byte-for-byte.

- **`system: str | None = None`** kwarg on every public inference
  surface: `run_agent`, `Agent.run`, `Agent.stream`,
  `run_structured_inference`, and `StreamingAgentLoop.stream`.  When
  set, the string is forwarded as `input.system` (raw form) to
  TensorZero on every inference call.  Only valid when the configured
  function does **not** declare `schemas.system.path`; the gateway
  returns HTTP 400 otherwise.  Documented as an escape hatch — prefer
  `system_arguments` (with a configured system template) for
  production code so ClickHouse captures structured prompt variables
  for fine-tuning, A/B testing, and observability.
- **`docs/api-reference.md` § "Runtime system prompt: `system` vs
  `system_arguments`"** — decision matrix for the two kwargs across
  schema'd / schema-less functions, plus guidance on when to reach
  for each.
- **`examples/agent/02_agent_class/`** — runnable PaaS-style demo:
  the same `Agent` is invoked three times with three runtime-supplied
  personalities (haiku, pirate, lawyer), driving the model purely
  through `system="..."`.  Companion to the templated, recommended
  pattern that the same example folder demonstrates via
  `system_template`.

Up-front validation:

- `system` and `system_arguments` are mutually exclusive on every
  surface — passing both raises `ValueError` before any inference is
  issued.  Both kwargs target `input.system`, so the gateway would
  reject the mixed shape anyway; t0-agents fails fast with a clearer
  message.
- `Agent.run` / `Agent.stream` additionally reject `system=<str>` when
  `memory_strategy="system_template"` is in effect.  That strategy
  injects `memory_context` into `input.system` via the configured
  template (dict shape), which is incompatible with a raw string.
  Callers needing raw `system` alongside memory should switch to
  `memory_strategy="prepended_message"` (the default) or
  `memory_strategy="messages"`; both leave `input.system` untouched.

#### TensorZeroMemory backend and the agent testing framework

- **`TensorZeroMemory` backend**
  (`from t0_agents.memory import TensorZeroMemory`).  Reads cross-turn
  memory directly from TensorZero's `list_inferences` observability API
  instead of running a parallel store.  Ships in the base distribution —
  no new extra, no new dependency, because `tensorzero` is already
  required.  `save()` is a no-op (TZ is the writer); `load()` returns
  one `MemoryRecord` per chat inference that emitted text, oldest-first.
  Multi-tenant filtering via `tag_filter: dict[str, str]` (equality,
  AND-combined) plus `extra_filters: InferenceFilter` passthrough for
  time/AND/OR/NOT predicates.  `search()` returns `[]` by default; pass
  `experimental_search=True` to forward the query to TZ's
  `search_query_experimental` endpoint (flagged experimental in the SDK).
- **`memory_strategy="messages"`** — fourth `AgentConfig.memory_strategy`
  option, paired with the `MemoryProvider.load_history` extension
  method.  When set, the `Agent` skips the records-shaped `load()` and
  instead calls `memory.load_history(episode_id) -> list[Message] | None`
  and threads the result through `input.messages` as a TZ-native
  conversation history.  Sidesteps the unbounded summary growth that
  `prepended_message` exhibits when the backend reads its own writes
  back from the TZ log; preserves tool calls and tool results across
  turns so the model can reason about its own prior work.  Validated at
  `AgentConfig.__post_init__` — picking `"messages"` against a backend
  that does not implement `load_history` raises `ConfigurationError`.
- **`run_agent(prior_messages=...)`** — optional kwarg on the
  low-level loop (mirrored on `StreamingAgentLoop.stream`) that seeds
  the conversation with a TZ-native message history before the new user
  prompt is appended.  Default `None` preserves byte-identical
  behaviour, honouring the frozen-signature invariant of the `mcp/`
  submodule.
- **Agent testing framework** (`from t0_agents.testing import MockGateway,
  MockToolProvider, EmbeddedGatewayHelper, assert_tool_called, …`).
  Ships from the base distribution — no new extra, no new dependency.
  Three complementary primitives:
  - **`MockGateway`** — duck-typed `AsyncTensorZeroGateway` with three
    construction modes.  `script(turns)` pops a pre-scripted content
    list per `inference()` call; `canned(fn)` delegates to a callable
    receiving the full kwargs; `recording(real_gateway)` forwards to a
    real gateway and captures `(kwargs, response)` pairs that
    `.replay()` converts into a deterministic replay `MockGateway`.
    Records `.calls` (kwargs) and `.responses`
    (`ChatInferenceResponse`) in parallel so assertions can inspect
    both what the loop asked for and what the model decided.
  - **`MockToolProvider`** — dict-backed `ToolProvider` Protocol
    implementation.  Sync + async handlers; unknown tool raises
    `KeyError`; schemas configurable via the `schemas=` constructor
    kwarg.  Returns a real `fastmcp.CallToolResult` so the production
    `serialize_tool_result` handles the mock output with the exact same
    rules as a real tool.
  - **`EmbeddedGatewayHelper`** — async-context wrappers around
    `AsyncTensorZeroGateway.build_embedded` (source-validated at
    `tensorzero.pyi:1016-1035`).  `from_config(path)`,
    `with_clickhouse(path, url)`, `with_postgres(path, url)` for
    increasingly-observable integration tests that still don't need
    Docker.
- **Assertion helpers** — `assert_tool_called`, `assert_tool_not_called`,
  `assert_tool_args_equal`, `assert_max_turns`, `assert_response_contains`.
  The tool-call assertions accept exclusive `gateway=` (inspects what
  the model emitted) or `tool_provider=` (inspects what actually
  executed after guardrails + the loop).
- **Reusable scenarios** (`t0_agents.testing.scenarios`) —
  `single_tool_happy`, `tool_error_recovery`, `max_turns_exhausted`
  canned `MockGateway` factories for the three most common loop shapes.
- **Streaming mocks** (`t0_agents.testing.streaming_mocks`) —
  `fake_event_stream(events)` wraps an iterable into
  `AsyncIterator[AgentEvent]` (ideal for testing consumers of
  `Agent.stream`); `MockStreamingGateway` yields scripted `ChatChunk`s
  from `inference(stream=True)` for basic streaming round-trip tests.
- **Pytest fixtures** (`t0_agents.testing.fixtures`) — `mock_t0`,
  `mock_tool_provider`, `embedded_t0`, `agent_test_factory`.  Registered
  by adding `pytest_plugins = ["t0_agents.testing.fixtures"]` to a
  project's `conftest.py`.
- **`embedded` pytest marker** registered in `pyproject.toml` — opt-in
  via `RUN_EMBEDDED_TESTS=1` so contributor CI stays fast.
- **Shape-contract test** at
  [`tests/test_testing/test_mock_shape.py`](tests/test_testing/test_mock_shape.py)
  asserts every kwarg the agent loop passes to
  `AsyncTensorZeroGateway.inference` is accepted by `MockGateway`, that
  `MockToolProvider` satisfies the runtime-checkable `ToolProvider`
  Protocol, and that `MockToolProvider.call_tool` returns a real
  `fastmcp.CallToolResult`.  Fails fast on any future drift between
  mocks and the real surfaces.
- **Public docs** [`docs/testing.md`](docs/testing.md) and
  [`CONTRIBUTING.md`](CONTRIBUTING.md) — two-mode comparison table,
  `MockGateway` / `MockToolProvider` guides, assertion reference,
  scenarios, embedded setup, three test modes
  (unit / integration / embedded), and a skeleton for writing a new
  test with `MockGateway`.

#### Documentation

- **`docs/run.md`** — canonical user-facing guide for the `Run`
  surface: lifecycle, two-channel consumption, replay / fork
  semantics, error model, `Run(agent=None)` mode for third-party
  Composer authors, runnable snippets.
- **`docs/middleware.md`** — canonical guide for the Middleware
  chain: protocol, `RunContext` mutability, default chain ordering,
  the 6 built-ins, `middleware=[]` opt-out trade-off, custom
  authoring walkthrough, `allow_retry = True` opt-in.
- **`docs/output-modes.md`** — canonical guide for `OutputMode[T]`:
  protocol, the 5 built-ins, per-call narrowing via
  `Agent.invoke(output=...)` overloads, retry-and-recovery handshake,
  `OutputDiagnostics.failure_detail`, third-party `YamlOutput`
  walkthrough, the `[pydantic]` extra.
- **`docs/migration-v0.12-to-v1.0.md`** — migration walkthrough for
  the two v0.12 surfaces removed in v1.0 (`Agent.run` / `.stream` →
  `Agent.invoke` / `Run.events`; five JSON config kwargs →
  `output=JsonOutput(...)`) with side-by-side snippets.
- **`docs/api-reference.md`** — new "Run", "Middleware", "RunUsage",
  "Output modes", and "Composers" sections; cross-links to the
  canonical narrative pages.
- **`docs/guardrails.md`** — new section "Guardrails vs. TensorZero
  rate limits" distinguishing per-invocation guardrails from
  gateway-layer `rate_limiting.rules`.
- **`docs/production.md`** — new section "Rate limits and quotas"
  with a cross-link to the guardrails section.
- **`docs/agent.md`** preamble rewritten pointing at the four
  canonical doc pages.
- **`docs/index.md`** reading path extended with Run / Middleware /
  Output Modes / Composition between Agent and Concepts.
- **`docs/structured-outputs.md`** header cross-links to the new
  `docs/output-modes.md` canonical narrative.
- **`docs/multi-agent.md`** preview rewritten with cross-links into
  `docs/concurrency.md` and `examples/agent/12_concurrency/` for the
  orthogonal code-driven path.

#### Experimental — `t0_agents.glow`

- **`t0_agents.glow` sub-package scaffolded** on the
  `experimental/glow` branch. Purely additive Pythonic skin over v1.0
  primitives (FastAPI-style decorators, type-hint-driven OutputMode
  inference, fluent `Run` wrapper, native-collection composition,
  typed MCP). Public API is **unstable**; ships behind the
  `t0-agents[glow]` extra so the canonical v1.0 surface stays clean.
  See [`docs/glow/limitations.md`](docs/glow/limitations.md) for the
  known caveats.
- **Output-mode inference from return-type hints**
  (`t0_agents.glow.output_inference`, internal). `str` /
  `dict[*]` / `BaseModel` / `list[BaseModel]` / `Optional[*]` /
  `T | None` map to `ChatOutput`, `JsonOutput`,
  `StructuredOutput(M)`, and a synthetic-wrapper `StructuredOutput`
  respectively. Forward-ref failures raise `ConfigurationError` at
  registration time so `@app.agent` surfaces them eagerly.
- **`FluentRun[T]` lazy invocation wrapper**
  (`from t0_agents.glow import FluentRun`). Defers `OutputMode`
  selection until `__await__`; exposes `.text()`, `.json()`,
  `.json(M)`, `.as_(mode)`, `.events(*selectors)`, and a `.result()`
  escape hatch for the full `AgentResult[T]`. `.json(A).json(B)` is
  last-wins (second override on an already-overridden FluentRun logs
  DEBUG).
- **`App` + `@app.agent` decorator landed.** `from t0_agents.glow
  import App` exposes the registry+lifecycle class; `@app.agent("name",
  *, function_name="agent")` walks the function signature at decoration
  time, picks the `OutputMode` from the return annotation, uses the
  docstring as the system prompt, and registers an internal
  `_AgentSpec`.  The returned callable preserves the original
  parameters via `ParamSpec`, so `mypy`-strict callers see
  `tutor("hi")` typed as `FluentRun[str]` and `await tutor("hi")` as
  `str`.  Forward-ref failures and duplicate names raise
  `ConfigurationError` eagerly at decoration time.  The wrapper
  exposes `._agent` (None pre-build) and `._spec` for escape-hatch
  access.  `App.build()` and `app.lifespan()` are the eager
  construction hooks.
- **Event selectors.** `from t0_agents.glow import Thought,
  ToolCall, Text, ToolResult, TurnEnd, Done, Error` — seven
  module-level `Selector[E]` instances covering the v1.0 `AgentEvent`
  union.  `.where(pred)` AND-chains predicates and `.map(fn)` replaces
  the transform; both return *new* selectors so the constants stay
  immutable.  `FluentRun.events(*selectors)` filters with first-match-
  wins semantics; empty selectors preserve the yield-every-event
  contract.  `Text` defaults to `TextEndEvent` (whole-text variant)
  per the spec — token-by-token delta filtering remains reachable
  through `Selector(TextDeltaEvent).where(...)`.
- **`FluentRun(value_transform=...)`.** New keyword-only
  constructor argument that maps `AgentResult.value` through a callable
  when no per-call override is active.  `list[BaseModel]` return
  annotations unwrap the synthetic `<Inner>List` wrapper so
  `await agent(...)` returns `list[Inner]` directly, matching the
  user's annotation.  Per-call overrides
  (`.text` / `.json` / `.as_`) reset the transform via the existing
  `_spawn` path (deliberate opt-out).
- **`[glow]` pip extra.** `pip install t0-agents[glow]` materialises
  `pydantic>=2.0` (FastAPI precedent — the glow surface uses pydantic
  heavily for type introspection).
- **Two runnable examples.** `examples/glow/01_hello.py` and
  `examples/glow/02_structured.py` ship with a self-contained
  TensorZero gateway configuration under `examples/glow/config/`
  (chat function `agent` + JSON-structured function `plan_trip` with a
  schema matching `TripPlan`).  Boot with `docker compose -f
  examples/glow/docker-compose.yml --env-file .env up -d`.
- **`Depends` + `DependencyResolver`.** `from t0_agents.glow
  import Depends` provides the FastAPI-style injection sentinel
  (`provider: Callable`, `use_cache: bool = True`).  The internal
  `DependencyResolver` walks `inspect.signature(fn)`, recursively
  resolves provider chains, caches per call when `use_cache=True`,
  and consults `app.dependency_overrides[provider]` on every call
  (not snapshotted at registration).  Sync + async + sync-returns-
  awaitable providers are all supported.
- **`@app.tool` decorator.** Registers user functions as
  FastMCP tools on the App-owned in-memory server.  Parameters with a
  `Depends` default are **stripped** from the auto-generated JSON
  Schema (so the LLM never sees them) and re-injected at call time
  via the App's resolver.  Tool docstring becomes the MCP description.
  `Depends(<glow agent wrapper>)` inside `@app.tool` raises
  `ConfigurationError` — sub-agent semantics belong on `@app.agent`.
- **Sub-agents via `Depends(agent_wrapper)` on `@app.agent`.**
  The signature walk now requires every non-message parameter to
  declare `Depends(<glow agent wrapper>)`; the provider's spec is
  added to `AgentConfig.sub_agents` declaratively and the parameter
  is compiled out of the runtime call signature.  Value-style
  `Depends(get_db)` on `@app.agent` raises `ConfigurationError`
  with a pointer to `@app.tool`.  Forward refs are rejected eagerly
  (register sub-agents BEFORE their orchestrators).
- **App-owned FastMCP server + Client lifecycle.**
  `App.__init__` allocates a long-lived in-memory `FastMCP("glow-app")`
  + `Client`; every glow agent is constructed with
  `tool_provider=app._owned_client`, so glow agents never run the
  v1.0 core's toolless-chat short-circuit — they always have a real
  `Client` to call into.
  `app.build()` enters the Client once (idempotent); `app.shutdown()`
  closes it; `app.lifespan()` does `build → yield → shutdown` in a
  `try/finally`.  Examples and integration tests are updated to wrap
  their hot path in `async with app.lifespan():`.
- **`app.dependency_overrides` is now load-bearing.**  The
  override dict is consulted on every `Depends` resolution.
  `app.dependency_overrides[get_db] = mock_db` replaces the resolved
  value at call time without re-instantiating the decorator —
  FastAPI's testing-ergonomics pattern.
- **`examples/glow/05_di_tools.py`.**  Runnable demo showing
  `@app.tool` + `Depends` stripping + `dependency_overrides` over a
  direct MCP round-trip (no agent / no TZ inference).  Adds a
  `[functions.orchestrator]` chat function to
  `examples/glow/config/tensorzero.toml` used by the new integration
  test `test_orchestrator_with_glow_sub_agents`.
- **Typed MCP integration via `MCPServer`.**  `from t0_agents.glow
  import MCPServer` exports a dual-personality base class:
  - Class form (canonical): `class GithubMCP(MCPServer, url="...",
    namespace="github"): ...` records an `MCPServerSpec` on the class
    via `__init_subclass__`.
  - Annotated form: `Annotated[Any, MCPServer(url="...")]` carries
    the spec as Annotated metadata.
  `App.__init__` accepts `servers=list[type[MCPServer]] | None`;
  `@app.agent` parameter walks via `_extract_mcp_specs` append
  discovered specs to
  `app._mcp_specs`.  `App.build()` mounts every spec onto the
  long-lived `glow-app` composed FastMCP BEFORE entering the owned
  Client — `FastMCP.mount(server)` for in-memory `FastMCP` instances,
  `FastMCP.mount(create_proxy(url))` for URL-based remotes (HTTP,
  SSE, or any `ClientTransport` `create_proxy` accepts).  Per-agent
  MCP subsets are deferred — every glow agent on a given App sees
  every mounted MCP server.
- **`Annotated[T, Guardrail(...)]` extraction.**  `@app.agent`
  parameter and return annotations are walked at registration time;
  every `Annotated` metadata entry satisfying the runtime-checkable
  v1.0 `Guardrail` Protocol is harvested and folded into
  `AgentConfig.guardrails` on first build.  The runtime path
  (`GuardrailsMiddleware`) is unchanged.  `_AgentSpec` grows a
  `guardrails: list[Guardrail]` slot for the harvested set.
- **Iterative `Annotated`/`Optional` peel in `output_inference`.**
  `_peel_to_core(rt)` is an iterative `while changed:` loop that
  converges on the inner type regardless of nesting depth, so both
  `Optional[Annotated[T, m]]` and `Annotated[Optional[T], m]`
  resolve the same way — important for guardrails-on-return-type,
  which produces the mirror nesting.
- **`examples/glow/06_mcp_typing.py`.**  Runnable demo: in-memory
  `FastMCP` math server, `class MathMCP(MCPServer, server=...)`,
  `App(servers=[MathMCP])`, end-to-end verification that
  `app._owned_client.list_tools()` returns the namespaced tool
  followed by a live agent call.  Commented-out HTTP variant points
  at the `create_proxy(url)` path for remote MCP servers.
- **Public glow `__all__` settles at 13 entries.**  `App`,
  `Depends`, `Done`, `Error`, `FluentRun`, `MCPServer`, `Selector`,
  `Text`, `Thought`, `ToolCall`, `ToolResult`, `TurnEnd`,
  `regex_guardrail`.
- **`examples/glow/03_streaming.py`.**  Runnable demo of
  `FluentRun.events(*selectors)`: a chat agent + a `@app.tool`
  weather lookup + a live `async for` loop driving three
  selectors at once (`Thought`, `ToolCall.map(...)`, `Text`).
  Reuses the existing `[functions.agent]` chat function — no new
  TZ config.
- **`docs/glow/getting-started.md`.**  Canonical user-facing
  tutorial covering every public glow symbol plus the load-bearing
  callouts: toolless-chat, sub-agents via `Depends(agent)`, typed
  MCP, `Annotated[T, Guardrail(...)]`, and the
  `app.lifespan()`-as-entry-point recommendation.
- **`docs/glow/limitations.md`.**  Known limitations of the
  glow surface: no streaming JSON, no `AsyncIterator` tools, no
  per-agent MCP subsets, the `[glow]` extra's hard pydantic dep,
  and the `_AgentWrapper.__call__` ParamSpec dual-personality.
- **Top-level `README.md` points at the glow tutorial.**  A
  small "Experimental: `t0_agents.glow`" section sits between the
  Documentation reading-path and the Requirements block; flags
  unstable status; links to `docs/glow/getting-started.md`; mentions
  the `[glow]` install extra.

### Changed

- **Breaking — `mcp_client` → `tool_provider` rename on low-level
  entry points.**  `run_agent` (in `t0_agents.mcp.loop`) and
  `StreamingAgentLoop.stream` (in `t0_agents.stream.runner`) rename
  their second positional parameter from `mcp_client` to
  `tool_provider`; the type widens to `Client[Any] | None` and the
  default is `None`.  Positional callers (the vast majority) keep
  working unchanged; keyword callers must update
  `mcp_client=client` → `tool_provider=client`.  Both `run_agent`
  and the streaming runner skip `list_tools()` when
  `tool_provider is None`, advertise an empty tool list to
  TensorZero, and return after a single inference.  If the LLM
  emits a `ToolCall` while `tool_provider` is `None`, the loop
  raises `T0AgentsFatalError` rather than stringifying the policy
  violation into a `ToolResult`.  The low-level helpers
  `execute_tool_calls` and `execute_tool_calls_parallel` rename
  the same parameter in lock-step but stay non-optional — they
  only run inside the tool-dispatch branch, after the optional
  guard has already proven a provider is present.  v1.0 is the
  first stable release, so no deprecation shim for the prior
  `mcp_client` name ships.

- **`Agent.__init__` accepts `tool_provider=None` for chat-mode
  agents — no MCP allocation behind the scenes.** v1.0 originally
  rejected chat configs (`output=ChatOutput()`, the default)
  constructed without a tool provider; that rule is relaxed so
  `Agent(config, t0=t0)` works for every output mode.  When
  chat-mode is selected and the caller passes `None`, the chat
  terminals forward `None` directly to `run_agent` /
  `StreamingAgentLoop.stream`; the loop skips `list_tools()`,
  advertises an empty tool list, runs a single inference, and
  returns the text response.  No FastMCP server is allocated;
  no `Client` lifecycle is implied.  If the LLM ever emits a
  `ToolCall` under these conditions, `T0AgentsFatalError`
  surfaces the policy violation instead of being stringified into
  a tool error.  Single-shot output modes (`JsonOutput`,
  `StructuredOutput`, `ToolCallOutput`) were already tolerant of
  `tool_provider=None`.  The `mcp/` standalone-submodule invariant
  is preserved — `run_agent` and `StreamingAgentLoop.stream`
  share a `Client[Any] | None`-typed `tool_provider` parameter and
  do not import any higher-layer protocol.

- **`AgentConfig.sub_agents=[...]` no longer requires a FastMCP
  server.**  `Agent.__init__` auto-allocates a
  `SubAgentToolProvider` (new public class in
  `t0_agents.delegation`) that synthesises one
  `mcp.types.Tool` per sub-agent and dispatches matching
  `call_tool` invocations inline through `Agent.invoke()`.  Pass
  `tool_provider=None` when the orchestrator has no MCP tools of
  its own; pass a real `Client` (or any `ToolProvider`) and it
  composes as `inner` — `list_tools()` returns the inner tools
  followed by the synthetic `{name}_ask` entries, and unknown
  names fall through to `inner.call_tool`.  This collapses the
  pre-existing dual-vía (server-side `install_sub_agents`
  registration + middleware-side `_OrchestrationToolProvider`
  wrap) into a single object the loop consumes directly.  Pure
  sub-agent orchestrators now pay zero FastMCP cost.

- **`glow.App` lazily allocates `_mcp_server` / `_owned_client`.**
  The pair was constructed in `App.__init__` unconditionally;
  apps with no `@app.tool` registrations and no `MCPServer` specs
  paid for a FastMCP server they never used.  Allocation now
  happens on first use via `_ensure_mcp_server()` — triggered by
  `@app.tool` decoration or `build()` when `_mcp_specs` is
  non-empty.  Sub-agents-only apps leave both `None` and pass
  `tool_provider=None` to `Agent`, which auto-allocates the
  `SubAgentToolProvider` for the delegation surface.  Existing
  apps with `@app.tool` or `MCPServer` specs see no behavioural
  change.

- **`SubAgentMiddleware` is now ContextVar-only.**  The middleware
  no longer wraps `ctx.tool_provider` — dispatch lives in
  `SubAgentToolProvider`, which `Agent.__init__` pre-installs on
  `self._tool_provider`.  What remains is the
  `parent_episode_id` ContextVar plumbing (initial bind +
  `on_inference` refresh hook).  The default middleware chain
  position (`Memory → SubAgents → Guardrails → CostTracking →
  JsonRetry`) is unchanged; `SubAgents` precedes `Guardrails` so
  the parent-episode binding is in scope when guardrails fire.

- **`Agent` internals** refactored to a middleware pipeline.
  `core/agent.py` shrinks from ~1670 LoC (v0.12) to ~870 LoC.
  `_execute_chat_legacy`, `_execute_chat_streaming_legacy`, and
  `_execute_json_legacy` (≈1100 LoC inline orchestration) are
  replaced by `_drive(ctx, is_streaming, middleware)` + three slim
  terminals.  `Agent.invoke` docstring slimmed from ~140 LoC to
  ~11 LoC (full surface moves to `docs/run.md`).
- **`AgentResult` becomes `AgentResult[T]` generic** with
  `value: T` as the canonical field.  Read-side properties
  preserved.  `response=` is accepted as a constructor alias for
  `value=` because `mcp/loop.py` (invariant §2.1) constructs the
  chat result with that kwarg.
- **`Agent.invoke(..., output=...)` per-call overload** — when a
  per-call `output: OutputMode[T]` is supplied, `Run[T]` is narrowed
  by mypy via overload pair.  Default (no per-call override) returns
  `Run[Any]`.
- **`JsonRetryMiddleware`** — consults `ctx.output.recover(failure,
  attempts)` between attempts; mutates `ctx.user_message` and
  `ctx.prior_messages` to thread the corrective hint into the next
  inference.  `allow_retry = True` class attribute signals
  `build_pipeline` to skip the `_NextGuard`.
- **`Run.cost()` / `Run.usage()`** populated by
  `CostTrackingMiddleware` on every inference.  v0.12 only tracked
  cost when guardrails were configured; v1.0 makes it always-on
  unless the caller overrides with `middleware=[]`.
- **`Run.__init__`** widens `agent: Agent` to `agent: Agent | None`
  (one-line widening + weakref guard).  Required so `CompositeRun`
  can satisfy the `Run` nominal type without a synthetic Agent
  stub.
- **HTTP / WS serialization of `DoneEvent`** explicitly projects
  through the v0.12 read-side keys (`response`, `parsed`,
  `parse_strategy`, `json_parse_attempts`, `function_kind`) so
  downstream clients keep working byte-identical.
- **`docs/structured-outputs.md`** updated for the v1.0 surface
  with explicit v0.12 → v1.0 migration block.

### Removed

- **`compose` topology primitives** — the 9 factories
  (`sequential`, `parallel`, `race`, `conditional`, `consensus`,
  `map_reduce`, `saga`, `retry_chain`, `circuit_breaker`) and the
  `Composable` / `Composer` / `CompositeRun` / `ComposerError`
  protocol layer.  Use `await agent.invoke(...)` for sequential
  chaining, `asyncio.gather` for parallel execution, and
  `AgentConfig(sub_agents=[...])` with a routing system prompt for
  the dispatch case.  See
  `examples/agent/12_concurrency/{01_sequential,02_parallel,03_orchestrator}.py`
  and `examples/glow/{04_sequential,05_parallel,06_orchestrator}.py`.
- **`t0_agents.glow.compose`** — removed alongside the underlying
  topology layer.
- **Sub-agent orchestration relocated** — `agent_as_tool`,
  `install_sub_agents`, and friends moved to `t0_agents.delegation`
  before being deleted (see next entry).
- **`install_sub_agents`, `agent_as_tool`, `_OrchestrationToolProvider`**
  — three internal helpers that wired sub-agents into a server-side
  FastMCP registration plus a middleware-side dispatch wrap.
  `SubAgentToolProvider` (see `Changed`) replaces all three with a
  single class.  `t0_agents.agent_as_tool` and
  `t0_agents.delegation.install_sub_agents` are no longer
  importable; callers using the declarative
  `AgentConfig(sub_agents=[...])` path need no change, callers using
  the internal helpers directly should switch to `SubAgentToolProvider`.

- **`Agent.run(...)`** — replaced by `await agent.invoke(msg, ...)`
  (a `Run` is awaitable and returns the same `AgentResult`).
  Migration is a one-liner per call site; see
  [`docs/migration-v0.12-to-v1.0.md`](docs/migration-v0.12-to-v1.0.md).
- **`Agent.stream(...)`** — replaced by `async for ev in
  agent.invoke(msg, ...).events()`.  Two-channel consumption is now
  supported — see [`docs/run.md`](docs/run.md).
- **`AgentConfig.function_kind`** — replaced by
  `AgentConfig.output: OutputMode`.  Passing it raises `TypeError`.
- **`AgentConfig.json_mode`** — replaced by
  `JsonOutput(json_mode=...)`.  Passing it raises `TypeError`.
- **`AgentConfig.output_schema`** — replaced by
  `JsonOutput(schema=...)` (callers load the JSON themselves and
  pass a `dict`).  Passing it raises `TypeError`.
- **`AgentConfig.permissive_json_parsing`** — replaced by
  `JsonOutput(permissive=...)`.  Passing it raises `TypeError`.
- **`AgentConfig.max_json_retries`** — replaced by
  `JsonOutput(max_retries=...)`.  Passing it raises `TypeError`.

  Migration: `AgentConfig(function_kind="json", json_mode="strict",
  output_schema=S, permissive_json_parsing=True, max_json_retries=2)`
  → `AgentConfig(output=JsonOutput(schema=S, json_mode="strict",
  permissive=True, max_retries=2))`.
- **`Agent.invoke(..., output_schema=, json_mode=, max_json_retries=)`**
  per-call kwargs — pass a per-call `output=JsonOutput(...)` instead.
- **`RunInput.function_kind/output_schema/json_mode/max_json_retries`**
  — dispatch routes via `Run.output.expects_tool_loop`.
- **`Agent._RETRY_HINT_BASE` + `Agent._build_retry_hint`** — moved
  into `JsonOutput` as a class attribute + instance method.
- **`StructuredOutput`** (the dataclass result of
  `run_structured_inference`) — **renamed to
  `StructuredInferenceResult`** to free the canonical name for the
  new Pydantic-driven `OutputMode`.  Direct callers of
  `run_structured_inference` that imported the dataclass name
  should switch to `StructuredInferenceResult`.

### Fixed

- **7 broken ClickHouse image tags** —
  `examples/{02,03,04,05,06,07,18}/docker-compose.yml` pinned the
  non-existent `clickhouse/clickhouse-server:lts`; replaced with
  the canonical `clickhouse:lts`.
- **`examples/agent/01_first_agent/README.md` directory drift** — README
  still referenced the directory by its old name `01_basic_agent`;
  replaced and stale forward-reference removed.
- **429 regression test** — `tests/test_mcp/test_harness.py::
  TestRecoverableInferenceError::test_429_rate_limit_re_raised`
  locks in the "propagate, never auto-retry" contract.  Closes the
  parked rate-limit follow-up.
- **Reasoning models that emit content via the summary channel no longer
  yield empty responses.** ``ThoughtChunk.summary_text`` (streaming) and
  ``Thought.summary[*].text`` (non-streaming) — populated by OpenAI
  o-series reasoning summaries and harmony-format providers in some
  configurations — were previously silently dropped because the
  assembler and the ``mcp/loop.py`` thought-fallback only read the
  ``text`` field.  Symptom: ``DoneEvent.result.response == ""`` and
  no ``ThoughtDeltaEvent``s emitted for affected providers.
  - The streaming assembler now folds ``text + summary_text`` into a
    single ``ThoughtDeltaEvent`` delta.  When a chunk carries both, a
    ``"\n"`` is interposed so the streamed content matches the
    non-streaming path's ``"\n".join(...)`` of the same logical entries.
  - The non-streaming Thought-fallback now mines
    ``Thought.summary[*].text`` alongside ``Thought.text``.
  - No public API or wire-format change; consumers that already render
    ``thought_delta`` events automatically receive the recovered
    content, and ``AgentResult.response_source`` correctly reports
    ``"thought"`` when only summary content is present.

### Notes

- **Industrial middlewares deferred** — `BulkheadMiddleware`,
  `CircuitBreakerMiddleware`, `TenantIsolationMiddleware`,
  `PiiRedactorMiddleware`, `RedisRateLimitMiddleware`, and an
  `OTelTracerAdapter` were considered for the v1.0 release and
  dropped after a scope review (no concrete consumer asks for them
  today).

### Release verification

- **`examples/SMOKE.md`** — release-gate manual checklist (not CI).
  Twenty-four example scripts plus four manual UI checks, organised
  into one offline batch and two docker batches that share composes
  across multiple examples.  Two known model-side flakes
  (`test_require_approval_auto_decline_surfaces_to_llm`,
  `test_tool_timeout_does_not_crash_run_agent`) are documented with
  reroll guidance and a non-Groq model-swap escape hatch.

## [0.12.0] — 2026-05-15

Relax the `Agent` constructor contract so JSON-mode agents
(`function_kind="json"`) can be built without a `tool_provider`.  JSON
functions are single-shot and never dispatch MCP tools, so requiring an
empty `fastmcp.Client` at construction time was friction with no
upside.  Chat-mode agents still require a `tool_provider`; the
constructor now rejects that case eagerly with a friendly message.
Backwards compatible: every existing call signature (chat or JSON
passing an explicit provider) continues to work byte-identically.

### Changed

- **`Agent.__init__` accepts `tool_provider: ToolProvider | None = None`.**
  Constructing a JSON-mode agent (`AgentConfig(function_kind="json", …)`)
  no longer requires passing a provider — `Agent(config, t0=t0)` is now
  enough.  Chat-mode agents (`function_kind="chat"`, the default) raise
  `ConfigurationError` if no provider is supplied, with a message
  pointing at the two valid options (pass a provider, or set
  `function_kind="json"`).
- **`Agent._run_json` safety check tolerates a missing provider.**  The
  defensive `list_tools()` round-trip that runs when `json_mode="tool"`
  (TZ guide §11.1 — tool-mode forbids advertised tools) is now a vacuous
  pass when `self._tool_provider is None`: no provider means no
  advertised tools, so the invariant holds without an MCP call.
- **`serve.create_app(..., share_client=True)` rejects provider-less
  agents.**  When the agent was built without a `tool_provider`, there
  is nothing to share — `create_app` now raises `ConfigurationError`
  early instead of letting `SessionFactory`'s XOR fail with a less
  helpful message.  Use `client_factory=...` / `session_provider=...`
  instead, or construct the agent with an explicit `tool_provider`.

### Examples and docs

- `examples/03_output_modes/structured_agent.py`,
  `ner_dynamic_schema.py`, `recovery_real_provider.py`, and
  `recovery_demo.py` no longer build a placeholder `FastMCP("noop")`
  server just to satisfy the constructor; they pass only the gateway.
  Imports of `fastmcp.Client` / `fastmcp.FastMCP` drop out of every
  JSON-mode example.
- `docs/structured-outputs.md` — the canonical JSON snippet builds the
  agent with just `Agent(config, t0=t0)`; the prose now explains that a
  `tool_provider` is optional for JSON-mode and required for chat-mode.
- `docs/agent.md` — composition table and signature reflect the
  optional kwarg; a short JSON-mode snippet sits alongside the chat
  lifecycle example.
- `docs/serving.md` — the `create_app` modes table calls out the
  provider-less guard explicitly.

## [0.11.0] — 2026-05-14

First-class support for TensorZero **JSON functions** in t0-agents,
while preserving every v0.9.0 invariant for chat callers.  Adds
`function_kind`, `output_schema`, `json_mode`,
`permissive_json_parsing`, and `max_json_retries` on `AgentConfig`;
`parsed`, `parse_strategy`, `json_parse_attempts`, and `function_kind`
on `AgentResult`; per-call kwargs on `Agent.run`; the opt-in
`require_parsed()` guardrail; the `MockGateway.script_json` /
`canned_json` testing primitives and five JSON-aware assertion helpers.
Chat-path behaviour is byte-identical to v0.9.0; `src/t0_agents/mcp/`
is bit-for-bit untouched.

### Added

#### Core (configuration and dispatch)

- **`run_structured_inference()` and `StructuredOutput` dataclass**
  (`from t0_agents.core.structured import run_structured_inference,
  StructuredOutput`).  Single-shot helper for TensorZero JSON functions
  — the sibling of `run_agent` for `type = "json"` functions.  Accepts
  per-call `output_schema` (dynamic schema override) and `json_mode`
  (`off`/`on`/`strict`/`tool`), threads through `system_arguments`,
  `episode_id`, `tags`, `namespace`, and `dryrun`, and fires an
  optional `on_inference` callback that mirrors `run_agent`'s.
- **`AgentConfig` fields for JSON functions**: `function_kind`
  (`Literal["chat", "json"]`, default `"chat"`), `output_schema`
  (`Mapping[str, Any] | Path | None`), `json_mode`
  (`Literal["off", "on", "strict", "tool"] | None`),
  `permissive_json_parsing` (`bool`, default `True`),
  `max_json_retries` (`int`, default `2`).  `__post_init__` enforces
  the five validation cases (JSON requires `json_mode`, chat forbids
  `json_mode` and `output_schema`, JSON forbids `sub_agents`, JSON
  forbids `memory_strategy="messages"`, `max_json_retries >= 0`).
  `output_schema: Path` is read as JSON at construction time; relative
  paths resolve against `cwd` (absolute paths recommended).
- **`AgentResult` fields**: `function_kind` (`Literal["chat", "json"]`,
  default `"chat"`), `parsed` (`Any | None`, default `None`),
  `parse_strategy` (`Literal["strict","permissive","retry","none"]`),
  `json_parse_attempts` (`int`, default `0`).  Defaults preserve
  byte-identity for chat-function callers.
- **`Agent.run` JSON dispatch.**  When `config.function_kind == "json"`,
  the method routes to `run_structured_inference` instead of the MCP
  tool loop and returns an `AgentResult` with `function_kind="json"`,
  `parsed` populated, `response` set to the raw JSON string, and the
  retry-aware `turns` / `inference_ids` semantics described below.
  The chat path is byte-identical to v0.9.0 for callers that do not
  pass the new kwargs.
- **Per-call `output_schema`, `json_mode`, and `max_json_retries`
  kwargs on `Agent.run`.**  Each overrides the corresponding
  `AgentConfig` field for a single call.  Passing them on a
  chat-flavoured agent raises `ConfigurationError`.
- **Run-level guardrails fire on JSON runs.**  `Guardrail.before_run`
  and `Guardrail.after_run` execute exactly as on the chat path —
  same fatal/non-fatal contract, same `guardrail_log` shape.  Tool
  hooks (`before_tool`/`after_tool`) never fire because JSON
  functions are single-shot and call no tools.
- **Memory load/save on the JSON path.**  `memory_strategy` values
  `"prepended_message"` (the default) and `"system_template"` work
  unchanged.  `MemoryRecord.agent_response` carries the raw JSON
  string of the final attempt; `inference_ids` accumulates every
  retry attempt's id in order; `turns` equals `1 + json_parse_attempts`.
  `memory_strategy="messages"` is rejected at runtime with
  `ConfigurationError` because the helper reserves `prior_messages`
  for the retry orchestration.

#### Robust parsing and retry

- **`permissive_parse(raw)`** (`t0_agents.permissive_parse`,
  `t0_agents.core.structured.permissive_parse`) — pure-function utility
  that tries a three-layer recovery ladder on a raw model string
  (`json.loads(raw)` → balanced `{…}`/`[…]` scan → fenced code block)
  and returns `(value, strategy)` where `strategy ∈ {"strict",
  "permissive", "none"}`.  Never raises.  Supports both top-level
  objects and top-level arrays; an internal earliest-opener
  disambiguator picks the right span when both are present.
- **`run_structured_inference(prior_messages=...)`** — optional list
  of `{role, content}` dicts that the helper prepends before the
  `user_message` turn.  Reserved for the retry orchestration; direct
  callers should leave it at `None`.
- **Bounded retry orchestration in `Agent._run_json`.**  When
  TensorZero's strict parser returns `parsed == None`, the orchestration
  consults `permissive_parse(output.raw)` (when the layer is enabled);
  when that also fails and the retry budget is non-zero, it re-emits
  the inference with the previous (invalid) assistant turn and a
  corrective user-turn hint.  The first response's `episode_id` is
  captured and threaded into every subsequent attempt so retries stay
  in the same TensorZero episode.

#### Safety and provider quirks

- **`require_parsed()`** (`t0_agents.guardrails.require_parsed`) —
  opt-in guardrail whose `after_run` raises
  `GuardrailBlockedError` when a JSON run completes with
  `result.parsed is None`.  No-op on chat runs (`parsed` is
  structurally absent on the chat path).  Lives in
  `src/t0_agents/guardrails/structured.py`.
- **JSON `Agent.stream(...)` guard.**  Iterating
  `agent.stream(...)` on an agent with `function_kind == "json"`
  raises `ConfigurationError` at the first `__anext__` with a
  message pointing at `Agent.run(...)`.  TensorZero does not emit
  `output.parsed` mid-stream (guide §7.1, §11.3), so surfacing a
  stream of `parsed=None` events would be a worse contract than
  refusing.  Chat streams are byte-identical to v0.9.0.
- **`json_mode="tool"` + tool-provider conflict check.**
  `AgentConfig(function_kind="json", json_mode="tool")` combined
  with a `tool_provider` that advertises tools raises
  `ConfigurationError` on the first `Agent.run(...)` (TZ guide
  §11.1 — `"tool"` mode cannot share a request with other tools).
  The check is async, so it runs once at the top of `_run_json` and
  is cached on the Agent instance.  Per-call `json_mode` overrides
  re-trigger the check when the effective mode resolves to `"tool"`.
- **Anthropic extended-thinking warning.**  A `WARNING`-level log
  is emitted on the `t0_agents.core.agent` logger the first time
  an agent runs with an effective `json_mode in {"on", "tool"}`.
  Once per Agent instance.  TZ guide §11.4 documents the
  incompatibility; the SDK does not expose provider routing, so the
  warning is generic.

#### Testing helpers

- **`MockGateway.script_json(turns)` and `MockGateway.canned_json(fn)`**
  — JSON-flavoured siblings of the existing `script` / `canned`
  classmethods.  Each script entry is either a dict (recognised keys:
  `raw`, `parsed`, `episode_id`, `inference_id`, `usage`,
  `variant_name`, `finish_reason`) or a pre-built
  `JsonInferenceResponse`.  When a dict omits `parsed`, the gateway
  best-effort-parses `raw`; an explicit `parsed: None` is preserved
  verbatim for retry-script semantics.
- **`MockJsonResponse` builder** — symmetric counterpart to
  `MockChatResponse` for tests that want explicit `to_response()`
  control over the materialised `JsonInferenceResponse`.
- **Five JSON-aware assertion helpers** in `t0_agents.testing`:
  `assert_parsed_equals`, `assert_parsed_matches_schema` (soft-imports
  `jsonschema`; raises `ImportError` with install hint when missing —
  not a hard dependency), `assert_invalid_json`,
  `assert_parse_strategy`, `assert_retry_attempts`.

#### Examples and docs

- **`examples/03_output_modes/`** — five runnable demos:
  - `main.py` — `run_structured_inference()` against the
    `extract_customer_info` function.
  - `structured_agent.py` — same function through `Agent.run`.
  - `recovery_demo.py` *(no docker)* — local fake gateway scripting
    all four `parse_strategy` outcomes in under a second.
  - `recovery_real_provider.py` *(docker)* — real-provider exercise
    of the recovery ladder with `json_mode="off"`.
  - `ner_dynamic_schema.py` *(docker)* — NER agent against the new
    `extract_entities` function showing per-call `output_schema`
    overrides and a per-call `json_mode = "tool"` swap.
- **`config/functions/extract_entities/`** and the mirrored
  `examples/03_output_modes/config/functions/extract_entities/` —
  four-category NER schema (people, locations, organizations, topics)
  and system template, declared in both the repo-root and example-folder
  `config/tensorzero.toml` under `[functions.extract_entities]`.
- **`docs/structured-outputs.md`** — new public page covering the
  helper, the configuration shape, the `Agent.run` dispatch, the
  robust-parsing layer with the three-layer recovery diagram, the
  safety surface (configuration errors, why JSON cannot stream, the
  Anthropic warning, undetectable provider quirks, `require_parsed`),
  the testing primitives, the Pydantic recipe, the `json_mode` provider
  matrix, and the See-also footer linking to every adjacent doc.
- **Cross-doc links**: `docs/concepts.md` (chat-vs-json primer),
  `docs/agent.md` (rows for every new `AgentConfig` and `AgentResult`
  field, plus the per-call kwargs on `Agent.run`), `docs/guardrails.md`
  (`require_parsed` reference), `docs/testing.md` (Testing JSON agents
  + JSON assertions sections), `docs/streaming.md` (JSON-not-supported
  note), `docs/memory.md` (JSON `agent_response` semantics),
  `docs/index.md` (module table + Appendices link).
- **Public re-exports**: `from t0_agents import StructuredOutput`,
  `from t0_agents import permissive_parse`.

### Changed

- **`AgentResult.turns` and `AgentResult.inference_ids` for JSON runs**
  reflect every attempt: `turns == 1 + json_parse_attempts` and
  `len(inference_ids) == 1 + json_parse_attempts` (always equal).
- **Retry-hint phrasing**: the corrective hint sent on every retry
  starts with a fixed base sentence and, when an `output_schema`
  exposes them, appends the schema's top-level `title` and `required`
  field names.  Hard-coded English; an i18n / per-call override hook
  is deferred.
- **`Agent.run` / `Agent.stream` / `Agent._run_json` realign
  `after_run` exception handling with the documented contract.**
  Per `docs/guardrails.md` ("A custom hook that needs to abort
  should raise a subclass of `T0AgentsFatalError`, which bypasses
  the non-fatal trap"), fatal hooks raised from `after_run` must
  propagate past the broad `except Exception` belt-and-braces trap.
  Until this release the three `after_run` loops swallowed every
  exception, silently neutering the documented hard-abort path.
  `except T0AgentsFatalError: raise` now sits ahead of the existing
  `except Exception` at each site, mirroring the `before_run`
  precedent.  Required for `require_parsed()` to actually abort a
  run.
- **`MockGateway.inference()` return type widened to
  `ChatInferenceResponse | JsonInferenceResponse`** (the SDK's
  `InferenceResponse` union).  Chat-only assertion helpers
  (`assert_tool_called` / `assert_tool_not_called` / etc.) now skip
  JSON responses when walking `gateway.responses`; JSON responses
  carry no `ToolCall` blocks by contract.

### Compatibility

- **Chat path byte-identical to v0.9.0.**  `function_kind == "chat"`
  agents see no behavioural change: every new `AgentConfig` and
  `AgentResult` field defaults to a value that matches the v0.9.0
  shape, and every chat dispatch path short-circuits before the JSON
  surfaces.  Existing tests under `tests/test_mcp/`, `tests/test_core/`,
  `tests/test_stream/`, `tests/test_serve/`, `tests/test_compose/`,
  `tests/test_guardrails/`, `tests/test_memory/`, `tests/test_cli/`,
  and `tests/test_testing/` stay green unchanged.
- **TensorZero-pure mode.**
  `AgentConfig(..., permissive_json_parsing=False, max_json_retries=0)`
  reproduces the byte-for-byte behaviour of the raw JSON dispatch
  without the recovery layers: an invalid TZ response surfaces as
  `parsed is None` and `parse_strategy == "none"` on the first attempt
  with no in-Python fallback and no retries.
- **`mcp/` byte-identical to v0.9.0.**  No source changes.
- **`after_run` realignment is observationally compatible for every
  existing guardrail.**  None of the built-in guardrails
  (`require_approval`, `max_cost`, `max_tool_calls`, `block_tools`)
  raise fatal errors from `after_run`; user-supplied guardrails that
  depend on the old swallow-everything behaviour can replace the
  hard-raise with a `logger.error` in their own implementation to
  retain the pre-realignment semantics.

## [0.8.0] — Unreleased

### Added

- **Multi-agent composition** (`from t0_agents import agent_as_tool,
  SubAgentError`; `from t0_agents.compose import install_sub_agents,
  assert_unique_names, default_tool_name`).  No new dependencies —
  built on top of the existing `core/` + `mcp/` surfaces and
  FastMCP's public runtime-registration APIs
  (`Tool.from_function(...)` + `server.add_tool(tool)`).
- **`AgentConfig.sub_agents: list[Agent]`** — declarative
  delegation.  When non-empty, `Agent.__init__` registers each entry
  as a `{sub_agent.config.name}_ask` tool on the orchestrator's
  in-process FastMCP server via
  `t0_agents.compose.install_sub_agents`.  Empty (the default)
  preserves single-agent byte-identity.  Requires a `fastmcp.Client`
  backed by an in-process `FastMCPTransport`; remote MCP transports
  raise `ConfigurationError` at construction time, as do duplicate
  sub-agent names and collisions with pre-existing tools.
- **`SubAgentError`** (`from t0_agents import SubAgentError`) —
  non-fatal wrapper for errors raised inside a delegated sub-agent.
  The generated tool handler in `t0_agents.compose.delegation`
  re-raises `T0AgentsFatalError` subclasses unchanged (so a
  `GuardrailBlockedError` in a sub-agent reaches the orchestrator's
  caller with its original type) and wraps everything else in
  `SubAgentError` with the original exception preserved as
  `__cause__`, so the orchestrator LLM sees a predictable tool
  error it can decide how to handle.
- **`t0_agents.compose.bind_parent_episode_id`** +
  **`get_parent_episode_id`** +
  **`update_parent_episode_id`** — `ContextVar`-based plumbing used
  by `Agent.run` / `Agent.stream` to surface the orchestrator's
  episode id to sub-agent tool handlers.  Sub-agents tag every
  inference with
  `t0_agents.parent_episode_id=<orchestrator-episode>` so ClickHouse
  can reconstruct the delegation tree.  For runs that start a fresh
  episode (``episode_id=None``), the orchestrator refreshes the
  binding as soon as TensorZero returns the allocated id on the
  first inference, so sub-agents invoked later in the run carry the
  real parent episode id instead of the `"unknown"` fallback.
- **Public doc** [`docs/multi-agent.md`](docs/multi-agent.md)
  covering the why, the public API, the error model, episode
  propagation, and the in-process / stable-schema caveats.
- **Example** [`examples/08_multi_agent/`](examples/08_multi_agent/)
  ships a researcher + writer + orchestrator trio with its own
  `docker-compose.yml`, `config/tensorzero.toml` (three functions:
  `research`, `writing`, `orchestrator`), and both a `main.py`
  entrypoint and an `orchestrator_factory` for
  `t0-agents run examples.08_multi_agent.agents:orchestrator_factory`.

## [0.7.0] — Unreleased

### Added

- **CLI** (`from t0_agents.cli import TUIRenderer, rich_elicitation_handler,
  make_rich_elicitation_handler, load_agent_factory`) — Typer + Rich
  terminal UI. Ships behind the new `[cli]` extra
  (`pip install 't0-agents[cli]'`), which pulls `typer>=0.12.0`,
  `rich>=13.7.0`, plus the streaming + serving deps (`fastapi`,
  `uvicorn[standard]`, `pydantic`, `sse-starlette`).
- **`t0-agents` entry-point script** wired to `t0_agents.cli.app:main`.
  Exposes three subcommands:
  - `t0-agents run <module:attr>` — interactive streaming REPL with
    live deltas, tool-call spinners, and Rich-rendered elicitation
    prompts. `--verbose` surfaces reasoning content.
  - `t0-agents serve <module:attr>` — wraps
    `t0_agents.serve.create_app(..., share_client=True)` in uvicorn.
    Non-streaming routes by default (`POST /chat`, `/feedback`,
    `/health`, `/info`); opt in to SSE + WS with `--streaming`. WS
    elicitation is incompatible with shared-Client mode (FastMCP Q3),
    so `--streaming` is only useful for tools that never call
    `ctx.elicit` — for full browser elicitation, call `create_app`
    directly with a `client_factory` + `elicitation_client_factory`
    (see `examples/07_streaming/agent_app.py`). `--host`,
    `--port`, `--deployment` flags.
  - `t0-agents info <module:attr>` — prints a Rich `Table` of the
    loaded `AgentConfig`.
- **Agent factory convention** — every subcommand's ``<module:attr>``
  argument must resolve to a zero-argument callable returning an async
  context manager yielding an :class:`Agent` (idiomatically
  `@asynccontextmanager async def assistant(): ...`). The loader raises
  `typer.BadParameter` for bare `Agent` instances with an inline snippet
  pointing users at the factory pattern, and for malformed specs /
  missing modules / missing attributes.
- **`TUIRenderer`** — Rich `Live` + `Panel` renderer that folds
  `AsyncIterator[AgentEvent]` into a live terminal frame. Exposes
  `pause()` / `resume()` so an elicitation handler can stop Live while
  it runs a prompt, and `last_episode_id` / `last_result` so the REPL
  can thread continuity across turns.
- **`rich_elicitation_handler`** + **`make_rich_elicitation_handler`** —
  FastMCP-compatible elicitation handler rendered through Rich prompts.
  Dispatches on the FastMCP Q5 schema subset: primitives + enum + const,
  description and default preserved, scalar-wrap (`{"value": ...}`)
  supported. Coerces enum answers back to the original Python type. Fails
  soft to `Prompt.ask(str)` for unknown types (non-FastMCP servers).
  Returns `ElicitResult(action="cancel", content=None)` on
  `KeyboardInterrupt`.
- **`load_agent_factory`** (`t0_agents.cli.loader`) — public resolver
  that distinguishes "malformed spec", "missing module", "missing
  attribute", "resolved to a bare Agent instance", and "resolved to a
  non-callable" so the error path is actionable.
- Example `examples/10_cli/` — runnable demo with a FastMCP server
  (`add`, `set_reminder` with `ctx.elicit(response_type=Reminder)`),
  a `@asynccontextmanager` factory, and a README walkthrough of the
  three subcommands.
- Public documentation at `docs/cli.md` covering install, the three
  commands, the factory convention, custom renderers, elicitation
  semantics, and troubleshooting.

### Changed

- This release does not modify `core/`, `mcp/`, `stream/`, `serve/`,
  `memory/`, or `guardrails/`. `run_agent`, `AgentResult`, `Agent`,
  and every existing public symbol are byte-identical to 0.6.0.
  Pre-existing offline tests remain green (363 passed total).

## [0.6.0] — Unreleased

### Added

- **Streaming primitives** (`from t0_agents import AgentEvent, DoneEvent,
  TextDeltaEvent, TextEndEvent, ToolCallDeltaEvent, ToolCallEvent,
  ToolResultEvent, ThoughtDeltaEvent, ThoughtEvent, TurnEndEvent,
  ErrorEvent, StreamingAgentLoop, TurnStreamAssembler`) — typed event
  union emitted by `Agent.stream()`.  Ships from the base distribution
  (no extra needed).
- **`Agent.stream(message, *, episode_id=None) -> AsyncIterator[AgentEvent]`**
  — streaming analogue of `Agent.run`.  Same memory / guardrail
  scaffolding; the terminal event is always `DoneEvent(result=...)`
  carrying a full `AgentResult`.  Non-streaming `Agent.run` is
  byte-identical to the previous release.
- **`TurnStreamAssembler`** — folds `AsyncIterator[ChatChunk]` from
  `AsyncTensorZeroGateway.inference(stream=True)` into an
  `AsyncIterator[AgentEvent]` with stable correlation ids per T0 Q1.
- **`StreamingAgentLoop`** — streaming analogue of `mcp.loop.run_agent`;
  reuses `convert_tools`, `execute_tool_calls[_parallel]` and the
  recoverable-retry contract from `mcp/`.  `run_agent` and the `mcp/`
  submodule are untouched (contract §2.1, §6.1).
- **`ElicitationForwarder` full completion** (`from t0_agents import
  ElicitationForwarder`) — the bridge grows `on_request` +
  `default_timeout` constructor kwargs, a Future-based `handler`, and
  public `resolve(elicitation_id, payload)` / `cancel_all(action=...)`
  methods.  The empty-constructor "always decline" behaviour is
  preserved for backward compatibility.
- **`SessionProvider` Protocol + `CallableSessionProvider` adapter**
  (`from t0_agents.serve import SessionProvider,
  CallableSessionProvider`) — advanced extension point for per-session
  `Client` construction.  Implement the Protocol directly for auth,
  multi-tenant routing, custom elicitation wiring; the default
  `client_factory` / `elicitation_client_factory` kwargs wrap in a
  `CallableSessionProvider` automatically.
- **`create_app(..., streaming=True, session_provider=...,
  elicitation_client_factory=...)`** — three additive kwargs.
  `streaming=True` mounts `GET /chat/stream` (SSE) and `WS /chat/ws`
  (bidirectional).  `session_provider` is the advanced WS wiring slot.
  `elicitation_client_factory: Callable[[ElicitationHandler],
  Client[Any]]` is the simple WS wiring slot — pair it with the existing
  `client_factory`.  Existing callers that set only `client_factory` or
  `share_client=True` keep working unchanged.
- **WS wire protocol** (`from t0_agents.serve.ws_protocol import
  WSUserMessage, WSElicitationResponse, WSErrorFrame,
  WSElicitationRequest`) — Pydantic v2 envelopes for every frame in each
  direction.  Documented in `docs/streaming.md`.
- **`sse-starlette>=2.0`** added to the `[serve]` extra.
- Example `examples/07_streaming/` — runnable demo with a FastMCP
  server, `agent_app.py`, browser WS client (`static/index.html` +
  `static/app.js`) covering every `AgentEvent.type` and the elicitation
  round-trip.
- Public documentation at `docs/streaming.md` — simple +
  `SessionProvider`-advanced cookbooks, event-type reference, browser
  recipe, elicitation diagram, backpressure contract.

### Changed

- `SessionFactory` now accepts `session_provider=...` alongside
  `client_factory=...` and `shared_tool_provider=...` (three-way XOR).
  The original XOR validation error wording is preserved when exactly
  `client_factory` + `share_client` are both set (test compat).  The
  existing `session(extra_tags=...)` context manager gains an optional
  `elicitation_handler=...` kwarg that threads through to the
  provider's `build_client`.
- **`Agent.run(message, *, episode_id=None, tags=None)`** and
  **`Agent.stream(message, *, episode_id=None, tags=None)`** accept a
  per-call `tags` override that shallow-merges onto `AgentConfig.tags`
  (override wins on key collisions, configured deployment label is
  preserved).  Critical for the WebSocket path where one `Agent` is
  reused across frames — `AgentConfig.tags` alone cannot vary per
  frame.  Default `tags=None` leaves behaviour byte-identical for
  callers that never set it.
- **`WSUserMessage.tags`** is now honoured by the WebSocket route — the
  field existed in the wire protocol and was documented in
  `docs/streaming.md`, but `_drive_stream` dropped it before the
  gateway could observe it.  Now propagates through `agent.stream(tags=...)`.
- **`GET /chat/stream`** (SSE) accepts an optional `tags=<json>` query
  parameter (JSON-encoded `dict[str, str]`) for parity with `POST /chat`.
  Invalid JSON or non-object values surface as the first SSE frame — an
  `ErrorEvent` with `detail.kind == "invalid_tags"` — and the stream
  terminates.
- **`Guardrail.after_run`** Protocol signature tightened from
  `result: Any` to `result: AgentResult | None` and the docstring now
  documents explicitly that `result=None` is a valid value when the
  run aborted before producing an `AgentResult`.  Runtime behaviour is
  unchanged — `Agent.run` / `Agent.stream` have always invoked
  `after_run` in their `finally` blocks regardless of the result, per
  the `docs/guardrails.md` contract.

## [0.5.0] — Unreleased

### Added

- **HTTP serving layer** (`from t0_agents.serve import create_app,
  mount_agent, SessionFactory`) — declarative FastAPI wrap of an
  `Agent`. Requires the new `[serve]`
  extra (`pip install 't0-agents[serve]'`), which pulls
  `fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0`, and `pydantic>=2.0`.
- **`create_app(agent, *, client_factory=..., share_client=False,
  deployment_label=None)`** — builds a FastAPI app with `POST /chat`,
  `POST /feedback`, `GET /health`, `GET /info`. Validates that
  exactly one of `client_factory` / `share_client=True` is provided
  (no silent ambiguity). `deployment_label` is merged into every
  inference's tags as `t0_agents.deployment=<label>`.
- **`mount_agent(parent, agent, prefix, **kwargs)`** — mounts a
  per-agent sub-app under `prefix` on an existing FastAPI
  application. Same kwargs as `create_app`. Enables hosting several
  agents in one process on disjoint URL prefixes (`/mobile`,
  `/analytics`).
- **`SessionFactory`** — advanced: per-session `(Client, Agent)`
  builder that `create_app` uses internally. Exposes
  `session(extra_tags=...)` async context manager whose lifetime is
  the caller's `async with` block (one HTTP request initially; SSE /
  WS streams in the next release reuse the same primitive unchanged).
  Enforces the per-session-Client mandate by default.
- **`ElicitationForwarder`, `ElicitationRequest`**
  (`from t0_agents import ElicitationForwarder, ElicitationRequest`)
   — future-based elicitation bridge. The skeleton ships here with
  a default `handler` that declines everything (HTTP has no return
  channel). A later release completes `forward()` / `resolve()` /
  `wait_for()` for SSE + WS piping.
- **`Agent.warmup()`** — async readiness hook. Default no-op;
  override in a subclass to front-load readiness probes. Called by
  `serve.build_lifespan` during FastAPI startup.
- Example `examples/09_serving/` — runnable single-agent demo with
  `server.py`, `agent_app.py`, `run.sh`, `client.sh`, and a README
  that covers the multi-agent `mount_agent` pattern and the
  `share_client=True` trade-off.
- Public documentation at `docs/serving.md` — quickstart, per-surface
  deployment recipe, concurrency model, `share_client` trade-off,
  GKE / cloud-LB deployment notes (sticky sessions arrive with the
  streaming release).

### Changed

- This release does not modify `AgentConfig`; the serving layer
  wraps an existing `Agent` instance. `run_agent` and the `mcp/`
  submodule are untouched.

## [0.4.0] — Unreleased

### Added

- **Guardrail subsystem** (`from t0_agents.guardrails import require_approval,
  max_cost, max_tool_calls, block_tools`) — declarative policy layer on top
  of `Agent`.
- **`require_approval(tools=..., handler=...)`** — interactive approval gate
  riding FastMCP's `elicitation_handler` signature. Developers can reuse
  the exact handler they already pass to `Client(elicitation_handler=...)`.
- **`max_cost(per_episode=...)`** — cumulative cost ceiling per run. Hard
  aborts via `GuardrailBlockedError` when `response.usage.cost` crosses
  the cap.
- **`max_tool_calls(per_run=...)`** — per-run tool-call cap; counter
  incremented pre-dispatch (race-safe under `parallel_tools=True`).
- **`block_tools(names)`** — soft deny-list for tool names. Enforcement
  is at call time; does NOT filter `list_tools()`.
- **`Guardrail` Protocol** (`runtime_checkable`) — implement four optional
  hooks (`before_run`, `before_tool`, `after_tool`, `after_run`) to bring
  your own guardrail. Returning `str` from `before_tool` soft-blocks;
  raising `GuardrailBlockedError` hard-aborts.
- **`GuardrailContext`** — mutable per-run state (counter, cost, extra).
  Constructed inside `Agent.run()` so each run gets a fresh context.
- **`GuardrailLogEntry`** — one hook activation appended to
  `AgentResult.guardrail_log`. Attached to the result even on fatal abort
  via the `try/finally` around `run_agent`.
- **`GuardedToolProvider`** — the wrapper `Agent.run()` installs around
  the caller's `ToolProvider` when `config.guardrails` is non-empty.
  Exposed for advanced composition; normal users don't instantiate it.
- **`GuardrailBlockedError`** — hard-abort marker. Inherits from new
  `T0AgentsFatalError` base, which `_execute_single_tool_call` re-raises
  past its broad `except Exception` so fatal blocks propagate cleanly.
- **`T0AgentsFatalError`** — new base class for exceptions the agent
  loop must propagate (versus stringify into a tool error).
- **`on_inference` keyword argument on `run_agent`** —
  `Callable[[ChatInferenceResponse], Awaitable[None]] | None = None`.
  Fires after every `t0.inference()` response, BEFORE tool execution.
  `None` (the default) preserves behaviour byte-for-byte. Used internally
  by `Agent.run()` to track cumulative cost for `max_cost`.
- **`AgentResult.guardrail_log`** — new field (default empty list)
  carrying per-hook activations for audit and manual feedback emission.
- Example `examples/06_guardrails/` demonstrating all four built-ins
  composed on one agent with a CLI approval handler.
- Public documentation at `docs/guardrails.md`.

### Changed

- `AgentConfig` gained a `guardrails: list[Guardrail]` field (default
  empty). When empty, `Agent.run()` forwards the raw `tool_provider` and
  passes `on_inference=None` — byte-identical to 0.2.0.

## [0.3.0] — Unreleased

### Added

- **Memory subsystem** (`from t0_agents.memory import InMemoryMemory, SqliteMemory,
  MemoryProvider`) — repository-pattern memory for `Agent`. Configure
  `AgentConfig(memory=SqliteMemory("./db"))` to persist state across
  `.run()` calls.
- **`InMemoryMemory`** — ephemeral, dict-backed backend for tests and
  single-process apps.
- **`SqliteMemory`** — persistent backend behind the `[memory-sqlite]`
  extra (`pip install 't0-agents[memory-sqlite]'`).
- **`MemoryProvider` Protocol** — implement `load`, `save`, `search` to
  bring your own backend (Redis, Postgres, etc.).
- **`memory_strategy`** field on `AgentConfig` — controls how memory is
  injected into the prompt (`"auto"`, `"prepended_message"`,
  `"system_template"`).
- `system_arguments` keyword argument on `run_agent` — forwards template
  arguments to `input.system.arguments` in TensorZero inference calls.
- Example `examples/05_memory/` demonstrating cross-session memory.
- Public documentation at `docs/memory.md`.

## [0.2.0] — Unreleased

### Added

- **Agent class** (`from t0_agents import Agent, AgentConfig`) — declarative
  wrapper around `run_agent`. Configure once with `AgentConfig`, then call
  `agent.run(message)` with no boilerplate.
- **ToolProvider Protocol** — structural type matching `fastmcp.Client`'s
  `list_tools()` + `call_tool()` interface for testing and extensibility.
- **T0AgentsError** base exception — all library exceptions now inherit from
  `T0AgentsError` instead of bare `Exception`.
- **ConfigurationError** — raised when `AgentConfig` or `Agent` construction
  parameters are invalid (empty name, negative max_turns, etc.).
- `tags` and `namespace` keyword arguments on `run_agent` — forwarded to every
  `t0.inference()` call for observability and experimentation routing.
- Example `examples/02_agent_class/` demonstrating `Agent` usage.
- Public documentation at `docs/agent.md`.

### Changed

- `MaxTurnsExceededError` now inherits from `T0AgentsError` instead of
  `Exception`. Existing `except MaxTurnsExceededError` handlers are unaffected.

## [0.1.0] — 2026-04-01

Initial release. MCP-to-TensorZero bridge with `run_agent`, tool conversion,
sequential/parallel tool execution, elicitation support, and observability.
