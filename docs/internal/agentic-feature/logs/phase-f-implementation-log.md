# Phase F Implementation Log

Phase: F (TUI — Typer + Rich CLI)
Milestone: M6 — TUI Agent
Date: 2026-04-16

## Divergences from Spec

### D1: Loader contract is factory-only, not "bare Agent"

**Spec (§4.1 / §8 Code checklist)** sketched
`load_agent(spec: str) -> Agent` returning a module-level `Agent`
instance (mirroring FastAPI's `myapp:app` convention).

**Actual.** The public function is
`load_agent_factory(spec: str) -> AgentFactory`, where
`AgentFactory = Callable[[], AbstractAsyncContextManager[Agent]]`.
The user's `module:attr` must resolve to a zero-arg callable (idiomatically
`@asynccontextmanager async def attr(): ...`) that yields an `Agent`
after both the `AsyncTensorZeroGateway` and the `fastmcp.Client` have
been entered.

**Reason.** `Agent.stream()` / `Agent.run()` ultimately call through to
`fastmcp.Client.list_tools()` which fails unless the Client's MCP
session has been opened (i.e. the caller has entered `async with
Client(mcp):`). `AsyncTensorZeroGateway.build_http(..., async_setup=True)`
also returns an awaitable that must be `await`-ed. A bare module-level
`Agent(..., tool_provider=Client(mcp))` — the Phase-E serve idiom for
`create_app(share_client=True)` — is NOT usable directly from the REPL
because the Client session is never open.

Three alternatives were considered and rejected:

- *Reach into `agent._tool_provider` and enter it from the CLI.* Touches
  a private attribute; the `ToolProvider` Protocol only guarantees
  `list_tools` / `call_tool` so `async with` may not even be meaningful.
- *Extend `Agent` with `__aenter__` / `__aexit__`.* Widens the Agent
  public surface mid-roadmap; contract §3.2 explicitly says "no phase
  should remove or rename an `AgentConfig` field" and the same spirit
  applies to adding new lifecycle methods.
- *Accept both `Agent` and a factory (duck-typed).* The loader would have
  to pick between "enter the private tool_provider" and "enter the
  factory" — two separate failure modes; the second bare path would
  silently diverge from the factory path for ToolProviders that don't
  support `async with`.

The factory convention loses no expressive power and gains a clean
lifecycle bracket: the CLI opens the factory once around the whole
session (REPL loop or uvicorn lifespan), and user code is free to
compose memory backends / guardrails / custom gateways inside.

`typer.BadParameter` on a bare-Agent input includes an inline
`@asynccontextmanager` snippet pointing the user at the correct shape
— that's the most common mistake and the error message makes it
one-edit-to-fix.

**Impact.** Documented in `docs/cli.md` ("The agent factory
convention") and in the developer guide §12 knobs entry. No contract
amendment: `Agent` / `AgentConfig` / `AgentResult` / `run_agent` / the
`mcp/` submodule are byte-identical. `examples/13_tui/agent.py` is the
canonical reference; same factory works for `run`, `serve`, and `info`.

### D2: `t0-agents serve` defaults `streaming=False` (corrected post-merge — see D10)

**Spec (§4.5).** Sketched `serve_cmd` as a thin wrap of
`create_app(agent, deployment_label=deployment)` without a streaming
flag, implicitly inheriting Phase D's `streaming=False` default.

**Initial choice.** Defaulted to `streaming=True` with `--no-streaming`
as the opt-out, on the theory that Phase F's purpose was a
streaming + elicitation demo and pointing the user at the browser
client in `examples/12_streaming_http/static/index.html` from a single
`t0-agents serve` command was the most demo-friendly default.  The
subcommand always uses `share_client=True` so the factory-entered
`Client` is reused across sessions.

**Why this was wrong.**  The user's first manual smoke-test against
the streaming default crashed every WebSocket connection:
`SessionFactory.session()` raises `ConfigurationError` when shared
mode is asked for an `elicitation_handler`, and the WS route always
passes the per-connection forwarder handler.  The combination
`streaming=True` + `share_client=True` is structurally broken.  See
**D10** for the post-merge correction.

### D3: `make_rich_elicitation_handler` factory alongside `rich_elicitation_handler`

**Spec (§4.3).** Sketched a single `rich_elicitation_handler` function
matching FastMCP's `ElicitationHandler` signature.

**Actual.** Two symbols:

- `rich_elicitation_handler(message, response_type, params, context)` —
  zero-configuration handler. Internally delegates to
  `make_rich_elicitation_handler()`. Safe to plug into
  `Client(mcp, elicitation_handler=rich_elicitation_handler)` without
  any CLI scaffolding (used in `examples/13_tui/agent.py`).
- `make_rich_elicitation_handler(*, renderer=None, console=None)` —
  factory that returns a fresh handler bound to an optional
  `TUIRenderer`. When a renderer is supplied, the handler pauses the
  Live region before each prompt and resumes after (addressing the
  Phase F §10 Live-vs-Prompt risk).

**Reason.** The core design problem is that Rich's `Live` redirects
`stdout` / `stderr` by default (`rich/live.py:199`) and does not
auto-pause during a `Prompt.ask` call. A stock handler that doesn't
know about the renderer would scramble the panel on every prompt.
A factory that closes over a `TUIRenderer` is the cleanest way to
express "pause before prompting, resume after" without making every
caller manage that coordination themselves.

**Impact.** Exported from `t0_agents.cli`:
`rich_elicitation_handler`, `make_rich_elicitation_handler`. Documented
in `docs/cli.md` ("Elicitation in the TUI") and the developer guide §4
("CLI symbols"). No contract amendment — the handler shape still
matches FastMCP's `ElicitationHandler` type alias byte-for-byte.

### D4: URL-mode elicitation dispatch added explicitly

**Spec (§4.3).** Only covered binary approval (`response_type=None`) and
form-mode schemas.

**Actual.** The handler `isinstance`-checks `params` for
`ElicitRequestFormParams` (via `mcp.types`). When the params are
`ElicitRequestURLParams` (the OAuth / payment-flow path), the handler
prints the URL, asks the user to confirm out-of-band completion via a
`Confirm`, and returns `ElicitResult(action="accept"|"decline",
content=None)`.

**Reason.** `mcp.types.ElicitRequestParams` is a Union of form / URL
variants (`mcp/types.py:1884`). FastMCP Q5's schema guarantees only
apply to the form mode; URL mode has no schema at all. The handler has
to branch on this at the top, not try to introspect `requestedSchema`
on a URL-variant params object (which would raise `AttributeError`).

**Impact.** Tested with `test_url_elicitation_prompts_for_done`. No
documentation changes needed beyond mentioning the two modes in
`docs/cli.md`.

### D5: `response_type is None` is NOT the binary-approval trigger

**Spec (§4.3).** Code sketch wrote:

```python
if response_type is None:
    # Binary approval
    ok = Confirm.ask(message, default=False)
```

**Actual.** The binary trigger is `not params.requestedSchema.get("properties")`.
The handler reads the schema directly and ignores `response_type`.

**Reason.** In FastMCP's wrapper
(`fastmcp/client/elicitation.py:47-54`), `response_type is None` only
happens when `requestedSchema == {"type": "object", "properties": {}}`
— the two states are always in lockstep. But using the schema as the
source of truth (rather than trusting that `response_type` mirrors it)
is more robust to non-FastMCP MCP servers that may pass `None` for
`response_type` while still sending a populated schema. The test
`test_unknown_type_falls_back_to_string` exercises exactly this
scenario.

**Impact.** None user-visible — for a FastMCP server the two code
paths converge. Noted here for future contributors who may otherwise
wonder why the handler doesn't read `response_type`.

### D6: `[cli]` extra bundles `[serve]` dependencies transitively

**Spec (§4.7).** Declared `[cli]` as `["typer>=0.12.0", "rich>=13.0.0"]`.

**Actual.** `[cli]` includes `typer`, `rich`, and every dep from `[serve]`
(`fastapi`, `uvicorn[standard]`, `pydantic`, `sse-starlette`).

**Reason.** `t0-agents serve` imports from `t0_agents.serve.create_app`
(and transitively `fastapi`), and the REPL's streaming path uses
`sse-starlette` the same way Phase E does. Splitting the extras so
users who want `run` but not `serve` could avoid `fastapi` would leave
`t0-agents serve` with a cryptic import failure. Bundling is the
minimum that makes all three subcommands self-contained behind a
single `pip install 't0-agents[cli]'`. Users who really want a
serving-only install still have `pip install 't0-agents[serve]'`
(unchanged).

**Impact.** Documented in `docs/cli.md` "Install". No contract
amendment — extras are still opt-in per contract §2.3.

### D7 (post-merge bugfix): Live ↔ Prompt coordination via module-global

**Observed by user during manual smoke-test (2026-04-16).** With
`examples/13_tui/agent.py` wiring the zero-config
`rich_elicitation_handler` into `Client(mcp,
elicitation_handler=rich_elicitation_handler)`, the Rich `Live` panel
kept repainting every time the user typed a character in response to
an elicitation prompt — producing a stack of "🔧 set_reminder(...)"
frames on top of each other instead of a clean `Confirm`.

**Root cause.** Two layers, discovered in order:

1. The factory-only version of the handler (in D3 above) hard-coded
   the `renderer` binding at factory construction time.  The public
   `rich_elicitation_handler` symbol called
   `make_rich_elicitation_handler()` with no renderer, so its
   `pause()` / `resume()` no-oped.  `Live`'s 20 Hz refresh thread
   stayed running while the prompt blocked on stdin, redrawing the
   panel every 50 ms and scrambling the user's input line.
2. *First-attempt fix* used a :class:`contextvars.ContextVar` to
   expose the active renderer.  That did not work in practice: the
   MCP `ClientSession` calls `anyio.create_task_group()` inside its
   ``__aenter__`` and immediately runs the ``_receive_loop`` as a
   sub-task of that group (`mcp/shared/session.py:222-224`).  That
   task's context snapshot is taken at Client ``__aenter__`` time —
   i.e. *before* `TUIRenderer.consume` ever runs.  The elicitation
   callback fires from inside that task, so ``ContextVar.get()``
   returned ``None`` even though ``consume`` had ``set(self)`` on a
   different task.

**Fix.** Replaced the ContextVar with a module-level binding
(`src/t0_agents/cli/tui.py::_active_renderer`) guarded by a
:class:`threading.Lock` (the TUI is single-process, Rich's refresh
thread is the one other writer that matters).
`TUIRenderer.consume` now calls ``_set_active_renderer(self)`` at
entry and ``_set_active_renderer(None)`` in the outer ``finally`` so
any asyncio task or thread can read the current renderer via
``get_active_renderer()``.

`rich_elicitation_handler` and `make_rich_elicitation_handler(renderer=None)`
both fall back to `get_active_renderer()` at call-time; explicit
`renderer=` still wins.  The zero-config plug-it-in-and-go pattern
that `examples/13_tui/agent.py` uses now picks up the live renderer
regardless of which anyio task group the FastMCP session happens to
run its receive loop on.

`get_active_renderer()` is exposed as a public symbol on
`t0_agents.cli` for advanced users who want to inspect the state (e.g.
from a custom handler layered on top of the stock one).

**Tests.**  Three new tests in `test_elicit_tui.py`:

- `test_handler_pauses_active_renderer_during_prompt` — installs a
  tracking fake via the module-level setter, invokes the handler,
  and asserts the `pause / prompt / resume` ordering.
- `test_handler_sees_renderer_across_task_boundary` — spawns the
  handler call on an `asyncio.create_task(...)` created *after* the
  renderer registration, mirroring the MCP `_receive_loop` scenario.
  This test would fail with the ContextVar implementation because the
  new task's context snapshot does not include the fresh `set()`
  value when the task was spawned from outside the `ContextVar.set()`
  block.
- `test_handler_without_active_renderer_skips_pause` — regression
  guard so the non-CLI usage keeps working.

**Impact.** No contract amendment.  Public API grew one symbol
(`get_active_renderer`).  The Phase F §10 "Rich + Live conflicts with
stdin prompts during elicitation" risk row is now addressed in code,
not just in docs.  The only caveat: `_active_renderer` is a
single-process global, so two concurrent `run` REPLs in the same
process would confuse each other — not a realistic scenario for a
human-facing TUI.

**Followup investigation (2026-04-16).** Cross-checked the
`Live.stop()` / `Live.start(refresh=True)` pattern against the Rich
source after a second user-reported repaint:

- Rich does not ship a native `Live.pause()`.  `Live.start` /
  `Live.stop` are the only public knobs (`rich/live.py`); the official
  guide (`docs/source/live.rst`) does not describe a "pause + prompt +
  resume" pattern, so the design here is necessarily ad-hoc.
- `Live.stop()` does not `join()` the refresh thread, but it acquires
  `Live._lock` before signalling `done`.  An in-flight `refresh()`
  contends for the same lock and finishes before `stop()` returns; the
  `if not self.done.is_set()` re-check inside `_RefreshThread.run`
  prevents any further frame from being painted afterwards.  Net
  effect: when `pause()` returns, the panel will not paint over the
  prompt.
- `Console.pop_render_hook()` is synchronous and atomic under the
  console lock, so `Prompt.ask(console=X)` immediately after `stop()`
  is not intercepted by the (now-popped) Live render hook.
- `Console.input` is a thin wrapper around the builtin `input()` (plus
  optional `readline`) and does not touch `termios` or stdin; nothing
  in `Live` puts the terminal in raw mode.  No state to restore beyond
  the render hook + stdout/stderr redirection that `Live.stop()`
  already unwinds.

Conclusion: the current `pause()` / `resume()` pair is the only
documented option, and it is correct against the source.  The handler
also catches `EOFError` (in addition to `KeyboardInterrupt`) so a
stdin-exhausted prompt collapses into a clean `cancel` action instead
of propagating a tool-side error.  A `--debug-log <path>` flag on
`t0-agents run` writes DEBUG-level logs from `t0_agents.*` to a file,
left in place as the supported diagnostic for future regressions in
the Live ↔ elicitation interaction.

### D8 (ergonomics): loader auto-adds cwd to `sys.path`

**Observed during manual smoke-test (2026-04-16).** `uv run t0-agents
info examples.13_tui.agent:assistant` failed with `No module named
'examples'`. Python adds the script's own directory (``.venv/bin``) to
``sys.path`` for installed console scripts, not the user's cwd the way
`python script.py` does — so the `examples/` folder wasn't
importable.

**Fix.** `load_agent_factory` now calls an internal helper
`_ensure_cwd_on_path` that prepends `os.getcwd()` to `sys.path` if
neither `""` nor the absolute cwd is already present. Idempotent; no
effect on test runs where pytest has already injected cwd. Uvicorn and
FastAPI leave this to the user, but the user's explicit Phase F hint
("como es de fácil integrar a un TUI") rewards the gentler default.

**Impact.** None on callers or the contract — the cwd was always
importable when users did the right thing; this just removes the
tripwire. Tested manually (`uv run t0-agents info
examples.13_tui.agent:assistant` now renders the full config table
without a `PYTHONPATH=` prefix).

### D9: Offline run-subcommand test input ends at `\n` rather than `\x04`

**Spec (§5.2).** Suggested `CliRunner.invoke(app, [...], input="...\n\x04")`
with `\x04` as a Ctrl-D sentinel.

**Actual.** Tests use `input="...\n"` (no `\x04`). When the scripted
stdin is exhausted, Python's `input()` raises `EOFError` which the REPL
catches and exits cleanly.

**Reason.** `click.testing.CliRunner` writes `input` to the subprocess
stdin verbatim. `\x04` is the EOT byte but on a piped stdin (not a
tty) it's treated as normal data — `Prompt.ask` returns the literal
`"\x04"` string, `"\x04".strip()` is truthy, and the REPL tries to
stream that as a user message. Letting stdin exhaust naturally is the
correct idiom for piped input.

**Impact.** Integration tests (`tests/test_cli/test_integration.py`)
follow the same pattern.

### D10 (post-merge bugfix): `t0-agents serve` defaults `streaming=False`

**Observed by user during manual smoke-test (2026-04-16).** The PR's
test plan included `uv run t0-agents serve
examples.13_tui.agent:assistant --port 8080` followed by `GET /health`.
The server booted, `/health` returned, but the user's first
`new WebSocket('ws://localhost:8080/chat/ws')` collapsed every
connection with:

```
ConfigurationError: Streaming elicitation is incompatible with
'shared_tool_provider' mode — each caller needs its own elicitation
handler (FastMCP Q3).  Use 'client_factory' +
'elicitation_client_factory' or a 'session_provider'.
```

**Root cause.** The combination *baked in by D2*
(`streaming=True` + `share_client=True`) is structurally invalid: the
WS route at `serve/streaming_routes.py:114` always passes a
per-connection `ElicitationForwarder.handler` to
`SessionFactory.session(...)`, and shared-mode raises on any non-`None`
elicitation handler (`serve/session.py:245-252`).  D2's "demo-friendly
default" only worked for tools that never call `ctx.elicit` — not the
`set_reminder` tool the example ships.

**Fix.** Default `--streaming` to `False` so the most common path
(`POST /chat`, `/feedback`, `/health`, `/info`) Just Works against the
factory's shared `Client`.  `--streaming` remains opt-in and the
docstring + `docs/cli.md` + `examples/13_tui/README.md` now spell out
the limitation: streaming serves SSE fine, but every WS connection
that triggers `ctx.elicit` will close with the original error.  For
real browser-elicitation streaming the user is pointed at
`examples/12_streaming_http/agent_app.py`, which wires
`client_factory` + `elicitation_client_factory` directly into
`create_app`.

**Why no startup check.**  Adding "raise at `create_app(streaming=True,
share_client=True)` startup" would be a Phase D/E cross-cutting
change.  Phase F's surface ends at the CLI; the safer scoped fix is
to flip the CLI default and document the trade-off.  A future phase
can promote the check into the library if it ever ships a non-CLI
caller that hits the same combination.

**Tests.**  Existing `tests/test_cli/test_integration.py` tests serve
without `--streaming`, so they were already exercising the new
default.  No new test added — adding one would just re-cover the
Phase E `create_app(share_client=True, streaming=False)` integration
that `tests/test_serve/` already covers.

**Impact.**  No contract amendment.  Public API unchanged.  `D2` is
left in place as the historical record of the original choice with a
forward pointer to this entry.

## Resolved Open Questions

No open questions from `02-open-questions.md` were newly answered
during Phase F. The three questions the spec referenced were all
pre-resolved:

- **FastMCP Q5 (elicitation schema)** — The flat-object subset with
  primitives + `enum` + `const`, preserved `description` / `default`,
  and scalar-wrap semantics drove the renderer's dispatch logic in
  `cli/elicit_tui.py`. Fail-soft fallback to `Prompt.ask(str)` is kept
  for schemas originating from non-FastMCP MCP servers.
- **FastMCP Q3 (Client concurrency)** — `t0-agents serve` uses
  `share_client=True` which explicitly violates the
  per-session-Client-per-elicitation-handler guarantee.  D10 corrects
  the original `streaming=True` default to `False` so the dev-server
  use case stays inside the safe non-streaming subset; opting in to
  `--streaming` is documented as "SSE-only; WS elicitation will
  close every connection with `ConfigurationError`".
- **T0 Q1 (streaming chunks)** — The renderer relies on the stable
  block-ID correlation key: `TextDeltaEvent(id=...)` deltas accumulate
  per id, `ToolCallEvent(id=...)` → `ToolResultEvent(id=...)` pair via
  the shared id.

## Contract Amendments

None. `00-integration-contract.md` is unchanged:

- §2.1 (`mcp/` standalone): `mcp/` is byte-identical.
  `from t0_agents.mcp import run_agent, convert_tools` continues to
  work without the `[cli]` extra. `cli/` depends on `core/`, `stream/`,
  `serve/` (permitted per §2.2 / the module dep graph).
- §2.3 (extras opt-in): `[cli]` is a new opt-in extra; base
  installation does not grow.
- §2.5 (elicitation-only HITL): `rich_elicitation_handler` is a thin
  renderer over the existing FastMCP `ElicitationHandler` signature.
  No parallel HITL primitive introduced.
- §2.6 (async-first): every `Agent` / `Client` interaction in the CLI
  is async. The Typer subcommand handlers are sync entry points that
  call `asyncio.run(...)` internally — matches the contract's
  "sync wrapper only when it genuinely simplifies the caller's life"
  allowance.
- §3.2 / §3.3 (Agent public contract): unchanged. `cli/` consumes
  `Agent.run` / `Agent.stream` through the existing signatures.
- §6.1–§6.3 (`run_agent` / `AgentResult` / `mcp/` frozen): unchanged.

## Verification

### Offline

- `uv run pytest -m "not integration"` → **363 passed, 40 deselected**
  (includes 45 new CLI tests).
- `uv run ruff check src/` → **All checks passed.**
- `uv run mypy src/` → **Success: no issues found in 41 source files**
  (strict mode).

### Integration

Integration tests at `tests/test_cli/test_integration.py` cover:

- `test_info_subcommand_against_live_example` — spawns
  `uv run t0-agents info examples.13_tui.agent:assistant` in a
  subprocess; asserts the rendered `AgentConfig` table contains
  `tui-assistant` and `function_name`.
- `test_serve_subcommand_health_endpoint` — spawns
  `uv run t0-agents serve ...` on port 18081; polls `GET /health`
  until 200; then SIGTERM.
- `test_run_subcommand_single_turn` — drives one user turn through
  `uv run t0-agents run ...` with scripted stdin; asserts the
  assistant response contains the arithmetic result "5".

These require `docker compose up -d` plus a provider API key in the
environment. They were NOT executed during this phase — Phase F was
implemented while the gateway was down. The offline suite covers every
unit-level code path in the new `cli/` submodule.

### Manual smoke-test — not run this session

- `uv run t0-agents run examples.13_tui.agent:assistant` — will exercise
  streaming deltas + `set_reminder` elicitation against a live gateway.
- `uv run t0-agents serve examples.13_tui.agent:assistant --port 8080` —
  will boot `create_app(streaming=True, share_client=True)`.

## Release Notes Snippet

> **v0.7.0 — TUI Agent (M6)**
>
> Terminal UI for `Agent`: `pip install 't0-agents[cli]'` gets you a
> `t0-agents` console script with three subcommands — `run`, `serve`,
> and `info`. The streaming REPL drives `Agent.stream(...)` under a
> Rich `Live` panel with per-tool-call status lines, accumulated
> assistant text, and Rich-rendered elicitation prompts (boolean
> `Confirm`, typed scalar prompts, `enum` choice prompts). `t0-agents
> serve` wraps `create_app(streaming=True, share_client=True)` in
> uvicorn for a one-shot dev server. The `module:attr` loader expects
> a zero-arg factory returning an async context manager yielding an
> `Agent` — idiomatically `@asynccontextmanager async def attr(): ...`
> — so the CLI can bracket the gateway + Client lifecycle around the
> whole session. Public hooks for custom renderers: `TUIRenderer`
> consumes `AsyncIterator[AgentEvent]`; `rich_elicitation_handler` /
> `make_rich_elicitation_handler(renderer=..., console=...)` is the
> FastMCP elicitation entry point. Phase A–E behaviour is
> byte-identical (regression covered by 363 offline tests). See
> `docs/cli.md` and `examples/13_tui/`.
