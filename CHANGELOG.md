# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

- **`examples/17_structured/`** — five runnable demos:
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
  `examples/17_structured/config/functions/extract_entities/` —
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

## [0.9.0] — Unreleased

### Fixed

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

### Added

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
  option, paired with the new `MemoryProvider.load_history` extension
  method.  When set, the `Agent` skips the records-shaped `load()` and
  instead calls `memory.load_history(episode_id) -> list[Message] | None`
  and threads the result through `input.messages` as a TZ-native
  conversation history.  Sidesteps the unbounded summary growth that
  `prepended_message` exhibits when the backend reads its own writes
  back from the TZ log; preserves tool calls and tool results across
  turns so the model can reason about its own prior work.  Validated at
  `AgentConfig.__post_init__` — picking `"messages"` against a backend
  that does not implement `load_history` raises `ConfigurationError`.
- **`run_agent(prior_messages=...)`** — new optional kwarg on the
  low-level loop (mirrored on `StreamingAgentLoop.stream`) that seeds
  the conversation with a TZ-native message history before the new user
  prompt is appended.  Default `None` preserves byte-identical
  behaviour, honouring the frozen-signature invariant of the `mcp/`
  submodule.
- **Agent testing framework** (`from t0_agents.testing import MockGateway,
  MockToolProvider, EmbeddedGatewayHelper, assert_tool_called, …`).
  Milestone M8: Testable Agents.  Ships from the base distribution — no
  new extra, no new dependency.  Two complementary paths:
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
    Docker.  Resolves the final open question (T0 Q7) in the
    agentic-feature roadmap.
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
- **Example** [`examples/15_testing/`](examples/15_testing/) — a
  three-tool demo agent (`agent.py`) with both `test_agent_unit.py`
  (runs in < 1 s, no Docker, no API keys) and `test_agent_embedded.py`
  (opt-in via `RUN_EMBEDDED_TESTS=1`), plus a self-contained
  `pytest.ini` and `tensorzero.toml`.
- **Public doc** [`docs/testing.md`](docs/testing.md) — two-mode
  comparison table, `MockGateway` / `MockToolProvider` guides,
  assertion reference, scenarios, embedded setup, and a `dryrun=True`
  comparison.
- **`CONTRIBUTING.md`** (new) — prerequisites, three test modes
  (unit / integration / embedded), style gates, single-test commands,
  and a skeleton for writing a new test with `MockGateway`.

### Changed

- Phase H does not modify `core/`, `mcp/`, `stream/`, `serve/`,
  `memory/`, `guardrails/`, `compose/`, or `cli/`.  `run_agent`,
  `AgentResult`, `Agent`, and every pre-existing public symbol are
  byte-identical to 0.8.0 (contract §2.1, §6.1–§6.3).  All 399 prior
  offline tests remain green alongside the 56 new `tests/test_testing/`
  tests — 455 total passing in `-m "not integration and not embedded"`.

## [0.8.0] — Unreleased

### Added

- **Multi-agent composition** (`from t0_agents import agent_as_tool,
  SubAgentError`; `from t0_agents.compose import install_sub_agents,
  assert_unique_names, default_tool_name`).  Milestone M7: Composed
  Agents.  No new dependencies — built on top of the existing
  `core/` + `mcp/` surfaces and FastMCP's public runtime-registration
  APIs (`Tool.from_function(...)` + `server.add_tool(tool)`).
- **`AgentConfig.sub_agents: list[Agent]`** — declarative
  delegation.  When non-empty, `Agent.__init__` registers each entry
  as a `{sub_agent.config.name}_ask` tool on the orchestrator's
  in-process FastMCP server via
  `t0_agents.compose.install_sub_agents`.  Empty (the default)
  preserves Phase A byte-identity.  Requires a `fastmcp.Client`
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
- **Example** [`examples/14_multi_agent/`](examples/14_multi_agent/)
  ships a researcher + writer + orchestrator trio with its own
  `docker-compose.yml`, `config/tensorzero.toml` (three functions:
  `research`, `writing`, `orchestrator`), and both a `main.py`
  entrypoint and an `orchestrator_factory` for
  `t0-agents run examples.14_multi_agent.agents:orchestrator_factory`.

## [0.7.0] — Unreleased

### Added

- **CLI** (`from t0_agents.cli import TUIRenderer, rich_elicitation_handler,
  make_rich_elicitation_handler, load_agent_factory`) — Typer + Rich
  terminal UI. Milestone M6: TUI Agent. Ships behind the new `[cli]`
  extra (`pip install 't0-agents[cli]'`), which pulls `typer>=0.12.0`,
  `rich>=13.7.0`, plus the streaming + serving deps (`fastapi`,
  `uvicorn[standard]`, `pydantic`, `sse-starlette`).
- **`t0-agents` entry-point script** wired to `t0_agents.cli.app:main`.
  Exposes three subcommands:
  - `t0-agents run <module:attr>` — interactive streaming REPL with
    live deltas, tool-call spinners, and Rich-rendered elicitation
    prompts. `--verbose` surfaces reasoning content.
  - `t0-agents serve <module:attr>` — wraps
    `t0_agents.serve.create_app(..., share_client=True)` in uvicorn.
    Phase D non-streaming routes by default (`POST /chat`, `/feedback`,
    `/health`, `/info`); opt in to SSE + WS with `--streaming`. WS
    elicitation is incompatible with shared-Client mode (FastMCP Q3),
    so `--streaming` is only useful for tools that never call
    `ctx.elicit` — for full browser elicitation, call `create_app`
    directly with a `client_factory` + `elicitation_client_factory`
    (see `examples/12_streaming_http/agent_app.py`). `--host`,
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
- Example `examples/13_tui/` — runnable demo with a FastMCP server
  (`add`, `set_reminder` with `ctx.elicit(response_type=Reminder)`),
  a `@asynccontextmanager` factory, and a README walkthrough of the
  three subcommands.
- Public documentation at `docs/cli.md` covering install, the three
  commands, the factory convention, custom renderers, elicitation
  semantics, and troubleshooting.

### Changed

- Phase F does not modify `core/`, `mcp/`, `stream/`, `serve/`,
  `memory/`, or `guardrails/`. `run_agent`, `AgentResult`, `Agent`,
  and every existing public symbol are byte-identical to 0.6.0
  (contract §2.1, §6.1–§6.3). Pre-existing offline tests remain
  green (363 passed across Phase A–F).

## [0.6.0] — Unreleased

### Added

- **Streaming primitives** (`from t0_agents import AgentEvent, DoneEvent,
  TextDeltaEvent, TextEndEvent, ToolCallDeltaEvent, ToolCallEvent,
  ToolResultEvent, ThoughtDeltaEvent, ThoughtEvent, TurnEndEvent,
  ErrorEvent, StreamingAgentLoop, TurnStreamAssembler`) — typed event
  union emitted by `Agent.stream()`.  Milestone M5: Live Agent.  Ships
  from the base distribution (no extra needed).
- **`Agent.stream(message, *, episode_id=None) -> AsyncIterator[AgentEvent]`**
  — streaming analogue of `Agent.run`.  Same memory / guardrail
  scaffolding; the terminal event is always `DoneEvent(result=...)`
  carrying a full `AgentResult`.  Non-streaming `Agent.run` is
  byte-identical to Phase D.
- **`TurnStreamAssembler`** — folds `AsyncIterator[ChatChunk]` from
  `AsyncTensorZeroGateway.inference(stream=True)` into an
  `AsyncIterator[AgentEvent]` with stable correlation ids per T0 Q1.
- **`StreamingAgentLoop`** — streaming analogue of `mcp.loop.run_agent`;
  reuses `convert_tools`, `execute_tool_calls[_parallel]` and the
  recoverable-retry contract from `mcp/`.  `run_agent` and the `mcp/`
  submodule are untouched (contract §2.1, §6.1).
- **`ElicitationForwarder` full completion** (`from t0_agents import
  ElicitationForwarder`) — Phase D skeleton grows `on_request` +
  `default_timeout` constructor kwargs, a Future-based `handler`, and
  public `resolve(elicitation_id, payload)` / `cancel_all(action=...)`
  methods.  Empty-constructor behaviour (always decline) is preserved
  for Phase D compatibility.
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
  `client_factory`.  Phase D callers that set only `client_factory` or
  `share_client=True` keep working unchanged.
- **WS wire protocol** (`from t0_agents.serve.ws_protocol import
  WSUserMessage, WSElicitationResponse, WSErrorFrame,
  WSElicitationRequest`) — Pydantic v2 envelopes for every frame in each
  direction.  Documented in `docs/streaming.md`.
- **`sse-starlette>=2.0`** added to the `[serve]` extra.
- Example `examples/12_streaming_http/` — runnable demo with a FastMCP
  server, `agent_app.py`, browser WS client (`static/index.html` +
  `static/app.js`) covering every `AgentEvent.type` and the elicitation
  round-trip.
- Public documentation at `docs/streaming.md` — simple +
  `SessionProvider`-advanced cookbooks, event-type reference, browser
  recipe, elicitation diagram, backpressure contract.

### Changed

- `SessionFactory` now accepts `session_provider=...` alongside
  `client_factory=...` and `shared_tool_provider=...` (three-way XOR).
  Phase D XOR validation error wording is preserved when exactly
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
  `Agent`. Milestone M4: Hosted Agent. Requires the new `[serve]`
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
  the caller's `async with` block (one HTTP request in Phase D; Phase
  E's SSE / WS streams will reuse the same primitive unchanged).
  Enforces the FastMCP Q3 / contract §3.5 per-session-Client mandate
  by default.
- **`ElicitationForwarder`, `ElicitationRequest`**
  (`from t0_agents import ElicitationForwarder, ElicitationRequest`)
   — future-based elicitation bridge. Phase D ships the skeleton with
  a default `handler` that declines everything (HTTP has no return
  channel). Phase E will complete `forward()` / `resolve()` /
  `wait_for()` for SSE + WS piping.
- **`Agent.warmup()`** — async readiness hook. Default no-op;
  override in a subclass to front-load readiness probes. Called by
  `serve.build_lifespan` during FastAPI startup.
- Example `examples/11_http_agent/` — runnable single-agent demo with
  `server.py`, `agent_app.py`, `run.sh`, `client.sh`, and a README
  that covers the multi-agent `mount_agent` pattern and the
  `share_client=True` trade-off.
- Public documentation at `docs/serving.md` — quickstart, per-surface
  deployment recipe, concurrency model, `share_client` trade-off,
  GKE / cloud-LB deployment notes (sticky sessions planned for
  Phase E).

### Changed

- Phase D does not modify `AgentConfig`; the serving layer wraps an
  existing `Agent` instance. `run_agent` and the `mcp/` submodule are
  untouched (contract §2.1, §6.1).

## [0.4.0] — Unreleased

### Added

- **Guardrail subsystem** (`from t0_agents.guardrails import require_approval,
  max_cost, max_tool_calls, block_tools`) — declarative policy layer on top
  of `Agent`. Milestone M3: Trusted Agent.
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
- Example `examples/10_guardrails/` demonstrating all four built-ins
  composed on one agent with a CLI approval handler.
- Public documentation at `docs/guardrails.md`.

### Changed

- `AgentConfig` gained a `guardrails: list[Guardrail]` field (default
  empty). When empty, `Agent.run()` forwards the raw `tool_provider` and
  passes `on_inference=None` — byte-identical to Phase B.

## [0.3.0] — Unreleased

### Added

- **Memory subsystem** (`from t0_agents.memory import InMemoryMemory, SqliteMemory,
  MemoryProvider`) — repository-pattern memory for `Agent`. Configure
  `AgentConfig(memory=SqliteMemory("./db"))` to persist state across
  `.run()` calls. Milestone M2: Stateful Agent.
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
- Example `examples/09_memory/` demonstrating cross-session memory.
- Public documentation at `docs/memory.md`.

## [0.2.0] — Unreleased

### Added

- **Agent class** (`from t0_agents import Agent, AgentConfig`) — declarative
  wrapper around `run_agent`. Configure once with `AgentConfig`, then call
  `agent.run(message)` with no boilerplate. Milestone M1: Declarative Agent.
- **ToolProvider Protocol** — structural type matching `fastmcp.Client`'s
  `list_tools()` + `call_tool()` interface for testing and extensibility.
- **T0AgentsError** base exception — all library exceptions now inherit from
  `T0AgentsError` instead of bare `Exception`.
- **ConfigurationError** — raised when `AgentConfig` or `Agent` construction
  parameters are invalid (empty name, negative max_turns, etc.).
- `tags` and `namespace` keyword arguments on `run_agent` — forwarded to every
  `t0.inference()` call for observability and experimentation routing.
- Example `examples/08_agent_class/` demonstrating `Agent` usage.
- Public documentation at `docs/agent.md`.

### Changed

- `MaxTurnsExceededError` now inherits from `T0AgentsError` instead of
  `Exception`. Existing `except MaxTurnsExceededError` handlers are unaffected.

## [0.1.0] — 2026-04-01

Initial release. MCP-to-TensorZero bridge with `run_agent`, tool conversion,
sequential/parallel tool execution, elicitation support, and observability.
