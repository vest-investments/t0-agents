# The wire contract — `t0_agents.wire`

```python
from t0_agents.wire import (
    WIRE_VERSION, WireFrame, frame_to_dict, frame_from_dict,
    project, as_terminal_safe, with_status, with_text_fallback, public_allowlist,
)
```

`t0_agents.wire` is the library's versioned promise to stream
consumers. Before it, the de-facto wire contract was "whatever
`dataclasses.asdict(event)` yields" — it shifted with every new field,
and every consumer re-invented a mirror union, a codec, a terminal-event
guarantee, a reasoning fallback, and a status synthesiser (~2,000 lines
the first time). The wire package makes those library promises: a
frozen frame union, a versioned dict codec, and pure composable
projection stages. Upstream event churn is absorbed here, in the same
repo and PR that introduces it — the `assert_never` lives alongside the
union it guards.

> **Not a broker.** The package publishes stable dicts to *any*
> transport and imports none. Redis buses, SSE serialisers, WebSockets,
> heartbeats, resume ids, and backpressure are consumer infrastructure
> — they compose *around* these stages, never inside them.

## Version policy

`WIRE_VERSION` (currently `1`) governs the codec shape, and evolution is
**additive only**:

- A new wire shape is a new frame class with a new `kind` literal.
  Existing frames never lose or rename fields; existing `kind` strings
  never change meaning. Consumers with exhaustive `match` statements
  add one arm and are otherwise untouched.
- New *fields* on an existing frame get defaults, so old payloads still
  decode — and `frame_from_dict` ignores unknown fields, so an older
  reader tolerates payloads from a newer additive writer. Unknown
  *kinds* still raise.
- `WIRE_VERSION` is bumped **only** on a non-additive change, which is
  a breaking release of the contract. `frame_from_dict` rejects
  mismatched versions with `ValueError` — an outdated payload fails
  loudly, never degrades silently.

Every stage `match`es the full union and ends in `assert_never`, so a
new frame variant refuses to type-check until each stage makes an
explicit decision about it. For `public_allowlist` that decision is
render-vs-drop — render-by-default would be a privacy regression.

## The frame union

| Frame | `kind` | Fields | Produced by |
|---|---|---|---|
| `InitFrame` | `init` | `conversation_id` | consumer, once per subscription |
| `TextDeltaFrame` | `text_delta` | `block_id, delta` | `project()` |
| `TextEndFrame` | `text_end` | `block_id` | `project()` |
| `ToolCallDeltaFrame` | `tool_call_delta` | `id, raw_name_delta, raw_arguments_delta` | `project()` |
| `ToolCallFrame` | `tool_call` | `id, name, arguments_json, truncated_from, raw_event_json` | `project()` (truncated) |
| `ToolResultFrame` | `tool_result` | `id, name, result, is_error, truncated_from, raw_event_json` | `project()` (truncated) |
| `ToolProgressFrame` | `tool_progress` | `id, progress, total, message` | `project()` |
| `ThoughtDeltaFrame` | `thought_delta` | `id, delta` | `project()` |
| `ThoughtFrame` | `thought` | `id, text` | `project()` |
| `TurnEndFrame` | `turn_end` | `turn` | `project()` |
| `DoneFrame` | `done` | `turns, response_source, response_text, inference_ids` | `project()` |
| `ErrorFrame` | `error` | `message, detail` | `project()` / `as_terminal_safe()` |
| `StatusFrame` | `status` | `text, server, tool` | `with_status()` |
| `ElicitationRequestFrame` | `elicitation_request` | `id, message, response_type, schema_json` | consumer, out-of-band |
| `ElicitationResolvedFrame` | `elicitation_resolved` | `id, action` | consumer, out-of-band |

Codec: `frame_to_dict(frame)` → `{"v": WIRE_VERSION, "kind": <kind>,
**fields}` (tuples encode as lists); `frame_from_dict(data)` rebuilds
the exact frame or raises `ValueError`. Every variant round-trips
through `json.dumps` / `json.loads`.

### Error semantics

Both failure paths produce the **same** frozen `ErrorFrame`:

- A terminal `RunErrorEvent` (`Run.events(on_error="event")`) maps
  through `project()`.
- An exception raised while draining maps through `as_terminal_safe()`.

`detail["exc_class"]` always carries the exception class name.
`MaxTurnsExceededError` is special-cased to the frozen public token:

```python
ErrorFrame(message="max_turns_exceeded",
           detail={"max_turns": 20, "exc_class": "MaxTurnsExceededError"})
```

A recoverable `ErrorEvent` also maps to `kind="error"` (verbatim, no
`exc_class`) — whether the stream continues is answered by whether more
frames follow, and `done` remains the only authoritative success
terminator.

## Stages

All stages are pure async generators `AsyncIterator → AsyncIterator`;
compose the ones you need, in this order:

```python
run = agent.invoke(query)
frames = as_terminal_safe(project(run.events()))
frames = with_status(frames, known_servers=("fred", "edgar"))
frames = with_text_fallback(frames)
async for name, data in public_allowlist(frames):
    await sse.send(event=name, data=json.dumps(data, ensure_ascii=False))
```

> **Consume unfiltered.** Feed the wire layer `run.events()` without
> selectors: with `on_error="event"` the terminal `RunErrorEvent` is
> subject to selector filtering like any other event, and a filtered
> stream would lose its terminal frame.

- **`project(events)`** — stateless event→frame translation. Tool
  arguments / results / raw-event JSON are byte-truncated here (8 KiB /
  8 KiB / 16 KiB, `…` indicator) so every downstream projection
  consumes the same bounded strings.
- **`as_terminal_safe(frames)`** — a raised exception becomes one
  terminal `ErrorFrame` (see above); `asyncio.CancelledError`
  propagates — cancellation is not a wire event.
- **`with_status(frames, *, known_servers=())`** — intercalates
  `StatusFrame` hints after their triggers: `"Calling {name}…"` per
  tool call (identity split against `known_servers`), `"Thinking…"`
  once per contiguous reasoning block, suppressed after any tool call;
  the terminal `DoneFrame`/`ErrorFrame` re-arms.
- **`with_text_fallback(frames)`** — the reasoning fallback +
  never-blank belt: when a `DoneFrame` arrives and no non-blank
  `TextDeltaFrame` streamed this run, synthesises a
  `TextDeltaFrame` + `TextEndFrame` pair before it — from
  `response_text` when non-blank, else from `EMPTY_TURN_RETRY_TEXT`.
  Block id: last inference id, else `FALLBACK_BLOCK_ID`. The tracked
  state re-arms at run boundaries (`DoneFrame`, *terminal*
  `ErrorFrame` — recognised by `detail["exc_class"]` — and
  `InitFrame`); a recoverable error passthrough never resets it.
- **`public_allowlist(frames)`** — the privacy guard and the terminal
  stage: yields `PublicEvent(name, data)` pairs under exactly eight
  frozen names. Tool activity, reasoning, and turn markers are
  **dropped** — arguments, results, and thought content never reach a
  public client.

## The public wire (frozen)

Exactly eight event names, with payload shapes frozen as production
emits them (including which payloads carry a `"type"` field):

| Name | Payload |
|---|---|
| `init` | `{"conversation_id"}` |
| `chunk` | `{"type", "content", "message_id"}` |
| `end` | `{"type", "id"}` |
| `status` | `{"text", "server", "tool"}` |
| `error` | `{"message", "detail"}` |
| `done` | `{"type", "id", "turns", "response_source"}` |
| `elicitation_request` | `{"type", "id", "message", "response_type", "schema"}` |
| `elicitation_resolved` | `{"type", "id", "action"}` |

Semantics consumers must honour:

- **`done` is the only authoritative turn end.** `end` fires per text
  block; a multi-turn flow emits several. Closing on the first `end`
  truncates tool-turns.
- **Elicitation typing** is by schema introspection, never class name:
  `response_type` is `null` for binary approval or a scalar type name
  (`"bool"`, …); `schema` is the re-parsed requested-schema object
  (`{}` when none, and malformed input degrades to `{}` rather than
  tearing the stream down).
- The golden for all of this is
  `tests/test_wire/test_public_golden.py` — a dict-level port of the
  production byte-compat gate. **If it fails, fix the stages, never
  the snapshot.**

## Migrating a hand-rolled adapter

A consumer that today owns a mirror union + codec + projector migrates
seam by seam, keeping its transport:

1. Replace the mirror frame union with `t0_agents.wire` frames and the
   hand-rolled codec with `frame_to_dict` / `frame_from_dict` (wrap the
   dict however the transport needs — e.g. a Redis stream entry).
2. Replace the event→frame mapper with `project()` and delete the
   worker's `try/except`-to-error-frame block in favour of
   `as_terminal_safe()`.
3. Replace the status translator with `with_status(known_servers=...)`
   (any product-specific attribution stays a thin consumer wrapper).
4. Replace the public projector's fallback + allowlist with
   `with_text_fallback()` + `public_allowlist()`, keeping only the
   byte serialiser (`event:`/`data:`/`id:` lines) at the edge.
5. Keep your byte-compat golden running through the migration — the
   library's dict-level golden guarantees the payloads; your golden
   guarantees the bytes.

## See also

- [docs/streaming.md](streaming.md) — the event taxonomy and `Run.events()`.
- [docs/glow/streaming-and-events.md](glow/streaming-and-events.md) —
  glow's `on_error="event"` default and selector caveats.
- `tests/test_wire/test_public_golden.py` — the frozen public golden.
