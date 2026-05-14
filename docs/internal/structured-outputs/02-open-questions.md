# Structured Outputs — Open Questions

> Questions raised during the planning phase.  Resolution happens
> during the milestone that needs the answer.  A resolved question
> includes a citation (file:line) when the resolution is source-
> validated.
>
> The `/structured-milestone <N>` command instructs the implementer
> to copy any answer they discover during implementation back into
> this file before writing their dev-log.

---

## Q1. Does `tensorzero.inference()` return distinct Python classes for `chat` vs `json` functions?

**Why this matters:** the implementation of `run_structured_inference`
needs to know whether to call a different SDK method, branch on a
union return type, or use the same call with `isinstance(...)`
narrowing.  Affects M1.

**Hypothesis:** the SDK exposes a single `inference(...)` method that
returns a union (`ChatInferenceResponse | JsonInferenceResponse`).

**To verify:** read
`.venv/lib/python*/site-packages/tensorzero/tensorzero.pyi` and
`/types.py` for the `inference` overloads.  Look at the response
classes' attributes.

**Status:** resolved during M1 (2026-05-14).

**Resolution:** `AsyncTensorZeroGateway.inference(...)` returns
`Union[InferenceResponse, AsyncIterator[InferenceChunk]]` where
`InferenceResponse = Union[ChatInferenceResponse,
JsonInferenceResponse]` (`tensorzero/types.py:262`).
`JsonInferenceResponse` (`tensorzero/types.py:229–238`) carries
`inference_id`, `episode_id`, `variant_name`, `output:
JsonInferenceOutput`, `usage: Usage`, and optional `raw_usage`,
`raw_response`, `finish_reason`, `original_response`.
`JsonInferenceOutput` (`tensorzero/types.py:209–212`) has `raw:
Optional[str]` and `parsed: Optional[Dict[str, Any]]`.  All three
classes are exported from top-level `tensorzero`
(`tensorzero/__init__.py:243-244`).  `run_structured_inference()`
narrows the union with `isinstance(response, JsonInferenceResponse)`
and raises `RuntimeError` if a `ChatInferenceResponse` came back
(caller pointed us at a chat function).

---

## Q2. Does TensorZero validate `parsed` against `output_schema`, or only `JSON.parse(raw)`?

**Why this matters:** if TZ runs schema validation, then `parsed`
being a non-null object means it is schema-conforming and we do not
need to re-validate.  If TZ only parses JSON, our retry layer should
optionally validate against `output_schema` to catch
structurally-valid-but-non-conforming outputs.

The TensorZero guide §11.15 explicitly flags this as undocumented.

**Hypothesis:** TZ runs schema validation when the provider supports
strict mode and is given the schema; otherwise it parses only.  In
practice, this means `parsed is not None` means "valid JSON",
not necessarily "schema-conforming".

**To verify:** read the source of
`.venv/.../tensorzero/`'s response materialisation.  Try a test
with an output schema requiring a field and a model that omits it —
inspect the resulting `parsed` and `raw`.

**Status:** open.  Resolution expected during M3 (the retry layer's
contract depends on this).

---

## Q3. How is `output_schema: Path` resolved?

**Why this matters:** the user can pass either an inline mapping or
a path.  If a path, we need to load it.  Three reasonable choices:

1. Relative to the current working directory.
2. Relative to the `tensorzero.toml` location.
3. Relative to the caller's module file (`inspect.getfile(...)`).

TensorZero itself resolves `[functions.x] output_schema = "..."`
**relative to the configuration file** (guide §3.1, configuration
reference).

**Hypothesis:** for parity with TensorZero's own behaviour, resolve
relative to `cwd` (since t0-agents has no knowledge of a tensorzero.toml
location at the agent layer — the gateway is already running) but
support absolute paths.  Document explicitly that the recommended
practice is absolute paths.

**To verify:** decide during M1 based on what feels least surprising.
Document in `docs/structured-outputs.md`.

**Status:** resolved during M1 (2026-05-14).

**Resolution:** relative `output_schema: Path` is resolved against the
current working directory (`Path.cwd()`), matching the hypothesis.
Absolute paths are loaded verbatim.
`docs/structured-outputs.md` documents the recommendation to use
absolute paths in production code.  Implementation lives in
`src/t0_agents/core/config.py` (`__post_init__` block resolving
`output_schema: Path`).

---

## Q4. Retry hint phrasing — short and corrective vs verbose with schema text

**Why this matters:** the retry layer (M3) re-emits an inference
with a user-turn hint message ("your previous response was not
valid JSON; please return only valid JSON matching the schema").
Two variants to compare:

- **Short:** `"Your previous response was not valid JSON matching the schema. Return ONLY a valid JSON object."`
- **Verbose:** Above + a re-statement of the schema (potentially long,
  costs tokens).

**Hypothesis:** short is usually enough for capable models; verbose
helps weak models but spends tokens.  Default to short; allow the
caller to override via a hook (deferred — not in M3 scope unless
it surfaces as a need).

**Status:** open.  Resolution expected during M3.

---

## Q5. Should `permissive_parse` accept JSON arrays at the top level?

**Why this matters:** some schemas have `"type": "array"` at the
root, so a balanced-bracket scan should also handle `[...]`, not
just `{...}`.

**Hypothesis:** yes.  Try `{...}` first, then `[...]`, then fenced
code blocks.  Top-level scalars (`"a string"`, `true`, `42`) are
out of scope — those are valid JSON but rarely a meaningful
"structured output".

**Status:** open.  Resolution expected during M3.

---

## Q6. Should the retry loop bound by wall-clock time, not just attempt count?

**Why this matters:** a single inference can take 10–30 seconds for
large models.  `max_json_retries = 2` could mean a 60-second
worst-case latency.

**Hypothesis:** for v1, attempt-count-only.  Wall-clock budgets are a
familiar problem with prior art (`max_cost` in M3 of the agentic-feature
roadmap shipped per-run cost ceilings); a `max_json_retry_seconds`
field could be added in a follow-up if real workloads demand it.

**Status:** open.  Decision pending M3 implementation experience.

---

## Q7. Does the `Pydantic` model integration belong in v1?

**Why this matters:** instead of `parsed: Any`, many callers want
`parsed: TheirPydanticModel`.  Some other LLM frameworks (PydanticAI,
instructor) make this central.

**Hypothesis:** **defer**.  The contract ships `parsed: Any | None`
in M1, and a wrapper that calls `MyModel.model_validate(result.parsed)`
is a one-liner for the caller.  Building a `pydantic_output_model`
config field would couple `core/` to `pydantic`, which violates
"new third-party deps live behind an extra".

**Status:** decided.  Resolution: out of scope for this roadmap;
add a `docs/structured-outputs.md` recipe showing the one-liner
in M2.

---

## Q8. How does the retry loop preserve `episode_id` semantics?

**Why this matters:** each TensorZero inference creates an episode (or
continues one).  Our retry calls `inference(...)` 1–3 times for the
same logical operation.  All retries must use the same `episode_id`
so they show up as a continuous conversation in ClickHouse.

**Hypothesis:** the first inference passes `episode_id` from the
caller (or `None` if fresh).  Once TZ returns the first
`StructuredOutput`, we capture its `episode_id` and pass it to
subsequent retries.  Each retry adds a user-turn hint message and
the prior model response as an assistant turn (so the model sees
its own bad output and the correction request).

**Status:** open.  Resolution expected during M3.

---

## Q9. How does memory persist a multi-attempt JSON run?

**Why this matters:** a JSON run can produce N inferences (initial +
retries).  Memory backends save one `MemoryRecord` per call.  The
record's `agent_response` should be the **final** parseable JSON
(or the last `output.raw` if all retries failed).  The
`inference_ids` list should include every inference id, in order,
for full traceability.

**Hypothesis:** yes — the `MemoryRecord` written at the end of a JSON
run has `agent_response = final_output.raw`, `inference_ids = [id_0,
id_1, ..., id_N]`, `turns = 1 + json_parse_attempts`.

**Status:** open.  Resolution expected during M3.

---

## Q10. Provider auto-detection for safety warnings (Anthropic extended thinking)

**Why this matters:** the guide §11.4 says Anthropic's extended-thinking
models are incompatible with `json_mode != "strict" | "off"`.  We
cannot enforce this in `Agent.__init__` because the SDK does not
expose the configured provider.  Best we can do is `log.warning` when
we detect a likely combination at runtime.

**Hypothesis:** punt — log a generic warning at run start when
`json_mode in {"on", "tool"}` advising the caller to verify
provider compatibility.  This is no worse than what TensorZero itself
documents.  No auto-detection.

**Status:** open.  Resolution expected during M4.

---

## Q11. Does `dryrun=True` flow through `run_structured_inference`?

**Why this matters:** today's `run_agent` accepts `dryrun: bool = False`.
The structured helper should match for parity.  Affects M1's helper
signature.

**Hypothesis:** yes.  `run_structured_inference` accepts `dryrun=False`
default; when `True`, TZ does not store the inference but still
returns a response — behaves like a "free" extraction call useful for
schema validation tests.

**Status:** resolved during M1 (2026-05-14).

**Resolution:** the SDK exposes `dryrun: Optional[bool]` as a
top-level kwarg on `AsyncTensorZeroGateway.inference(...)`
(`tensorzero/tensorzero.pyi:1048`).  `run_structured_inference()`
accepts `dryrun: bool = False` and forwards verbatim, matching
`run_agent`'s parity.  Test coverage:
`tests/test_core/test_structured.py::test_threads_every_kwarg_into_inference_call`.

---

## Q12. Schema validation in `permissive_parse`

**Why this matters:** if M3 ships permissive parsing, the extracted
value may be valid JSON but not schema-conforming.  Should we run
JSON-Schema validation in Python after extraction to surface the
discrepancy via `parse_strategy`?

**Hypothesis:** no — JSON-Schema validation in Python requires a new
dep (`jsonschema`).  Surface `parsed` as-is.  The caller validates
against their schema in their own code (often via Pydantic, which
they already pull).

**Status:** decided.  Out of scope.  Document the recommendation
in `docs/structured-outputs.md`.

---

## Q13. Backwards-compat for existing `Agent.run` callers with `system_arguments` and a JSON function

**Why this matters:** `system_arguments` is a v0.3.0 feature for
memory rendering and template variables.  For JSON functions, the
caller may pass `system_arguments` to render the prompt template.
This works without changes — the structured helper threads them through
exactly like `run_agent` does.

**Status:** decided.  No special handling.  Documented in M1.
