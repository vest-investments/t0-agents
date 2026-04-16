# TensorZero — Comprehensive Developer Guide

> A complete reference for engineers building on TensorZero, from first inference to production-grade LLMOps,
> including architecture diagrams, data-flow sequences, and notes relevant to building declarative agent abstractions.

---

## Table of Contents

1. [What is TensorZero?](#1-what-is-tensorzero)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Concepts](#3-core-concepts)
4. [Infrastructure & Deployment](#4-infrastructure--deployment)
5. [Level 1 — Basic: First Inference](#5-level-1--basic-first-inference)
6. [Level 2 — Intermediate: Functions, Variants, Templates & Feedback](#6-level-2--intermediate-functions-variants-templates--feedback)
7. [Level 3 — Advanced: Experimentation, Episodes, Optimization & Evaluations](#7-level-3--advanced-experimentation-episodes-optimization--evaluations)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [API Quick Reference](#9-api-quick-reference)
10. [The Static Configuration Contract (Critical for Abstraction Builders)](#10-the-static-configuration-contract-critical-for-abstraction-builders)
11. [Summary Matrix: What Lives Where](#11-summary-matrix-what-lives-where)

---

## 1. What is TensorZero?

TensorZero is an **open-source, self-hosted LLMOps stack** built in Rust. Its design philosophy centers on a single idea: the performance of an LLM application is a function of data collected in production — and every layer of the stack should feed a continuous improvement flywheel.

The five pillars:

| Pillar | What it does |
|---|---|
| **Gateway** | Unified proxy to every major LLM provider. Sub-millisecond overhead (`<1ms P99` at 10 000 QPS). |
| **Observability** | Every inference and its downstream feedback lands in your own database (ClickHouse or Postgres). |
| **Optimization** | Pre-built and custom recipes turn that data into fine-tuned models, better prompts, or smarter retrieval. |
| **Evaluations** | Benchmark inferences or full multi-step workflows with heuristics or LLM judges. |
| **Experimentation** | Built-in static and adaptive A/B testing with variant routing, fallbacks, and retries. |

**Key architectural fact for abstraction builders:** the gateway is a *stateless Rust proxy*. Its entire behavior — every function, variant, model, metric, tool, and experiment — is declared in a **`tensorzero.toml`** file that is loaded once at startup and **cannot be modified at runtime**. This is the immovable boundary around which any dynamic abstraction layer must work.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  Your app / agent / abstraction bridge                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ OpenAI SDK   │  │ TensorZero SDK   │  │  Raw HTTP (curl) │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
└─────────┼────────────────── ┼────────────────────-─┼────────────┘
          │                   │                      │
          ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              TENSORZERO GATEWAY  :3000  (Rust)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  tensorzero.toml  (static, loaded at startup)            │   │
│  │  • [functions]   • [models]   • [metrics]                │   │
│  │  • [tools]       • [gateway]  • [provider_types]         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  POST /inference           POST /openai/v1/chat/completions     │
│  POST /feedback            POST /v1/inferences/list_inferences  │
│  POST /batch_inference     GET  /status   GET /health           │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Variant    │  │  Template    │  │  Fallback / Retry      │  │
│  │  Sampler    │  │  Renderer    │  │  Router                │  │
│  │ (per episode│  │ (MiniJinja)  │  │ (model → variant lvl)  │  │
│  │  sticky)    │  └──────────────┘  └────────────────────────┘  │
│  └─────────────┘                                                │
└──────────┬──────────────────────────────────────────────────────┘
           │  routes to
           ▼
┌────────────────────────────────────────────────────────────────┐
│                  MODEL PROVIDERS                               │
│  OpenAI · Anthropic · Google · Fireworks · Together · vLLM    │
│  Azure · Bedrock · Mistral · Cohere · XAI · …                 │
└─────────────────────────────────────┬──────────────────────────┘
                                      │ async write-back
                                      ▼
┌────────────────────────────────────────────────────────────────┐
│              OBSERVABILITY DATABASE                            │
│  ClickHouse (recommended >100 RPS)  OR  Postgres               │
│  • ChatInference / JsonInference tables                        │
│  • BooleanMetricFeedback / FloatMetricFeedback tables          │
│  • DemonstrationFeedback / CommentFeedback tables              │
│  • EpisodeById aggregation view                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   TENSORZERO UI  :4000                         │
│  Inference explorer · Metric trends · Experiment dashboard     │
│  Optimization launcher · Evaluation viewer                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Concepts

### 3.1 Function

A **function** is the top-level unit of work — what your application *does* (e.g. `draft_email`, `extract_entities`, `answer_question`). Functions have a type:

- `chat` — returns unstructured text (one or more content blocks).
- `json` — returns structured output validated against a JSON Schema.

Functions own their input/output schemas, tool list, tool-choice policy, and experimentation configuration.

### 3.2 Variant

A **variant** is one concrete implementation of a function: a specific model, prompt template, temperature, and inference strategy. You can define many variants per function. At inference time the gateway picks one (according to the experimentation config) and keeps it **sticky** for the entire episode.

Variant types:

| Type | Description |
|---|---|
| `chat_completion` | Standard chat with any supported model. |
| `experimental_best_of_n` | Generate N candidates with sub-variants; a judge picks the best. |
| `experimental_mixture_of_n` | Generate N candidates across sub-variants; a fuser synthesizes one answer. |
| `experimental_dynamic_in_context_learning` | Embeds the input and injects the K nearest high-quality historical examples into the prompt at runtime. |

### 3.3 Model & Model Provider

A **model** (`[models.my_model]`) is an arbitrary name you assign to an LLM. A model can have multiple **providers** with a `routing` list that defines priority order for fallbacks:

```toml
[models.smart_model]
routing = ["primary_openai", "backup_azure"]

[models.smart_model.providers.primary_openai]
type = "openai"
model_name = "gpt-4.1"

[models.smart_model.providers.backup_azure]
type = "azure"
model_name = "gpt-4"
deployment_id = "gpt-4-prod"
```

Shorthand notation (`openai::gpt-4.1`) creates an implicit model inline without explicit `[models]` config.

### 3.4 Episode

An **episode** is a group of related inferences that together form one logical interaction (e.g. a multi-turn conversation or an agentic loop). The gateway returns an `episode_id` on every inference. You pass that same `episode_id` on subsequent calls to keep them in the same episode. Feedback can be attached at the episode level for end-to-end scoring.

Within an episode, the gateway ensures **variant consistency**: once a variant is selected for a function in episode X, all subsequent calls to that function in episode X use the same variant (unless a fallback forces a change).

### 3.5 Metric

A **metric** captures downstream outcome signal. Defined in `[metrics.name]`:

```toml
[metrics.task_completed]
type = "boolean"   # or "float"
level = "episode"  # or "inference"
optimize = "max"   # or "min"
```

Reserved metric names with special semantics:
- `demonstration` — the ideal output for an inference (used for SFT/DPO datasets).
- `comment` — free-text annotation attached to an inference or episode.

### 3.6 Tool

A **tool** is a JSON Schema–defined function the model can call. Defined in `[tools.tool_name]` and referenced in function config. Tools can also be provided dynamically at inference time via `additional_tools` when runtime definition is required.

---

## 4. Infrastructure & Deployment

### 4.1 Required Components

```
┌──────────────────────────────────────────────────────┐
│  Minimum viable stack                                │
│  ─────────────────────────────────────────────────── │
│  tensorzero/gateway    ← always required             │
│  + Postgres OR ClickHouse  ← for observability       │
│    (omit both if you only need the proxy)            │
│  + tensorzero/ui       ← optional, recommended       │
└──────────────────────────────────────────────────────┘
```

### 4.2 Docker Compose (Development)

```yaml
# docker-compose.yml  — development only, not for production
services:
  clickhouse:
    image: clickhouse:lts
    environment:
      CLICKHOUSE_USER: "chuser"
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: "1"
      CLICKHOUSE_PASSWORD: "chpassword"
    ports:
      - "8123:8123"
    healthcheck:
      test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      TENSORZERO_CLICKHOUSE_URL: "http://chuser:chpassword@clickhouse:8123/tensorzero"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      clickhouse:
        condition: service_healthy

  ui:
    image: tensorzero/ui
    environment:
      TENSORZERO_GATEWAY_URL: "http://gateway:3000"
    ports:
      - "4000:4000"
    depends_on:
      gateway:
        condition: service_healthy
```

### 4.3 Database Choice

| Scenario | Recommended DB |
|---|---|
| Development / low traffic | Postgres (simpler setup) |
| > 100 inferences/sec | ClickHouse (columnar, analytical) |
| Adaptive A/B tests or auth | Postgres is **also** required alongside ClickHouse |

### 4.4 Environment Variables

| Variable | Purpose |
|---|---|
| `TENSORZERO_CLICKHOUSE_URL` | ClickHouse connection string |
| `TENSORZERO_POSTGRES_URL` | Postgres connection string |
| `OPENAI_API_KEY` | (or any provider key) |
| `TENSORZERO_CLICKHOUSE_CLUSTER_NAME` | Required for replicated ClickHouse |

### 4.5 Health Endpoints

```
GET /status  →  { "status": "ok" }
GET /health  →  { "gateway": "ok", "clickhouse": "ok" }   (checks DB connectivity)
```

---

## 5. Level 1 — Basic: First Inference

### 5.1 Minimal Config

```toml
# config/tensorzero.toml

[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
```

### 5.2 Calling via OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="not-used"  # gateway handles credentials
)

response = client.chat.completions.create(
    model="tensorzero::function_name::generate_haiku",
    messages=[{"role": "user", "content": "Write a haiku about software."}]
)

print(response.choices[0].message.content)
# TensorZero adds these to the response:
print(response.id)          # inference_id (UUID)
print(response.episode_id)  # episode_id (UUID)
```

### 5.3 Calling via Native TensorZero API

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    response = t0.inference(
        function_name="generate_haiku",
        input={
            "messages": [
                {"role": "user", "content": "Write a haiku about software."}
            ]
        }
    )
    print(response.content[0].text)
    print(response.inference_id)
    print(response.episode_id)
```

### 5.4 Calling via curl

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku",
    "input": {
      "messages": [{"role": "user", "content": "Write a haiku about software."}]
    }
  }'
```

**Response structure (native API):**
```json
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "gpt_4o_mini",
  "content": [{"type": "text", "text": "Bytes cascade to light..."}],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 19,
    "cost": 0.0001
  }
}
```

### 5.5 Basic Sequence Diagram

```
Client                  TensorZero Gateway          OpenAI
  │                            │                       │
  │── POST /inference ────────>│                       │
  │   function_name:           │                       │
  │   "generate_haiku"         │                       │
  │                            │── validate request    │
  │                            │── select variant ─────│
  │                            │── render prompt       │
  │                            │── POST /chat/compl ──>│
  │                            │                       │── LLM ──┐
  │                            │                       │<─────────┘
  │                            │<── model response ────│
  │<── inference_id + content ─│                       │
  │    episode_id               │                       │
  │                            │── async DB write ──>  DB
  │                            │   (non-blocking)      │
```

---

## 6. Level 2 — Intermediate: Functions, Variants, Templates & Feedback

### 6.1 Prompt Templates (MiniJinja)

Templates decouple prompts from code and enable per-variant customization.

**File layout:**
```
config/
  tensorzero.toml
  functions/
    draft_email/
      email_schema.json             ← input JSON Schema
      prompt_v1/
        system_template.minijinja   ← system prompt template
        user_template.minijinja     ← user prompt template
```

**`tensorzero.toml`:**
```toml
[functions.draft_email]
type = "chat"
schemas.system.path = "functions/draft_email/email_schema.json"

[functions.draft_email.variants.prompt_v1]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
templates.system.path = "functions/draft_email/prompt_v1/system_template.minijinja"
temperature = 0.4
```

**`system_template.minijinja`:**
```jinja
You are an email drafting assistant for {{ company_name }}.
Write in a {{ tone }} tone. Be concise and professional.
```

**`email_schema.json`:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "company_name": {"type": "string"},
    "tone": {"type": "string", "enum": ["formal", "casual", "friendly"]}
  },
  "required": ["company_name", "tone"],
  "additionalProperties": false
}
```

**Calling with template variables:**
```python
response = client.chat.completions.create(
    model="tensorzero::function_name::draft_email",
    messages=[
        {
            "role": "system",
            "content": [{
                "type": "tensorzero::template",
                "name": "system",
                "arguments": {"company_name": "Acme Corp", "tone": "formal"}
            }]
        },
        {"role": "user", "content": "Draft a follow-up email to client John Doe."}
    ]
)
```

### 6.2 Structured Output (JSON Functions)

```toml
[functions.extract_entities]
type = "json"
output_schema = "functions/extract_entities/output_schema.json"

[functions.extract_entities.variants.baseline]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
templates.system.path = "functions/extract_entities/baseline/system_template.minijinja"
json_mode = "strict"
```

**Output schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "people": {"type": "array", "items": {"type": "string"}},
    "organizations": {"type": "array", "items": {"type": "string"}},
    "locations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["people", "organizations", "locations"]
}
```

### 6.3 Tool Use

```toml
[functions.weather_bot]
type = "chat"
tools = ["get_temperature"]
tool_choice = "auto"   # none | auto | required | { specific = "tool_name" }

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "tools/get_temperature.json"
```

**`get_temperature.json`:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "location": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
  },
  "required": ["location"]
}
```

### 6.4 Multiple Models / Multi-Provider Routing

```toml
[models.production_llm]
routing = ["openai_primary", "azure_backup"]

[models.production_llm.providers.openai_primary]
type = "openai"
model_name = "gpt-4.1"

[models.production_llm.providers.azure_backup]
type = "azure"
model_name = "gpt-4"
deployment_id = "gpt4-prod-eastus"
endpoint = "https://myinstance.openai.azure.com"
api_key_location = "env::AZURE_OPENAI_API_KEY"
```

### 6.5 Retries & Fallbacks

Three levels of resilience:

```toml
# Level 1: Provider routing (model level)
[models.my_model]
routing = ["provider_a", "provider_b"]   # tries provider_b if provider_a fails

# Level 2: Variant retries
[functions.draft_email.variants.prompt_v1]
type = "chat_completion"
model = "openai::gpt-4.1"
retries = { num_retries = 3, max_delay_s = 10 }

# Level 3: Variant fallbacks (function level)
[functions.draft_email.experimentation]
type = "static"
candidate_variants = ["fast_variant", "standard_variant"]
fallback_variants = ["safe_fallback_variant"]
```

### 6.6 Metrics & Feedback

Define metrics in config:
```toml
[metrics.draft_accepted]
type = "boolean"
level = "inference"
optimize = "max"

[metrics.user_rating]
type = "float"
level = "episode"
optimize = "max"
```

Send feedback after an inference:
```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    # Boolean metric at inference level
    t0.feedback(
        metric_name="draft_accepted",
        inference_id="00000000-0000-0000-0000-000000000000",
        value=True
    )

    # Float metric at episode level
    t0.feedback(
        metric_name="user_rating",
        episode_id="11111111-1111-1111-1111-111111111111",
        value=0.92
    )

    # Demonstration (ideal output for SFT/DPO)
    t0.feedback(
        metric_name="demonstration",
        inference_id="00000000-0000-0000-0000-000000000000",
        value="The ideal email content here..."
    )

    # Comment (free-text annotation)
    t0.feedback(
        metric_name="comment",
        inference_id="00000000-0000-0000-0000-000000000000",
        value="The tone was too casual for this client."
    )
```

### 6.7 Multi-Step Inference with Episodes

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    # Step 1 — no episode_id yet; gateway creates one
    step1 = t0.inference(
        function_name="generate_query",
        input={"messages": [{"role": "user", "content": "Find recent AI papers"}]}
    )
    episode_id = str(step1.episode_id)

    # Step 2 — join the same episode
    step2 = t0.inference(
        function_name="generate_answer",
        episode_id=episode_id,  # same episode!
        input={"messages": [{"role": "user", "content": f"Context: {search_results}. Summarize."}]}
    )

    # Feedback at episode level (covers the whole workflow)
    t0.feedback(
        metric_name="task_success",
        episode_id=episode_id,
        value=True
    )
```

---

## 7. Level 3 — Advanced: Experimentation, Episodes, Optimization & Evaluations

### 7.1 Static A/B Testing

```toml
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_mini]
type = "chat_completion"
model = "openai::gpt-4.1-mini"

[functions.draft_email.variants.claude_haiku]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"

[functions.draft_email.variants.llama_local]
type = "chat_completion"
model = "my_llama_model"

# Explicit 50/50 between two variants; llama_local excluded from A/B
[functions.draft_email.experimentation]
type = "static"
candidate_variants = ["gpt_mini", "claude_haiku"]
fallback_variants = ["llama_local"]   # only used if both above fail
```

Weighted canary rollout:
```toml
[functions.draft_email.experimentation]
type = "static"
candidate_variants = {"gpt_mini" = 0.9, "claude_haiku" = 0.1}
```

### 7.2 Adaptive A/B Testing (Bandit Algorithm)

Adaptive experiments automatically route more traffic to better-performing variants and stop when a winner is identified.

```toml
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.prompt_v1]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
templates.system.path = "functions/draft_email/prompt_v1/system.minijinja"

[functions.draft_email.variants.prompt_v2]
type = "chat_completion"
model = "openai::gpt-4.1-mini"
templates.system.path = "functions/draft_email/prompt_v2/system.minijinja"

[functions.draft_email.experimentation]
type = "adaptive"
candidate_variants = ["prompt_v1", "prompt_v2"]
metric = "draft_accepted"          # must be defined in [metrics]
# Optional tuning:
# epsilon = 0.1                    # sensitivity threshold (default: 0.1)
# delta = 0.05                     # acceptable error rate (default: 0.05)
# update_period_s = 300            # how often to recompute weights (default: 300)
```

**Required:** Postgres must be available for adaptive A/B tests (it backs the background weight computation job).

### 7.3 Namespace-Scoped Experiments

Route different customers or tenants to separate experiments:

```toml
[functions.draft_email.experimentation]
type = "static"
candidate_variants = ["default_gpt", "default_claude"]

[functions.draft_email.experimentation.namespaces.acme_corp]
type = "static"
candidate_variants = ["acme_gpt", "acme_claude"]
```

Sending a namespaced inference:
```python
t0.inference(
    function_name="draft_email",
    namespace="acme_corp",
    input={...}
)
```

### 7.4 Inference-Time Optimizations

#### Best-of-N Sampling

```toml
[functions.draft_email.variants.best_of_3]
type = "experimental_best_of_n"
candidates = ["prompt_v1", "prompt_v1", "prompt_v2"]  # 3 total, 2 from v1

[functions.draft_email.variants.best_of_3.evaluator]
model = "openai::gpt-4.1-mini"
# TensorZero auto-constructs the judge prompt
```

#### Mixture-of-N Sampling

```toml
[functions.draft_email.variants.mixture_3]
type = "experimental_mixture_of_n"
candidates = ["prompt_v1", "prompt_v1", "prompt_v2"]

[functions.draft_email.variants.mixture_3.fuser]
model = "openai::gpt-4.1"
# System prompt for the fuser that synthesizes all candidates
```

#### Dynamic In-Context Learning (DICL)

```toml
[functions.draft_email.variants.dicl_v1]
type = "experimental_dynamic_in_context_learning"
model = "openai::gpt-4.1-mini"
# Embeddings model used to find similar historical examples
embedding_model = "openai::text-embedding-3-small"
k = 5   # number of examples to inject
```

### 7.5 Supervised Fine-Tuning (SFT)

```python
from tensorzero import TensorZeroGateway, OpenAISFTConfig

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    # 1. Build a curated dataset from historical inferences + feedback
    #    (done via UI or API: POST /v1/datasets/{name}/datapoints)

    # 2. Configure the SFT job
    optimization_config = OpenAISFTConfig(
        model="gpt-4.1-2025-04-14",
    )

    # 3. Launch
    job_handle = t0.experimental_launch_optimization_workflow(
        function_name="extract_entities",
        template_variant_name="baseline",
        dataset_name="ner_gold_dataset",
        optimizer_config=optimization_config,
        val_fraction=0.2,
    )

    # 4. Poll for completion
    import time
    while True:
        status = t0.experimental_get_optimization_workflow_status(job_handle)
        if status.status in ("completed", "failed"):
            break
        time.sleep(30)

    # 5. Add the resulting fine-tuned model to tensorzero.toml as a new variant
    #    (done manually — config is static)
```

**Supported SFT providers:** OpenAI, GCP Vertex AI (Gemini), Fireworks AI, Together AI.

### 7.6 Evaluations

Run offline evaluations against a dataset:

```bash
docker compose run --rm evaluations \
    --function-name write_haiku \
    --evaluator-names valid_haiku,exact_match \
    --dataset-name haiku_dataset \
    --variant-name gpt_4o \
    --concurrency 5
```

Evaluator types (configured in `tensorzero.toml`):
- **Exact match** — deterministic string comparison.
- **Heuristic** — custom scoring functions.
- **LLM judge** — another TensorZero function that scores outputs.

Workflow evaluations link multiple inferences in an episode, score the whole episode, and compare across runs in the UI.

---

## 8. Data Flow Diagrams

### 8.1 Single Inference Flow (Full Detail)

```
CLIENT                        GATEWAY                         PROVIDER            DATABASE
  │                              │                               │                    │
  │─── POST /inference ─────────>│                               │                    │
  │    {function_name,           │                               │                    │
  │     input, episode_id?}      │                               │                    │
  │                              │─ 1. Validate input schema     │                    │
  │                              │─ 2. Sample variant ──────────────────────────────> │
  │                              │    (or sticky in episode)     │                    │
  │                              │─ 3. Render MiniJinja templates│                    │
  │                              │─ 4. Build provider request    │                    │
  │                              │─── POST provider API ────────>│                    │
  │                              │                               │── LLM inference ──>│
  │                              │<── raw model response ────────│                    │
  │                              │─ 5. Parse & validate output   │                    │
  │                              │─ 6. Build TZ response         │                    │
  │<── {inference_id,           ─│                               │                    │
  │     episode_id, content,     │                               │                    │
  │     variant_name, usage}     │                               │                    │
  │                              │─ 7. Async write to DB ──────────────────────────>  │
  │                              │    (non-blocking, fire+forget)│                    │
```

### 8.2 Episode & Feedback Flow (Multi-Step Agent)

```
CLIENT                           GATEWAY                         DATABASE
  │                                 │                               │
  │─── Inference 1 (no episode_id) >│                               │
  │<── { inference_id_1,           ─│                               │
  │      episode_id: "ep-abc" }     │                               │
  │                                 │── async write ChatInference──>│
  │                                 │                               │
  │─── Inference 2 (episode_id=    >│                               │
  │     "ep-abc")                   │─ Same variant as Inference 1  │
  │<── { inference_id_2,           ─│   (sticky within episode)     │
  │      episode_id: "ep-abc" }     │                               │
  │                                 │── async write ChatInference──>│
  │                                 │                               │
  │   [application runs tool...]    │                               │
  │                                 │                               │
  │─── Inference 3 (episode_id=    >│                               │
  │     "ep-abc")                   │                               │
  │<── { inference_id_3, ... }     ─│                               │
  │                                 │                               │
  │─── POST /feedback ─────────────>│                               │
  │    { metric_name: "success",    │                               │
  │      episode_id: "ep-abc",      │── write FloatMetricFeedback──>│
  │      value: 1.0 }               │                               │
  │<── { feedback_id } ────────────-│                               │
```

### 8.3 Adaptive A/B Test Flow

```
GATEWAY                     POSTGRES                   BACKGROUND JOB
  │                              │                           │
  │─── Experiment starts         │                           │
  │    (50/50 variant_a/b)       │                           │
  │                              │                           │
  │─── Inference → variant_a ───>│ write inference row       │
  │─── Inference → variant_b ───>│ write inference row       │
  │   [feedback arrives]         │                           │
  │─── Feedback → episode ──────>│ write metric feedback     │
  │                              │                           │
  │                              │<── read metrics ──────────│
  │                              │    every update_period_s  │
  │                              │─── compute new weights ──>│
  │<── update routing weights ───│                           │
  │    (variant_a: 70%, b: 30%)  │                           │
  │                              │                           │
  │   [winner identified]        │                           │
  │<── routing: variant_a: 100% ─│                           │
  │    (experiment concluded)    │                           │
```

### 8.4 Optimization (SFT) Flow

```
CLIENT / UI            GATEWAY              PROVIDER FINE-TUNER        DATABASE
  │                       │                        │                       │
  │─ Build dataset ──────>│                        │                       │
  │  from inferences      │<── query inferences ───────────────────────────│
  │  + feedback           │    with good metrics   │                       │
  │                       │                        │                       │
  │─ Launch SFT job ─────>│                        │                       │
  │                       │─ render training data  │                       │
  │                       │  (apply templates)     │                       │
  │                       │─── upload dataset ────>│                       │
  │                       │─── start FT job ──────>│                       │
  │<── job_handle ────────│                        │── train model ────────│
  │                       │                        │                       │
  │─ Poll status ─────────│                        │                       │
  │<── "completed",       │                        │                       │
  │    model_id: "ft-xyz" │                        │                       │
  │                       │                        │                       │
  │─ Add to tensorzero.toml manually:              │                       │
  │  [models.my_ft_model.providers.openai]         │                       │
  │  model_name = "ft-xyz"                         │                       │
```

### 8.5 Client-Side View: All Entry Points

```
                    ┌───────────────────────────────────┐
                    │        tensorzero/gateway          │
                    │             :3000                  │
                    └───────────────────────────────────┘
                                    ▲
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           │                        │                        │
  ┌────────┴──────┐     ┌───────────┴──────────┐   ┌────────┴───────┐
  │ INFERENCE     │     │ FEEDBACK              │   │ OBSERVABILITY  │
  │               │     │                       │   │                │
  │ POST /inference     │ POST /feedback         │   │ POST /v1/      │
  │ POST /openai/ │     │                       │   │  inferences/   │
  │  v1/chat/     │     │ body: {               │   │  list_inferences│
  │  completions  │     │  metric_name,         │   │                │
  │               │     │  inference_id |       │   │ POST /v1/      │
  │ POST /batch_  │     │  episode_id,          │   │  inferences/   │
  │  inference    │     │  value }              │   │  get_inferences│
  └───────────────┘     └───────────────────────┘   └────────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ OPTIMIZATION & EVALUATION      │
                    │                               │
                    │ POST /v1/datasets/...          │
                    │ experimental_launch_           │
                    │  optimization_workflow()       │
                    │                               │
                    │ CLI: tensorzero/evaluations   │
                    └───────────────────────────────┘
```

---

## 9. API Quick Reference

### 9.1 POST /inference

| Field | Required | Notes |
|---|---|---|
| `function_name` | yes | Must be declared in `tensorzero.toml` |
| `input.messages` | yes | Array of `{role, content}` objects |
| `input.system` | no | System message (string or template block) |
| `episode_id` | no | Provide to join an existing episode |
| `stream` | no | `true` for SSE streaming |
| `tags` | no | Flat `{"key": "value"}` object |
| `additional_tools` | no | Dynamic tool definitions at runtime |
| `variant_name` | no | Force a specific variant (bypasses sampling) |
| `dryrun` | no | Execute but don't persist to DB |

### 9.2 POST /openai/v1/chat/completions

Drop-in OpenAI-compatible endpoint. Encode function name as:
```
model = "tensorzero::function_name::MY_FUNCTION"
```

Pass TensorZero-specific fields in `extra_body`:
```python
extra_body={
    "tensorzero::episode_id": "11111111-...",
    "tensorzero::variant_name": "my_variant",  # force specific variant
}
```

### 9.3 POST /feedback

| Field | Required | Notes |
|---|---|---|
| `metric_name` | yes | Must be declared in `[metrics]`, or `comment`/`demonstration` |
| `value` | yes | `bool` for boolean metrics, `float` for float metrics, `string` for demonstrations/comments |
| `inference_id` | conditional | Required when `level = "inference"` |
| `episode_id` | conditional | Required when `level = "episode"` |
| `tags` | no | `{"key": "value"}` |
| `dryrun` | no | Debug mode |

---

## 10. The Static Configuration Contract (Critical for Abstraction Builders)

> This section is specifically for developers building layers *on top of* TensorZero, particularly declarative agent frameworks that want to expose a higher-level API.

### 10.1 What is Fixed at Startup

The `tensorzero.toml` file is the **immutable control plane**. It is loaded once when the gateway process starts. The following are all static and cannot change while the gateway is running:

| Entity | Where declared | Can change at runtime? |
|---|---|---|
| Functions (name, type, schemas, tools) | `[functions.*]` | ❌ No |
| Variants (model, template paths, params) | `[functions.*.variants.*]` | ❌ No |
| Models & providers | `[models.*]` | ❌ No |
| Metrics (name, type, level, optimize) | `[metrics.*]` | ❌ No |
| Tools (name, description, schema) | `[tools.*]` | ❌ No |
| Experimentation config | `[functions.*.experimentation]` | ❌ No |
| Template files (`.minijinja`) | File paths in variant config | ❌ No |

### 10.2 What is Dynamic at Runtime

These things happen at request time and require **no config reload**:

| Entity | How it's passed |
|---|---|
| Input variables (template args) | `input.messages[].content[].arguments` |
| Episode grouping | `episode_id` field |
| Namespace routing | `namespace` field |
| Dynamic tools | `additional_tools` field |
| Force variant | `variant_name` field |
| Feedback / metrics values | `POST /feedback` |
| Tags | `tags` field |

### 10.3 Implications for a Declarative Agent Framework

Because the TOML cannot be hot-reloaded, a declarative agent abstraction that wants to create "agents" dynamically must operate in one of these modes:

1. **Pre-declaration mode (recommended):** All possible functions, variants, models, and tools are declared upfront in the TOML. The dynamic layer selects which function/variant/episode to use at runtime by composing requests. The TOML becomes the "registry" of all agent capabilities.

2. **Gateway-per-tenant / config-generation mode:** Generate a `tensorzero.toml` per agent or tenant at bootstrap time, spin up a gateway instance (or reload via restart), and route traffic accordingly. The abstraction layer manages gateway lifecycle.

3. **Dynamic tools mode (limited):** Use `additional_tools` at inference time for tools that cannot be pre-declared. This is the one escape valve for fully dynamic behavior, but it bypasses the observability and schema validation benefits of pre-declared tools.

### 10.4 Config File Composition (Glob Support)

The gateway CLI accepts a glob pattern for config files, enabling you to split the TOML across files that are concatenated at startup:

```bash
gateway --config-file /app/config/**/*.toml
```

This means an abstraction layer can generate separate TOML files (e.g. one per function family or tenant) and have them merged. Paths within each file are resolved relative to that file's directory.

### 10.5 Variant as the Unit of Abstraction

For agent frameworks, the **variant** is the most granular control point:

- Each variant maps to exactly one model + prompt + inference strategy.
- Variants are selectable per-request via `variant_name` (bypassing sampling).
- Variants accumulate performance data in the DB, enabling optimization.
- New variants can only be deployed with a config change + gateway restart.

A declarative T0 agent might look like:

```toml
# agent "email_drafter" = function "draft_email"
# persona "formal" = variant "draft_email_formal_v1"
# persona "casual" = variant "draft_email_casual_v1"

[functions.draft_email]
type = "chat"

[functions.draft_email.variants.formal_v1]
type = "chat_completion"
model = "openai::gpt-4.1"
templates.system.path = "agents/email_drafter/formal/system.minijinja"

[functions.draft_email.variants.casual_v1]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"
templates.system.path = "agents/email_drafter/casual/system.minijinja"
```

Your bridge layer translates the declarative agent spec → function calls with the right `variant_name`.

---

## 11. Summary Matrix: What Lives Where

| Concern | Config Layer (TOML) | Request Layer (API) | Feedback Layer | Optimization Layer |
|---|---|---|---|---|
| Define a task | `[functions.name]` | — | — | — |
| Choose a model + prompt | `[functions.name.variants.v]` | `variant_name` to force | — | Add new variant after SFT |
| Input structure | `schemas.*.path` | `input.messages` | — | — |
| Output structure | `output_schema` | — | — | — |
| Route between models | `routing = [...]` | — | — | — |
| A/B test | `[experimentation]` | — | `metric_name + value` | Adaptive weights auto-update |
| Track outcomes | `[metrics.name]` | — | `POST /feedback` | Used for fine-tuning |
| Multi-step workflows | — | `episode_id` | `episode_id` on feedback | Episode-level scoring |
| Namespace isolation | `namespaces.*` | `namespace` field | — | — |
| Dynamic tool calls | `[tools.name]` (static) | `additional_tools` (dynamic) | — | — |

---

## Sources

All information in this document is drawn exclusively from the project documentation files:

- `index.md`, `index.yaml` — Overview and table of contents
- `quickstart.md` — Getting started guide
- `gateway-index.md` — Gateway feature summary
- `gateway-configure-functions-and-variants.md` — Functions and variants
- `gateway-configure-models-and-providers.md` — Model configuration
- `gateway-configuration-reference.md` — Complete configuration reference
- `gateway-create-a-prompt-template.md` — MiniJinja templating
- `gateway-api-reference.md` — API endpoints reference
- `gateway-guides.md` — Metrics, feedback, retries, inference-time optimizations
- `gateway-data-model.md` — ClickHouse/Postgres schema
- `observability.md` — Observability API
- `experimentation.md` — A/B testing and namespaces
- `optimization.md` — SFT, DPO, GEPA, DICL
- `evaluations.md` — Evaluation workflows
- `deployment.md` — Infrastructure deployment
- `faq.md` — Technical FAQ
- `comparison.md` — Comparisons with LangChain, LiteLLM, Kong, OpenPipe
- `vision-and-roadmap.md` — Project vision
