# Example 03: System Prompt

Demonstrates how to configure a TensorZero **system prompt** that guides an LLM's tool-use behavior.

## What is a system prompt in TensorZero?

TensorZero supports a `system_template` directive on function variants. When set, the gateway injects the contents of the referenced file as a system message at the start of every conversation. The template file uses the Minijinja engine, so you can include `{{ variable }}` interpolation if needed, but plain text works just as well.

## How does a system prompt affect tool-use behavior?

Without a system prompt, LLMs may exhibit several failure modes when using tools:

- **Asking for clarification** instead of calling a tool with sensible defaults.
- **Skipping tool selection entirely** and answering from parametric knowledge.
- **Retrying indefinitely** when a tool call fails.

The system prompt in this example (`config/functions/agent/baseline/system.txt`) addresses these by instructing the model to:

1. Prefer calling tools directly over asking clarifying questions.
2. Keep reasoning brief (1-2 sentences) before making a tool call.
3. Give up after three consecutive failures of the same tool and report the error.
4. Respond in plain text once all necessary information has been gathered.

## Config directory structure

```
config/
  tensorzero.toml                          # Gateway config; variant references system_template
  functions/
    agent/
      baseline/
        system.txt                         # The system prompt injected into every conversation
```

The key line in `tensorzero.toml` is:

```toml
[functions.agent.variants.baseline]
type = "chat_completion"
model = "gpt_oss_20"
system_template = "functions/agent/baseline/system.txt"
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A `GROQ_API_KEY` environment variable (get one at https://console.groq.com)

## How to run

```bash
# 1. Export your API key
export GROQ_API_KEY="gsk_..."

# 2. Start the gateway and ClickHouse
docker compose -f examples/03_system_prompt/docker-compose.yml up -d

# 3. Run the example
uv run python examples/03_system_prompt/main.py

# 4. Tear down when done
docker compose -f examples/03_system_prompt/docker-compose.yml down
```

## What to observe

- The agent calls `get_weather` to fetch the weather for Paris, then calls `convert_temperature` to convert the result to Fahrenheit.
- With the system prompt active, the model avoids asking "Which city?" and instead calls the tool directly.
- Logging output (INFO level) shows each turn of the agent loop, including tool calls and results.
- The final output prints the response, turn count, response source, and episode ID.
