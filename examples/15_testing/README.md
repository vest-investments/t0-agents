# Example 15 — testing an agent

Two complementary test paths on a single agent:

- **Unit** (`test_agent_unit.py`) — deterministic, no Docker, no API
  keys. Every turn is scripted via `MockGateway`; every tool call
  returns a canned result via `MockToolProvider`. Milliseconds per test.
- **Embedded** (`test_agent_embedded.py`) — real TensorZero inference
  against the provider whose API key is in your env (default:
  `GROQ_API_KEY`). No ClickHouse, no Postgres, no Docker. Opt-in via
  `RUN_EMBEDDED_TESTS=1`.

The agent itself (`agent.py`) is deliberately small: three tools
(`search_notes`, `save_note`, `read_note`) and a plain `Agent`
constructor. Both test files import it and swap the gateway / tool
provider as needed.

## Run

From the repo root:

```bash
# Fast path — always safe, no external services
uv run pytest examples/15_testing/test_agent_unit.py

# Real-inference path — brings your own API key
GROQ_API_KEY=... RUN_EMBEDDED_TESTS=1 uv run pytest examples/15_testing -m embedded
```

Or from inside the example directory:

```bash
cd examples/15_testing
pytest .                       # runs unit tests; embedded are skipped
RUN_EMBEDDED_TESTS=1 pytest .  # runs both (needs an API key)
```

## Structure

```
examples/15_testing/
├── agent.py                 # build_agent(), build_mcp_server()
├── conftest.py              # sys.path hack so tests can `import agent`
├── pytest.ini               # registers the 'embedded' marker locally
├── tensorzero.toml          # config for the embedded test
├── test_agent_unit.py       # MockGateway + MockToolProvider tests
└── test_agent_embedded.py   # EmbeddedGatewayHelper tests (opt-in)
```

## When to pick which mode

| Question | Answer |
| --- | --- |
| "Did the agent call tool X with args Y?" | unit (MockToolProvider.calls) |
| "Does the agent hit max_turns?" | unit (scenarios.max_turns_exhausted) |
| "Did the agent recover from a raising tool?" | unit (MockToolProvider handler that raises) |
| "Does the real model understand this prompt?" | embedded |
| "Does our TOML variant actually parse?" | embedded |
| "Does a new provider key work?" | embedded |

If you have to pick one, start with unit — it fails fast, it's
deterministic, and most bugs land there. Use embedded sparingly and
only for questions mocks can't answer.

## Copying this example into your project

The example's layout is deliberately self-contained. Copy the whole
directory, rename the tools in `agent.py`, and you have a working
starting point. Keep the `conftest.py` — it's the one line that makes
`from agent import ...` work regardless of where pytest is launched.
