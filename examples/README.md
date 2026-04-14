# t0-agents Examples

Hands-on examples showing how to use t0-agents to connect TensorZero with FastMCP tool servers.

## Prerequisites

All examples require:

- Python 3.11+
- Dependencies installed: `uv sync`

Examples marked **Docker: Yes** additionally need:

- Docker and Docker Compose
- `GROQ_API_KEY` set in your environment (or in a `.env` file at the project root)

## Examples

| # | Directory | Difficulty | Topic | Docker |
|---|-----------|------------|-------|--------|
| 01 | `01_adapter_basics` | Easy | Adapter layer exploration | No |
| 02 | `01_basic_agent` | Easy | Minimal agent loop | Yes |
| 03 | `02_system_prompt` | Intermediate | System prompt + multi-tool | Yes |
| 04 | `03_multi_server` | Intermediate | Server composition | Yes |
| 05 | `04_error_handling` | Intermediate | Error recovery patterns | Yes |
| 06 | `05_feedback_loop` | Advanced | Observability + feedback | Yes |

> Directory names will be renumbered to match the table in a future cleanup.

## Start here

If you are new to t0-agents, start with **01_adapter_basics** -- it runs entirely offline and shows the core conversion layer that every other example builds on. Then move to the basic agent example to see the full inference loop in action.
