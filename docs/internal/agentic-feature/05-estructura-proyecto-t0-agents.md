# t0-agents — Estructura del Proyecto

## Principio de diseño

Cada subdirectorio dentro de `src/t0_agents/` es un **submódulo independiente**
que resuelve un problema específico. El único submódulo que los demás pueden
importar libremente es `core/` — que contiene los tipos, protocolos, y la
interfaz `Agent`. Todo lo demás es opt-in.

```
t0-agents/
│
├── src/
│   └── t0_agents/
│       │
│       ├── __init__.py                  ← API pública: re-exports selectivos
│       ├── py.typed                     ← marker para type checkers
│       │
│       │   ╔══════════════════════════════════════════════════════╗
│       │   ║  CORE — tipos y contratos que todos los módulos     ║
│       │   ║  pueden importar. Sin dependencias externas pesadas.║
│       │   ╚══════════════════════════════════════════════════════╝
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── types.py                 ← AgentResult, AgentEvent, ToolProvider (Protocol)
│       │   ├── config.py                ← AgentConfig dataclass
│       │   └── exceptions.py            ← T0AgentsError, MaxTurnsExceeded, etc.
│       │
│       │   ╔══════════════════════════════════════════════════════╗
│       │   ║  v0.1 — EL PUENTE (lo que construimos ahora)       ║
│       │   ╚══════════════════════════════════════════════════════╝
│       │
│       ├── mcp/                         ← Submódulo: FastMCP ↔ TensorZero bridge
│       │   ├── __init__.py              ← from t0_agents.mcp import MCPToolProvider
│       │   ├── adapter.py               ← convert_tools(), serialize_tool_result(), build_tool_result()
│       │   └── loop.py                  ← run_agent(), execute_tool_calls(), has_tool_calls()
│       │
│       │   ╔══════════════════════════════════════════════════════╗
│       │   ║  FUTURO — cada problema = un submódulo              ║
│       │   ╚══════════════════════════════════════════════════════╝
│       │
│       ├── stream/                      ← P2: Streaming de eventos del agent loop
│       │   ├── __init__.py
│       │   ├── events.py                ← ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent
│       │   └── handler.py               ← StreamingAgentLoop (async generator de eventos)
│       │
│       ├── memory/                      ← P3: Capas de memoria
│       │   ├── __init__.py
│       │   ├── base.py                  ← MemoryProvider (Protocol): load(), save(), search()
│       │   ├── conversation.py          ← ConversationMemory (in-memory, efímera)
│       │   ├── sqlite.py                ← SqliteMemory (persistente, single-node)
│       │   └── redis.py                 ← RedisMemory (persistente, distribuida)
│       │
│       ├── compose/                     ← P4: Multi-agent, delegación
│       │   ├── __init__.py
│       │   ├── delegation.py            ← agent_as_tool(): expone un Agent como MCP tool
│       │   └── orchestrator.py          ← OrchestratorAgent con sub-agents
│       │
│       ├── serve/                       ← P5: HTTP/WebSocket serving
│       │   ├── __init__.py
│       │   ├── app.py                   ← create_app(agent) → Starlette/FastAPI ASGI app
│       │   ├── routes.py                ← /chat, /chat/stream (SSE), /feedback, /health
│       │   └── protocol.py              ← ChatRequest, ChatResponse, StreamEvent schemas
│       │
│       ├── guardrails/                  ← P6: Safety y control
│       │   ├── __init__.py
│       │   ├── base.py                  ← Guardrail (Protocol): before_tool(), after_tool()
│       │   ├── approval.py              ← RequireApproval (human-in-the-loop)
│       │   └── limits.py                ← MaxCost, MaxTokens, RateLimit
│       │
│       ├── testing/                     ← P7: Framework de testing para agentes
│       │   ├── __init__.py
│       │   ├── mocks.py                 ← MockInference, MockToolProvider
│       │   ├── assertions.py            ← assert_tool_called(), assert_episode_consistent()
│       │   └── fixtures.py              ← pytest fixtures: agent_client, mock_t0, etc.
│       │
│       └── cli/                         ← P8: TUI runner
│           ├── __init__.py
│           ├── app.py                   ← CLI entrypoint (click/cyclopts)
│           └── tui.py                   ← Rich/Textual TUI para interacción
│
├── tests/
│   ├── conftest.py                      ← fixtures compartidos
│   ├── test_mcp/                        ← tests del puente (v0.1)
│   │   ├── test_adapter.py
│   │   ├── test_loop.py
│   │   └── test_integration.py          ← requiere T0 Gateway running
│   ├── test_stream/
│   ├── test_memory/
│   ├── test_compose/
│   ├── test_serve/
│   ├── test_guardrails/
│   └── test_cli/
│
├── examples/
│   ├── 01_basic_agent/                  ← Mínimo viable: 1 server, 1 agente
│   │   ├── server.py
│   │   ├── main.py
│   │   └── config/
│   │       └── tensorzero.toml
│   ├── 02_multi_server/                 ← Múltiples MCP servers con namespacing
│   ├── 03_streaming/                    ← Agent con streaming a terminal
│   ├── 04_web_app/                      ← FastAPI + agent como API
│   └── 05_multi_agent/                  ← Orchestrator con sub-agentes
│
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

---

## pyproject.toml

```toml
[project]
name = "t0-agents"
version = "0.1.0"
description = "The Pythonic agentic framework for TensorZero"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "Tu Nombre", email = "tu@email.com" },
]
keywords = ["tensorzero", "agents", "mcp", "llm", "ai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

# ─── Dependencias base: solo el puente ───
dependencies = [
    "tensorzero>=0.2.0",
    "fastmcp>=3.0.0",
]

[project.optional-dependencies]
# Futuro: cada submódulo agrega sus deps
serve = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
]
memory-sqlite = [
    "aiosqlite>=0.20.0",
]
memory-redis = [
    "redis[hiredis]>=5.0.0",
]
cli = [
    "rich>=13.0.0",
    "click>=8.0.0",
]
# Dev dependencies
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "inline-snapshot>=0.18.0",
]
# Todo junto
all = [
    "t0-agents[serve,memory-sqlite,memory-redis,cli]",
]

[project.urls]
Homepage = "https://github.com/tu-usuario/t0-agents"
Documentation = "https://t0-agents.dev"
Repository = "https://github.com/tu-usuario/t0-agents"

[project.scripts]
t0-agents = "t0_agents.cli.app:main"   # futuro: CLI entrypoint

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/t0_agents"]

# ─── Herramientas ───

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: requiere T0 Gateway running (deselect with '-m not integration')",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
```

---

## Grafo de dependencias entre submódulos

```
                    ┌──────────┐
                    │   core/  │  ← tipos, protocolos, config
                    │  types   │  ← AgentResult, ToolProvider (Protocol)
                    │  config  │  ← AgentConfig
                    │  except  │  ← excepciones
                    └────┬─────┘
                         │
            ┌────────────┼────────────────┬──────────────┐
            │            │                │              │
            ▼            ▼                ▼              ▼
       ┌─────────┐ ┌──────────┐   ┌───────────┐  ┌───────────┐
       │  mcp/   │ │ stream/  │   │  memory/  │  │guardrails/│
       │ adapter │ │  events  │   │   base    │  │   base    │
       │  loop   │ │ handler  │   │  sqlite   │  │ approval  │
       └────┬────┘ └────┬─────┘   │  redis    │  │  limits   │
            │           │         └─────┬─────┘  └─────┬─────┘
            │           │               │              │
            └─────┬─────┴───────────────┴──────────────┘
                  │
                  ▼
           ┌────────────┐
           │  compose/   │  ← importa mcp/ para agent_as_tool
           │ delegation  │
           │orchestrator │
           └──────┬──────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
    ┌──────────┐    ┌──────────┐
    │  serve/  │    │   cli/   │
    │   app    │    │   tui    │
    │  routes  │    └──────────┘
    └──────────┘

Regla: las flechas van HACIA ABAJO. Ningún módulo
importa algo que esté debajo de él en el grafo.
core/ es el único que todos pueden importar.
```

---

## API pública — lo que el developer importa

### v0.1 (solo el puente)

```python
# Uso simple — todo en una función
from t0_agents.mcp import run_agent, convert_tools, execute_tool_calls

# Uso con imports explícitos
from t0_agents.mcp.adapter import convert_tools, serialize_tool_result, build_tool_result
from t0_agents.mcp.loop import run_agent, execute_tool_calls, has_tool_calls
from t0_agents.core.types import AgentResult
from t0_agents.core.exceptions import MaxTurnsExceeded
```

### Futuro (framework completo)

```python
# El import principal — API de alto nivel
from t0_agents import Agent

agent = Agent(
    name="assistant",
    function="agent",
    tools=["server.py"],
)
result = await agent.run("Hello")

# Streaming
async for event in agent.stream("Hello"):
    ...

# Serving
from t0_agents.serve import create_app
app = create_app(agent)

# Testing
from t0_agents.testing import MockInference, assert_tool_called

# CLI
# $ t0-agents run assistant --tools server.py
```

---

## Lo que se construye AHORA (v0.1)

```
src/t0_agents/
├── __init__.py              ← "from t0_agents.mcp import run_agent"
├── py.typed
├── core/
│   ├── __init__.py
│   ├── types.py             ← AgentResult (TypedDict o dataclass)
│   └── exceptions.py        ← MaxTurnsExceeded
└── mcp/
    ├── __init__.py           ← re-exports: run_agent, convert_tools, execute_tool_calls
    ├── adapter.py            ← convert_tools, serialize_tool_result, build_tool_result
    └── loop.py               ← run_agent, execute_tool_calls, has_tool_calls

tests/
├── conftest.py              ← fixture mcp_client con server de prueba
├── test_mcp/
│   ├── test_adapter.py      ← unit tests del adaptador
│   ├── test_loop.py         ← unit tests del loop (con mocks)
│   └── test_integration.py  ← integration tests (requiere T0)
└── server.py                ← FastMCP server de prueba compartido

examples/
└── 01_basic_agent/
    ├── server.py
    ├── main.py
    └── config/
        └── tensorzero.toml

pyproject.toml
README.md
```

Solo 5 archivos de código fuente. Todo lo de la documentación v3
(contrato, referencia de implementación) mapea directamente a estos archivos.
