
## Problema 1: "No quiero escribir el loop cada vez"

**Hoy**: Cada agente es un `while True` artesanal. El desarrollador copia `run_agent()` y lo modifica. Si tiene 5 agentes, tiene 5 loops ligeramente diferentes que divergen con el tiempo.

**Lo que falta**: Una abstracción `Agent` declarativa que encapsule la configuración (qué función T0, qué tools, qué límites, qué system prompt) y exponga un `.run()` limpio.

```python
# Lo que el developer quiere escribir:
from t0_agents import Agent

agent = Agent(
    name="research_assistant",
    function="agent",                    # → tensorzero.toml
    tools=["server.py"],                 # → FastMCP servers
    max_turns=20,
    tool_timeout=30.0,
)

result = await agent.run("Find papers about RLHF")
```

**Lo que esto resuelve**: Elimina el boilerplate. El loop, episode management, tool discovery, error handling, y serialización dejan de ser responsabilidad del developer. También permite componer agentes de formas interesantes después (Problema 4).

---

## Problema 2 (Tal vez este no es el seguno en importancia, no es tan prioritario, ayudame a valorarlo por ejemplo hasta que tengamos listas otras cosas): "Mi agente responde todo de golpe, el usuario no ve nada por 10 segundos"

**Hoy**: `t0.inference()` retorna la respuesta completa. El usuario espera en silencio hasta que todo el razonamiento + tool calls + respuesta final termina.

**Lo que falta**: Streaming a dos niveles. Primero, streaming de tokens del LLM durante la respuesta final (T0 ya soporta streaming vía SSE). Segundo, streaming de *eventos del agent loop* — "llamando herramienta X...", "resultado recibido", "generando respuesta..." — para que el frontend pueda mostrar progreso.

```python
# Streaming de eventos del loop:
async for event in agent.stream("What's the weather?"):
    match event:
        case ToolCallEvent(name="get_weather", arguments={...}):
            show_spinner(f"Calling {event.name}...")
        case ToolResultEvent(name="get_weather", result="..."):
            hide_spinner()
        case TextDeltaEvent(delta="The weather"):
            append_to_response(event.delta)
        case DoneEvent(response="The weather in Tokyo is..."):
            finalize()
```

**Lo que esto resuelve**: UX de agente en tiempo real. Sin esto, cualquier frontend (web, mobile, TUI) se siente muerto mientras el agente trabaja.

**Dependencia**: Problema 1 (Agent abstraction) + investigar la API de streaming de T0.

---

## Problema 3 (En este caso me gustaría crear abstracciones base usando patterns como Repository para que el developer implemente sus propias memorias, pero claro tener en el proyecto algo funcional in-memory y al menos un ejemplo con sqlite, un developer va a querer persistir las interacciones en otra base de datos aparte de clickhouse [mongodb, redis]): "Mi agente no recuerda nada entre conversaciones"

**Hoy**: `episode_id` mantiene contexto dentro de una conversación. Pero si el usuario cierra la app y regresa mañana, el agente no recuerda nada.

**Lo que falta**: Capas de memoria.

- **Conversación** (ya existe vía episode_id + historial de mensajes)
- **Sesión**: Estado que persiste entre mensajes pero se resetea al cerrar. Ej: preferencias temporales, archivos subidos, contexto de tarea.
- **Persistente**: Estado que sobrevive entre sesiones. Ej: perfil del usuario, historial de interacciones, conocimiento acumulado.

```python
agent = Agent(
    name="personal_assistant",
    function="assistant",
    tools=["calendar.py", "email.py"],
    memory=SqliteMemory("./agent_memory.db"),  # o RedisMemory, PostgresMemory
)

# El agente puede recordar entre sesiones:
# "La última vez me pediste que te recordara la reunión del jueves"
```

**Lo que esto resuelve**: Agentes que se sienten como asistentes reales, no como chats stateless. También habilita retrieval augmented generation (RAG) como un tipo de memoria.

**Dependencia**: Problema 1. También necesita diseñar cómo se inyecta la memoria en el system prompt de T0 (probablemente como contexto adicional en el `input`).

---

## Problema 4 (No hacer de momento hasta que tengamos el flujo completo y listo de 1 solo agente interactuando con mcp servers, memoria y abstracciones elegantes): "Necesito que un agente delegue a otro"

**Hoy**: Un solo agente con un solo conjunto de tools. Si quieres que un agente "de investigación" pase resultados a un agente "de redacción", lo cableas manualmente.

**Lo que falta**: Composición de agentes. Un agente puede delegar subtareas a otros agentes, cada uno con sus propias tools y especialización.

```python
researcher = Agent(name="researcher", function="research", tools=["arxiv.py"])
writer = Agent(name="writer", function="writing", tools=["markdown.py"])

orchestrator = Agent(
    name="report_generator",
    function="orchestrator",
    sub_agents=[researcher, writer],  # expuestos como tools al orchestrator
)

# El orchestrator decide cuándo llamar a researcher vs writer
result = await orchestrator.run("Write a report on RLHF advances in 2025")
```

**Nota**: FastMCP `mount()` con namespacing ya resuelve la parte de tools. Lo que falta es exponer *agentes* como tools de otro agente, donde la "herramienta" es en realidad otro loop de inferencia completo.

**Lo que esto resuelve**: Agentes especializados que colaboran. El patrón más común en sistemas multi-agente productivos.

**Dependencia**: Problema 1. Los sub-agentes son conceptualmente tools cuya implementación es otro `agent.run()`.

---

## Problema 5 (este se me hace muy interesante, un caso de uso que me imagino es deployar ciertos servers con egentes especificos para cada area, por ejemplo un server que solo se usa para la app mobile y otro server con otro grupo diferente para analytics [aunque puedan usar el mismo T0 gateway]): "¿Cómo expongo mi agente como API para mi app web/mobile?"

**Hoy**: El agente es un script Python. Para usarlo desde una web app, el developer tiene que construir toda la capa de serving: endpoints, WebSocket, autenticación, rate limiting.

**Lo que falta**: Un server integrado que exponga el agente como API HTTP/WebSocket con un protocolo bien definido.

```python
from t0_agents import Agent, serve

agent = Agent(name="assistant", function="agent", tools=["server.py"])

# Opción 1: Servidor standalone
serve(agent, host="0.0.0.0", port=8080)
# → POST /chat  {"message": "...", "episode_id": "..."}
# → GET  /chat/stream  (SSE)
# → POST /feedback  {"episode_id": "...", "metric": "...", "value": ...}

# Opción 2: Montar en FastAPI existente
from fastapi import FastAPI
app = FastAPI()
agent.mount(app, prefix="/agent")
```

Esto se apoya en que FastMCP ya sabe montarse en Starlette/FastAPI como ASGI app. El patrón sería similar pero para el agente completo, no solo para los tools.

**Lo que esto resuelve**: El developer pasa de "tengo un script" a "tengo un microservicio" con una línea. Web, mobile, cualquier cliente HTTP puede hablar con el agente.

**Dependencia**: Problema 1 + Problema 2 (streaming es esencial para la experiencia de chat).

---

## Problema 6 (creo que con lo que ya tenemos de elicitations podemos cumplir con esto, solo necesitamos pensar en mejores abstracciones): "El agente no debe poder hacer ciertas cosas sin aprobación"

**Hoy**: El agente ejecuta cualquier tool call que el LLM pida. No hay punto de control.

**Lo que falta**: Guardrails a múltiples niveles.

- **Approval gates**: Ciertas herramientas (enviar email, borrar datos, hacer compras) requieren confirmación humana antes de ejecutarse.
- **Input/output validation**: Filtrar contenido inapropiado, PII, o respuestas que violen políticas.
- **Cost/rate control**: Límite de tokens, de llamadas a tools, de costo total por episodio.

```python
from t0_agents import Agent, require_approval, max_cost

agent = Agent(
    name="executive_assistant",
    function="assistant",
    tools=["email.py", "calendar.py", "purchases.py"],
    guardrails=[
        require_approval(tools=["send_email", "make_purchase"]),
        max_cost(per_episode=0.50),  # USD
    ],
)
```

**Nota**: Ya tenemos Elicitations de fastmcp y ya hay eejmplos en t0-agents, no vamos a usr Approval

**Lo que esto resuelve**: Confianza para poner agentes en producción con acciones destructivas. Sin esto, nadie va a dar a un agente acceso a email o pagos de verdad.

**Dependencia**: Problema 1 + Problema 5 (necesitas un canal de comunicación con el humano para la aprobación).

---

## Problema 7 (esto suena interesante pero lo dejaría al final): "¿Cómo testeo que mi agente se comporta correctamente?"

**Hoy**: Tests manuales. Ejecutas el agente, ves si la respuesta tiene sentido, y cruzas los dedos.

**Lo que falta**: Framework de testing para agentes.

```python
from t0_agents.testing import AgentTestCase, mock_inference

class TestResearchAgent(AgentTestCase):
    agent = research_agent

    async def test_uses_search_tool(self):
        """El agente debe buscar antes de responder."""
        result = await self.run("What is RLHF?")
        self.assert_tool_called("search_papers")
        self.assert_tool_not_called("send_email")

    async def test_handles_tool_failure(self):
        """El agente debe manejar errores gracefully."""
        with self.fail_tool("search_papers", error="Service unavailable"):
            result = await self.run("What is RLHF?")
            assert "unable" in result.response.lower() or "try again" in result.response.lower()
```

Esto incluye: mocks de inferencia (respuestas LLM predeterminadas para tests deterministas), assertions sobre tool usage, simulación de errores, y snapshot testing de conversaciones.

**Lo que esto resuelve**: CI/CD para agentes. Sin testing automático, no hay confianza para hacer deploy.

**Dependencia**: Problema 1. El transporte in-memory de FastMCP ya resuelve el lado de tools; falta el lado de mock de inferencia T0.

---

## Problema 8 (esto tambien me gustaría, es más podría ser un demo de: "como es de facil integrar t0-agents a un TUI usando Typer [la lib de Tiangolo]"): "Mi agente corre en un TUI/CLI y quiero interacción rica"

**Hoy**: `input()` y `print()`.

**Lo que falta**: Un runner de TUI que renderice el streaming, los tool calls, los errores, y las approval gates de forma interactiva.

```bash
$ t0-agents run assistant --tools server.py

╭─ Research Assistant ─────────────────────────╮
│ You: Find recent papers about RLHF           │
│                                               │
│ 🔧 Calling search_papers("RLHF 2025")...    │
│ ✓ Found 3 results                            │
│                                               │
│ Assistant: Here are the most relevant...      │
╰───────────────────────────────────────────────╯
```

**Lo que esto resuelve**: Developer experience durante desarrollo. También demos instantáneas sin construir un frontend.

**Dependencia**: Problema 1 + Problema 2 (streaming).

---

## Roadmap visual

```
Fase 1 (actual)          Fase 2                Fase 3               Fase 4
MCP Bridge               Core Framework        Serving              Production
────────────             ──────────            ───────              ──────────

┌────────────┐     ┌───────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ tool_adapter│     │ P1: Agent class   │  │ P5: HTTP/WS      │  │ P6: Guardrails│
│ agent loop  │────▶│ P2: Streaming     │─▶│     serving       │─▶│ P7: Testing   │
│ error handle│     │ P3: Memory        │  │ P8: TUI runner    │  │    framework  │
└────────────┘     │ P4: Multi-agent   │  └──────────────────┘  └──────────────┘
                    └───────────────────┘

Timeline:       Semanas 1-3            Semanas 4-8           Semanas 6-10        Semanas 8-12
```

Las fases no son estrictamente secuenciales — P8 (TUI) puede arrancar en paralelo con P5 (serving), y P7 (testing) debería desarrollarse *junto* con cada feature, no al final.

---

## Lo que NO construiría

Hay cosas que es tentador incluir pero que crean más problemas de los que resuelven:

**No construiría un prompt manager**. T0 ya maneja templates y variantes. Duplicar eso en el framework sería confuso.

**No construiría un RAG engine**. Hay demasiadas librerías buenas (LlamaIndex, LangChain retrievers). Mejor exponer una interfaz de `Memory` que se integre con lo que el developer ya use.

**No construiría un DSL para workflows**. Los state machines y DAGs suenan bien en demos pero en producción el developer quiere Python normal. La composición de agentes (P4) cubre el 90% de los casos sin inventar un lenguaje nuevo.

**No abstraería el LLM provider**. T0 ya lo hace. El framework no debería saber si estás usando OpenAI o Anthropic — esa es responsabilidad del gateway.

