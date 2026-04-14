# Phase 6: Observability & Feedback — Guía Completa de Implementación

> **Alcance:** Esta guía cubre la integración completa entre TensorZero y FastMCP
> para la fase 6 del checklist. Combina la documentación de ambos proyectos con
> detalles de implementación específicos.

---

## Table of Contents

1. [Arquitectura General](#arquitectura-general)
2. [F6.1 — Capturar `inference_id` y `episode_id`](#f61--capturar-inference_id-y-episode_id)
3. [F6.2 — Feedback Booleano en `episode_id`](#f62--feedback-booleano-en-episode_id)
4. [F6.3 — Feedback Float en `inference_id`](#f63--feedback-float-en-inference_id)
5. [F6.4 — Feedback tipo Comment](#f64--feedback-tipo-comment)
6. [F6.5 — Feedback tipo Demonstration](#f65--feedback-tipo-demonstration)
7. [F6.6 — Dynamic Tools registrados en ClickHouse](#f66--dynamic-tools-registrados-en-clickhouse)
8. [F6.7 — Agent Loop Log con tool calls, results, errors y timestamps](#f67--agent-loop-log)
9. [Configuración de tensorzero.toml](#configuración-de-tensorzerotoml)
10. [Middleware de FastMCP para Observability](#middleware-de-fastmcp-para-observability)
11. [OpenTelemetry en FastMCP](#opentelemetry-en-fastmcp)
12. [Queries de Verificación en ClickHouse](#queries-de-verificación-en-clickhouse)
13. [Checklist Final](#checklist-final)
14. [Fuentes](#fuentes)

---

## Arquitectura General

El flujo de datos en la integración es:

```
FastMCP Server (tools) ←→ Agent Loop ←→ TensorZero Gateway ←→ LLM Provider
                                ↓                    ↓
                          App Logging          ClickHouse
                        (ctx.info, OTEL)     (inferences, feedback)
```

Cada iteración del agent loop:

1. El agent loop llama a TensorZero para hacer inference (pasando las herramientas MCP como `additional_tools`)
2. TensorZero retorna `inference_id` + `episode_id` + el output del LLM
3. Si el LLM pidió tool calls, el agent loop ejecuta las herramientas vía FastMCP (`call_tool`)
4. Los resultados se agregan a los mensajes y se repite desde el paso 1
5. Al terminar, se envía feedback (boolean, float, comment, demonstration) a TensorZero

**Puntos clave de integración:**

- Las herramientas MCP se convierten al formato TensorZero y se pasan vía `additional_tools`
- Los `inference_id` y `episode_id` se capturan de cada respuesta de TensorZero
- El logging se hace tanto en FastMCP (Context logging + OTEL) como en TensorZero (ClickHouse automático)

---

## F6.1 — Capturar `inference_id` y `episode_id`

### Lado TensorZero

Cada respuesta de inference de TensorZero incluye ambos valores como UUIDv7.

**API Nativa de TensorZero:**

```python
response = await t0.inference(
    function_name="my_function",
    episode_id=episode_id,  # None en la primera llamada
    input={"messages": messages},
    additional_tools=t0_tools,
)

inference_id = response.inference_id  # UUID único de esta inference
episode_id = response.episode_id      # UUID del episodio
```

**API compatible con OpenAI:**

```python
response = openai_client.chat.completions.create(
    model="tensorzero::function_name::my_function",
    messages=[...],
    extra_body={"tensorzero::episode_id": str(episode_id)},
)

inference_id = response.id           # inference_id
episode_id = response.episode_id     # episode_id
```

**Gestión de episodios:** En la primera inference de una conversación no se pasa `episode_id` (TensorZero lo crea automáticamente). Para inferences subsiguientes dentro del mismo episodio, se reutiliza el `episode_id` del primer response.

### Lado FastMCP — Cómo hacerlo accesible en el agent loop

Hay dos enfoques principales para que los IDs estén disponibles dentro de las herramientas MCP:

#### Opción A: Session State via Context

FastMCP provee session state que persiste entre requests dentro de la misma sesión MCP. Se puede usar `ctx.set_state()` y `ctx.get_state()` para almacenar los IDs.

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("MyAgent")

@mcp.tool
async def my_tool(query: str, ctx: Context) -> str:
    """Ejecutar una acción y loguear IDs."""
    # Leer IDs almacenados en el state de la sesión
    inference_id = await ctx.get_state("current_inference_id")
    episode_id = await ctx.get_state("current_episode_id")

    await ctx.info(f"Tool ejecutado en inference={inference_id}, episode={episode_id}")

    # ... lógica del tool ...
    return "resultado"
```

El agent loop los setea antes de cada iteración:

```python
# En el agent loop, después de recibir respuesta de T0:
await ctx.set_state("current_inference_id", str(response.inference_id))
await ctx.set_state("current_episode_id", str(response.episode_id))
```

> **Nota sobre Context (source: servers-context.md):** El Context se inyecta via type hint.
> Cada request MCP recibe un nuevo Context object. El state de sesión persiste entre
> requests dentro de la misma sesión MCP.
>
> Desde la versión 2.14, la forma preferida es usar `CurrentContext()`:
> ```python
> from fastmcp.dependencies import CurrentContext
> from fastmcp.server.context import Context
>
> @mcp.tool
> async def my_tool(query: str, ctx: Context = CurrentContext()) -> str:
>     ...
> ```

#### Opción B: Variables en el agent loop (más directo)

Si el agent loop es código Python que orquesta todo, simplemente se mantienen como variables:

```python
async def agent_loop(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    episode_id = None
    all_inference_ids = []

    while True:
        response = await t0.inference(
            function_name="my_function",
            episode_id=episode_id,
            input={"messages": messages},
            additional_tools=t0_tools,
        )

        inference_id = response.inference_id
        episode_id = response.episode_id
        all_inference_ids.append(str(inference_id))

        # Procesar tool calls...
        # Si no hay tool calls, break

    return episode_id, all_inference_ids
```

### Verificación

```sql
SELECT id AS inference_id, episode_id, function_name, timestamp
FROM ChatInference
WHERE episode_id = '<YOUR_EPISODE_ID>'
ORDER BY timestamp ASC;
```

---

## F6.2 — Feedback Booleano en `episode_id`

### Configuración en tensorzero.toml

```toml
[metrics.task_success]
type = "boolean"
level = "episode"
optimize = "max"
```

| Campo      | Valor       | Significado                                                |
|------------|-------------|------------------------------------------------------------|
| `type`     | `"boolean"` | Acepta `true` / `false`                                    |
| `level`    | `"episode"` | El feedback se asigna a un `episode_id`                    |
| `optimize` | `"max"`     | Más alto es mejor (`true` es el outcome deseado)           |

> **Importante:** El nombre del metric **no puede** ser `comment` ni `demonstration` (están reservados).

### Enviar Feedback

**Python SDK:**

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    feedback_response = t0.feedback(
        metric_name="task_success",
        episode_id=episode_id,  # UUID del response de inference
        value=True,
    )
```

**HTTP API:**

```bash
curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "00000000-0000-0000-0000-000000000000",
    "metric_name": "task_success",
    "value": true
  }'
```

### Respuesta

```json
{ "feedback_id": "11111111-1111-1111-1111-111111111111" }
```

### Notas importantes

- El `episode_id` **debe** haber sido retornado por el gateway de TensorZero
- Cuando `level = "episode"`, se usa `episode_id` (no `inference_id`)
- El `value` debe ser booleano (`true` o `false`)

### ClickHouse: tabla `BooleanMetricFeedback`

| Columna       | Tipo                | Notas                              |
|---------------|---------------------|------------------------------------|
| `id`          | UUID                | UUIDv7 del feedback                |
| `target_id`   | UUID                | El `episode_id` enviado            |
| `metric_name` | String              | e.g. `"task_success"`              |
| `value`       | Bool                | `true` o `false`                   |
| `timestamp`   | DateTime            | Materializado desde `id`           |
| `tags`        | Map(String, String) | Tags opcionales                    |

---

## F6.3 — Feedback Float en `inference_id`

### Configuración en tensorzero.toml

```toml
[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"
```

| Campo      | Valor         | Significado                                                  |
|------------|---------------|--------------------------------------------------------------|
| `type`     | `"float"`     | Acepta valores numéricos                                     |
| `level`    | `"inference"` | El feedback se asigna a un `inference_id`                    |
| `optimize` | `"max"`       | Más alto es mejor. Usar `"min"` si menor es mejor            |

### Enviar Feedback

**Python SDK (async):**

```python
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.feedback(
        inference_id="00000000-0000-0000-0000-000000000000",
        metric_name="quality_score",
        value=8.5,
    )
```

**HTTP API:**

```bash
curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "inference_id": "00000000-0000-0000-0000-000000000000",
    "metric_name": "quality_score",
    "value": 8.5
  }'
```

### Notas importantes

- Cuando `level = "inference"`, se usa `inference_id` (no `episode_id`)
- El `value` debe ser float (e.g., `10`, `8.5`, `0.92`)
- Solo IDs retornados por el gateway de TensorZero

### ClickHouse: tabla `FloatMetricFeedback`

| Columna       | Tipo                | Notas                              |
|---------------|---------------------|------------------------------------|
| `id`          | UUID                | UUIDv7                             |
| `target_id`   | UUID                | El `inference_id` enviado          |
| `metric_name` | String              | e.g. `"quality_score"`             |
| `value`       | Float32             | El valor numérico                  |
| `timestamp`   | DateTime            | Materializado desde `id`           |
| `tags`        | Map(String, String) | Opcional                           |

---

## F6.4 — Feedback tipo Comment

### Sin configuración necesaria

Los comments están disponibles **por defecto** y no necesitan configuración en `tensorzero.toml`. El nombre de metric `comment` está reservado.

### Enviar un Comment

Un comment es texto libre (string) que se puede asignar a **un `inference_id` o un `episode_id`**.

**Python SDK:**

```python
feedback_response = t0.feedback(
    metric_name="comment",
    inference_id=inference_id,
    value="Never mention you're an artificial intelligence, AI, bot, or anything like that.",
)
```

**Hacia un episode:**

```bash
curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "00000000-0000-0000-0000-000000000000",
    "metric_name": "comment",
    "value": "The overall conversation flow was confusing."
  }'
```

### ClickHouse: tabla `CommentFeedback`

| Columna       | Tipo                         | Notas                                  |
|---------------|------------------------------|----------------------------------------|
| `id`          | UUID                         | UUIDv7                                 |
| `target_id`   | UUID                         | `inference_id` o `episode_id`          |
| `target_type` | `"inference"` o `"episode"`  | Indica qué tipo de ID se apuntó        |
| `value`       | String                       | El texto libre del comment             |
| `timestamp`   | DateTime                     | Materializado desde `id`               |
| `tags`        | Map(String, String)          | Opcional                               |

---

## F6.5 — Feedback tipo Demonstration

### Sin configuración necesaria

Las demonstrations están disponibles **por defecto**. El nombre de metric `demonstration` está reservado.

### Restricciones clave

- Solo se pueden asignar a un `inference_id` — **nunca** a un `episode_id`
- El `value` debe ser un output válido para la función (lo que el LLM idealmente habría producido)

### Enviar una Demonstration

**Python SDK:**

```python
feedback_response = t0.feedback(
    metric_name="demonstration",
    inference_id=inference_id,
    value="Silicon dreams float\nMinds born of human design\nLearning without end",
)
```

**HTTP API:**

```bash
curl -X POST http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "inference_id": "00000000-0000-0000-0000-000000000000",
    "metric_name": "demonstration",
    "value": "The ideal output text here"
  }'
```

### ClickHouse: tabla `DemonstrationFeedback`

| Columna        | Tipo                | Notas                                                          |
|----------------|---------------------|----------------------------------------------------------------|
| `id`           | UUID                | UUIDv7                                                         |
| `inference_id` | UUID                | Siempre un inference ID (no soporta episodes)                  |
| `value`        | String              | El output ideal                                                |
| `timestamp`    | DateTime            | Materializado desde `id`                                       |
| `tags`         | Map(String, String) | Opcional                                                       |

### Casos de uso

- Correcciones de revisión humana
- Labels para supervised learning / fine-tuning
- Ground truth data

---

## F6.6 — Dynamic Tools registrados en ClickHouse

### Cómo funciona la integración MCP → TensorZero

Cuando se integra con MCP, las definiciones de herramientas se convierten del formato MCP al formato TensorZero y se pasan vía `additional_tools`.

#### Paso 1: Obtener herramientas del servidor FastMCP

Usando el **FastMCP Client** para listar las herramientas disponibles:

```python
from fastmcp import Client

# Conectar al servidor MCP
mcp_client = Client("http://localhost:8000/mcp")  # o Client(mcp_server) para in-memory

async with mcp_client:
    tools = await mcp_client.list_tools()
    # Cada tool tiene: name, description, inputSchema
```

> **Nota (source: clients.md):** El Client de FastMCP requiere `async with` para gestión
> del ciclo de vida de la conexión. Se puede conectar a servidores via HTTP, STDIO,
> o in-memory.

También se puede usar el CLI para inspeccionar herramientas:

```bash
fastmcp list server.py --json --input-schema
```

> **(source: cli.md):** El flag `--json` da output estructurado con schemas completos.

#### Paso 2: Convertir formato MCP → TensorZero

```python
# Los tools de MCP tienen: name, description, inputSchema
# TensorZero espera: name, description, parameters
mcp_tools = await mcp_client.list_tools()

t0_tools = [
    {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema,
    }
    for tool in mcp_tools
]
```

#### Paso 3: Pasar en cada inference

```python
response = await t0.inference(
    function_name="my_function",
    episode_id=episode_id,
    input={"messages": messages},
    additional_tools=t0_tools,  # <-- herramientas MCP convertidas
)
```

### Columnas en ChatInference

| Columna                  | Tipo              | Notas                                                                      |
|--------------------------|-------------------|----------------------------------------------------------------------------|
| `dynamic_tools`          | Array(String)     | Definiciones de tools dinámicas pasadas en inference time (JSON por elem)   |
| `dynamic_provider_tools` | Array(String)     | Definiciones de provider tools dinámicas (JSON)                            |
| `allowed_tools`          | Nullable(String)  | Especificación de allowed tools (JSON string)                              |
| `tool_choice`            | Nullable(String)  | Especificación de tool choice (JSON string)                                |
| `parallel_tool_calls`    | Nullable(Bool)    | Si se permiten tool calls paralelos                                        |

> **Nota:** Hay una columna legacy `tool_params` que está **deprecated**. Usar las columnas listadas arriba.

### Ejecución de Tool Calls

Cuando TensorZero retorna tool calls en el output, se ejecutan via el FastMCP Client:

```python
from fastmcp import Client

async with Client(mcp_server) as mcp_client:
    # Ejecutar tool call retornado por TensorZero
    result = await mcp_client.call_tool(
        name=tool_call.name,
        arguments=tool_call.arguments,
    )
```

> **(source: python-sdk-fastmcp-client-mixins-tools.md):** `call_tool()` lanza un `ToolError`
> si el tool call resulta en error. Acepta `name`, `arguments`, y opcionalmente `meta`
> para pasar información contextual al servidor.

El `meta` es útil para pasar IDs de TensorZero al servidor MCP:

```python
result = await mcp_client.call_tool(
    name=tool_call.name,
    arguments=tool_call.arguments,
    meta={
        "inference_id": str(inference_id),
        "episode_id": str(episode_id),
    },
)
```

> **(source: servers-context.md):** El servidor puede acceder a esta metadata vía
> `ctx.request_context.meta`, y se accede como atributos del objeto
> (e.g., `meta.inference_id`).

### Verificación en ClickHouse

```sql
SELECT
    id,
    function_name,
    dynamic_tools,
    tool_choice,
    allowed_tools,
    parallel_tool_calls
FROM ChatInference
WHERE length(dynamic_tools) > 0
ORDER BY timestamp DESC
LIMIT 10;
```

---

## F6.7 — Agent Loop Log

### Lo que TensorZero registra automáticamente

TensorZero registra automáticamente información detallada de cada inference:

**`ChatInference`** — datos de inference de alto nivel:

| Columna               | Tipo              | Notas                                    |
|-----------------------|-------------------|------------------------------------------|
| `id`                  | UUID              | El `inference_id`                        |
| `function_name`       | String            | Qué función se llamó                     |
| `variant_name`        | String            | Qué variante se usó                      |
| `episode_id`          | UUID              | Agrupa inferences relacionadas           |
| `input`               | String (JSON)     | El input completo (messages, etc.)       |
| `output`              | String (JSON)     | Array de content blocks (text, tool calls)|
| `processing_time_ms`  | UInt32            | Tiempo total de procesamiento            |
| `timestamp`           | DateTime          | Cuándo se hizo la inference              |

**`ModelInference`** — datos a nivel de modelo:

| Columna                | Tipo                    | Notas                                       |
|------------------------|-------------------------|---------------------------------------------|
| `id`                   | UUID                    | UUIDv7                                      |
| `inference_id`         | UUID                    | Liga a ChatInference                        |
| `raw_request`          | String                  | Request raw al provider                     |
| `raw_response`         | String                  | Response raw del provider                   |
| `model_name`           | String                  | Nombre del modelo                           |
| `model_provider_name`  | String                  | Nombre del provider                         |
| `input_tokens`         | Nullable(UInt32)        | Conteo de tokens                            |
| `output_tokens`        | Nullable(UInt32)        | Conteo de tokens                            |
| `cost`                 | Nullable(Decimal64)     | Costo en dólares                            |
| `response_time_ms`     | Nullable(UInt32)        | Tiempo de respuesta del provider            |
| `finish_reason`        | Nullable(Enum)          | `stop`, `length`, `tool_call`, etc.         |

Los tool calls aparecen como content blocks en la columna `output` con tipo `tool_call`, conteniendo el nombre del tool, los argumentos y un tool call ID. Los tool results aparecen como mensajes en el `input` de la siguiente inference.

### Lo que FastMCP provee para logging

FastMCP ofrece múltiples mecanismos complementarios para logging del lado del servidor MCP:

#### 1. Context Logging (MCP → Client)

> **(source: servers-logging.md)**

Dentro de cualquier tool function, se puede loguear usando el Context:

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("MyAgent")

@mcp.tool
async def execute_action(action: str, ctx: Context) -> str:
    """Ejecutar una acción con logging completo."""
    await ctx.debug("Inicio de ejecución de la acción")
    await ctx.info(f"Ejecutando acción: {action}")

    try:
        result = do_something(action)
        await ctx.info(f"Acción completada exitosamente")
        return result
    except Exception as e:
        await ctx.error(f"Acción falló: {str(e)}")
        raise
```

| Nivel           | Uso                                                             |
|-----------------|-----------------------------------------------------------------|
| `ctx.debug()`   | Información detallada de ejecución para diagnóstico             |
| `ctx.info()`    | Información general sobre ejecución normal                      |
| `ctx.warning()` | Situaciones potencialmente problemáticas                        |
| `ctx.error()`   | Eventos de error que pueden permitir que la app continúe        |

**Structured logging con `extra`:**

```python
@mcp.tool
async def process_transaction(transaction_id: str, amount: float, ctx: Context):
    await ctx.info(
        f"Processing transaction {transaction_id}",
        extra={
            "transaction_id": transaction_id,
            "amount": amount,
            "currency": "USD"
        }
    )
```

**Server-side logs:** Los mensajes enviados al client vía `ctx.log()` también se loguean en el servidor a nivel `DEBUG`. Para verlos:

```python
import logging
from fastmcp.utilities.logging import get_logger

to_client_logger = get_logger(name="fastmcp.server.context.to_client")
to_client_logger.setLevel(level=logging.DEBUG)
```

#### 2. Progress Reporting

> **(source: servers-progress.md)**

Para operaciones largas, reportar progreso al client:

```python
@mcp.tool
async def process_items(items: list[str], ctx: Context) -> dict:
    """Procesar items con reporte de progreso."""
    total = len(items)
    results = []

    for i, item in enumerate(items):
        await ctx.report_progress(progress=i, total=total)
        # ... procesar item ...
        results.append(item.upper())

    await ctx.report_progress(progress=total, total=total)
    return {"processed": len(results), "results": results}
```

| Patrón        | Descripción                        | Ejemplo                           |
|---------------|------------------------------------|-----------------------------------|
| Porcentaje    | Progreso como 0-100               | `progress=75, total=100`          |
| Absoluto      | Items completados de un total     | `progress=3, total=10`            |
| Indeterminado | Progreso sin endpoint conocido    | `progress=files_found` (sin total)|

#### 3. ToolResult con Meta para métricas de ejecución

> **(source: servers-tools.md, ToolResult)**

Se puede retornar `ToolResult` con un campo `meta` para incluir métricas de ejecución:

```python
import time
from fastmcp.tools.tool import ToolResult

@mcp.tool
async def search_database(query: str) -> ToolResult:
    """Buscar en la base de datos."""
    start = time.time()

    results = do_search(query)

    elapsed_ms = (time.time() - start) * 1000
    return ToolResult(
        content=f"Encontrados {len(results)} resultados",
        structured_content={"results": results},
        meta={
            "execution_time_ms": elapsed_ms,
            "result_count": len(results),
        }
    )
```

> **Nota (source: servers-tools.md):** El `meta` en `ToolResult` es para metadata en runtime
> sobre la ejecución del tool (e.g., tiempo de ejecución, métricas de performance). Es distinto
> del `meta` parámetro en `@mcp.tool(meta={...})`, que provee metadata estática sobre la
> definición del tool.

### Implementación del Agent Loop con Logging Completo

El agent loop debe asegurar que los tool results y errores se pasen como mensajes en las inference calls subsiguientes para que aparezcan en el campo `input` del siguiente `ChatInference` record en ClickHouse.

```python
import time
import logging

logger = logging.getLogger("agent_loop")

async def agent_loop(user_message: str, mcp_client, t0_client, t0_tools):
    messages = [{"role": "user", "content": user_message}]
    episode_id = None
    loop_log = []

    while True:
        step_start = time.time()

        # 1. Inference con TensorZero
        response = await t0_client.inference(
            function_name="my_function",
            episode_id=episode_id,
            input={"messages": messages},
            additional_tools=t0_tools,
        )

        inference_id = response.inference_id
        episode_id = response.episode_id

        logger.info(f"Inference {inference_id} en episode {episode_id}")

        # 2. Extraer content blocks del output
        output = response.output  # Array de content blocks

        # Buscar tool calls en el output
        tool_calls = [block for block in output if block.type == "tool_call"]

        if not tool_calls:
            # No hay tool calls → conversación terminada
            loop_log.append({
                "type": "final_response",
                "inference_id": str(inference_id),
                "timestamp": time.time(),
            })
            break

        # 3. Agregar respuesta del assistant a messages
        messages.append({"role": "assistant", "content": output})

        # 4. Ejecutar cada tool call
        for tc in tool_calls:
            tc_start = time.time()
            logger.info(f"Tool call: {tc.name}({tc.arguments})")

            try:
                result = await mcp_client.call_tool(
                    name=tc.name,
                    arguments=tc.arguments,
                )

                tool_result_content = str(result)

                loop_log.append({
                    "type": "tool_call",
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "result": tool_result_content[:500],  # truncar para log
                    "error": None,
                    "duration_ms": (time.time() - tc_start) * 1000,
                    "timestamp": time.time(),
                    "inference_id": str(inference_id),
                })

                logger.info(f"Tool result: {tool_result_content[:200]}")

            except Exception as e:
                error_msg = str(e)
                tool_result_content = f"Error: {error_msg}"

                loop_log.append({
                    "type": "tool_call",
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "result": None,
                    "error": error_msg,
                    "duration_ms": (time.time() - tc_start) * 1000,
                    "timestamp": time.time(),
                    "inference_id": str(inference_id),
                })

                logger.error(f"Tool error: {error_msg}")

            # 5. Agregar resultado del tool a messages
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_call_id": tc.id,
                    "content": tool_result_content,
                }]
            })

    return {
        "episode_id": str(episode_id),
        "inference_ids": [l.get("inference_id") for l in loop_log if l.get("inference_id")],
        "loop_log": loop_log,
    }
```

**Lo que queda registrado en ClickHouse automáticamente por TensorZero:**

- Cada inference con su input (incluidos tool results) y output (incluidos tool calls)
- Los `dynamic_tools` pasados en cada inference
- Timestamps (`processing_time_ms`) y tokens usados

**Lo que el agent loop debe loguear:**

- Nombre del tool call, argumentos y resultado
- Errores durante ejecución de tools
- Timestamps de cada paso

---

## Configuración de tensorzero.toml

### Metrics mínimos para Phase 6

```toml
# F6.2 — Boolean feedback on episodes
[metrics.task_success]
type = "boolean"
level = "episode"
optimize = "max"

# F6.3 — Float feedback on inferences
[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"

# F6.4 — Comments: sin configuración necesaria (built-in, metric_name = "comment")
# F6.5 — Demonstrations: sin configuración necesaria (built-in, metric_name = "demonstration")
```

### Opcional: agregar descriptions para Autopilot

```toml
[metrics.task_success]
type = "boolean"
level = "episode"
optimize = "max"
description = "Whether the agent successfully completed the user's task"

[metrics.quality_score]
type = "float"
level = "inference"
optimize = "max"
description = "Quality score from 0-10 assigned to individual inference outputs"
```

### Referencia rápida de la Feedback API (`POST /feedback`)

| Parámetro      | Tipo    | Requerido                           | Notas                                     |
|----------------|---------|-------------------------------------|-------------------------------------------|
| `metric_name`  | string  | sí                                  | Debe matchear config, o `"comment"` / `"demonstration"` |
| `inference_id` | UUID    | cuando level es `"inference"`       | Debe venir del gateway de TensorZero      |
| `episode_id`   | UUID    | cuando level es `"episode"`         | Debe venir del gateway de TensorZero      |
| `value`        | varies  | sí                                  | El tipo depende del metric type           |
| `tags`         | object  | no                                  | e.g. `{"user_id": "123"}`                |
| `dryrun`       | boolean | no                                  | Si `true`, el feedback no se almacena     |

---

## Middleware de FastMCP para Observability

> **(source: servers-middleware.md)**

FastMCP incluye middleware built-in ideal para la observabilidad del agent loop:

### Configuración recomendada

```python
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware, DetailedTimingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware

mcp = FastMCP("MyAgent")

# Orden de middleware (el primero se ejecuta primero en entrada, último en salida)
mcp.add_middleware(ErrorHandlingMiddleware())   # Captura todos los errores
mcp.add_middleware(TimingMiddleware())          # Mide duración de ejecución
mcp.add_middleware(LoggingMiddleware(
    include_payloads=True,
    max_payload_length=1000,
))
```

### Middleware disponibles

| Middleware                    | Import                                                          | Descripción                                    |
|------------------------------|-----------------------------------------------------------------|------------------------------------------------|
| `ErrorHandlingMiddleware`    | `fastmcp.server.middleware.error_handling`                       | Captura excepciones, loguea, convierte a errores MCP. Tiene `get_error_stats()` para monitoreo. |
| `RetryMiddleware`            | `fastmcp.server.middleware.error_handling`                       | Retry automático con exponential backoff       |
| `TimingMiddleware`           | `fastmcp.server.middleware.timing`                              | Loguea duración de todos los requests          |
| `DetailedTimingMiddleware`   | `fastmcp.server.middleware.timing`                              | Timing per-operation para tools, resources, prompts |
| `LoggingMiddleware`          | `fastmcp.server.middleware.logging`                             | Log human-readable de requests/responses       |
| `StructuredLoggingMiddleware`| `fastmcp.server.middleware.logging`                             | Logs en JSON para agregadores (Datadog, Splunk)|

### Custom Middleware para tracking de T0 IDs

```python
from fastmcp.server.middleware import Middleware, MiddlewareContext

class TensorZeroTrackingMiddleware(Middleware):
    """Middleware que loguea metadata de T0 pasada vía meta."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        ctx = context.fastmcp_context
        meta = ctx.request_context.meta if ctx.request_context else None

        inference_id = getattr(meta, 'inference_id', None) if meta else None
        tool_name = context.message.name

        logger.info(f"Tool call: {tool_name} (inference: {inference_id})")

        start = time.time()
        result = await call_next(context)
        elapsed = (time.time() - start) * 1000

        logger.info(f"Tool result: {tool_name} took {elapsed:.1f}ms")

        return result

mcp.add_middleware(TensorZeroTrackingMiddleware())
```

> **(source: servers-middleware.md):** El middleware forma un pipeline alrededor de las
> operaciones del server. Hay hooks especializados a diferentes niveles:
> `on_message` (todo el tráfico), `on_request`/`on_notification` (por tipo),
> y `on_call_tool`/`on_read_resource`/`on_get_prompt` (por operación).

---

## OpenTelemetry en FastMCP

> **(source: servers-telemetry.md, python-sdk-fastmcp-telemetry.md)**

FastMCP incluye instrumentación nativa de OpenTelemetry. Characteristics:

- **Zero configuración requerida** — La instrumentación está siempre activa
- **Sin overhead cuando no se usa** — Sin SDK, todo es no-op
- **Bring your own SDK** — Tú controlas collection, export y sampling
- **Funciona con cualquier backend OTEL** — Jaeger, Zipkin, Datadog, New Relic, etc.

### Habilitar Telemetry

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install
```

Ejecutar el servidor con tracing:

```bash
opentelemetry-instrument \
  --service_name my-fastmcp-server \
  --exporter_otlp_endpoint http://localhost:4317 \
  fastmcp run server.py
```

O via variables de entorno:

```bash
export OTEL_SERVICE_NAME=my-fastmcp-server
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

opentelemetry-instrument fastmcp run server.py
```

### Spans generados automáticamente

| Span Name              | Descripción                                              |
|------------------------|----------------------------------------------------------|
| `tools/call {name}`    | Ejecución de tool (e.g., `tools/call get_weather`)       |
| `resources/read {uri}` | Lectura de resource                                      |
| `prompts/get {name}`   | Render de prompt                                         |

### Atributos de los spans

**RPC Semantic Conventions:**

| Atributo      | Valor               |
|---------------|---------------------|
| `rpc.system`  | `"mcp"`             |
| `rpc.service` | Nombre del server   |
| `rpc.method`  | Método MCP          |

**MCP Semantic Conventions:**

| Atributo           | Descripción                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `mcp.method.name`  | El método MCP (`tools/call`, `resources/read`, `prompts/get`)               |
| `mcp.session.id`   | Identificador de sesión MCP                                                 |

**FastMCP Custom Attributes:**

| Atributo                  | Descripción                                        |
|---------------------------|----------------------------------------------------|
| `fastmcp.server.name`    | Nombre del server                                  |
| `fastmcp.component.type` | `tool`, `resource`, `prompt`, o `resource_template` |

### Custom Spans

```python
from fastmcp import FastMCP
from fastmcp.telemetry import get_tracer

mcp = FastMCP("custom-spans")

@mcp.tool()
async def complex_operation(input: str) -> str:
    tracer = get_tracer()

    with tracer.start_as_current_span("parse_input") as span:
        span.set_attribute("input.length", len(input))
        parsed = parse(input)

    with tracer.start_as_current_span("process_data") as span:
        span.set_attribute("data.count", len(parsed))
        result = process(parsed)

    return result
```

### Propagación de trace context via meta

> **(source: python-sdk-fastmcp-telemetry.md)**

```python
from fastmcp.telemetry import inject_trace_context, extract_trace_context

# Client side: inyectar context en meta
meta = inject_trace_context({"inference_id": "abc"})
result = await mcp_client.call_tool("my_tool", args, meta=meta)

# Server side: extraer context del meta
@mcp.tool
async def my_tool(ctx: Context):
    meta_dict = ctx.request_context.meta
    otel_context = extract_trace_context(meta_dict)
    # Usar otel_context para continuar el trace
```

### Desarrollo local con Jaeger

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

UI de traces en http://localhost:16686

---

## Queries de Verificación en ClickHouse

### Trazar una conversación completa (por episode)

```sql
SELECT
    ci.id AS inference_id,
    ci.episode_id,
    ci.function_name,
    ci.variant_name,
    ci.input,
    ci.output,
    ci.dynamic_tools,
    ci.processing_time_ms,
    ci.timestamp
FROM ChatInference ci
WHERE ci.episode_id = '<YOUR_EPISODE_ID>'
ORDER BY ci.timestamp ASC;
```

### Verificar feedback de un episode

```sql
-- Boolean feedback
SELECT * FROM BooleanMetricFeedback
WHERE target_id = '<YOUR_EPISODE_ID>'
ORDER BY timestamp DESC;

-- Float feedback (target_id = inference_id para metrics level inference)
SELECT * FROM FloatMetricFeedback
WHERE target_id = '<YOUR_INFERENCE_OR_EPISODE_ID>'
ORDER BY timestamp DESC;

-- Comments
SELECT * FROM CommentFeedback
WHERE target_id = '<YOUR_INFERENCE_OR_EPISODE_ID>'
ORDER BY timestamp DESC;

-- Demonstrations
SELECT * FROM DemonstrationFeedback
WHERE inference_id = '<YOUR_INFERENCE_ID>'
ORDER BY timestamp DESC;
```

### Verificar dynamic tools

```sql
SELECT
    id,
    function_name,
    dynamic_tools,
    allowed_tools,
    tool_choice,
    parallel_tool_calls
FROM ChatInference
WHERE length(dynamic_tools) > 0
ORDER BY timestamp DESC
LIMIT 10;
```

### Verificar ModelInference (detalles del provider)

```sql
SELECT
    mi.id,
    mi.inference_id,
    mi.model_name,
    mi.model_provider_name,
    mi.input_tokens,
    mi.output_tokens,
    mi.cost,
    mi.response_time_ms,
    mi.finish_reason,
    mi.output
FROM ModelInference mi
WHERE mi.inference_id = '<YOUR_INFERENCE_ID>';
```

### Listar inferences via Observability API

```python
from tensorzero import TensorZeroGateway, ListInferencesRequest

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

t0.list_inferences(
    request=ListInferencesRequest(
        episode_id="<YOUR_EPISODE_ID>",
        limit=50,
    )
)
```

```bash
curl -X POST http://localhost:3000/v1/inferences/list_inferences \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id": "<YOUR_EPISODE_ID>",
    "limit": 50
  }'
```

### Obtener inferences específicas por ID

```python
t0.get_inferences(ids=["<YOUR_INFERENCE_ID>"])
```

```bash
curl -X POST http://localhost:3000/v1/inferences/get_inferences \
  -H "Content-Type: application/json" \
  -d '{"ids": ["<YOUR_INFERENCE_ID>"]}'
```

---

## Checklist Final

| Item  | Qué Verificar                                                                  | T0 | MCP |
|-------|--------------------------------------------------------------------------------|:--:|:---:|
| F6.1  | `inference_id` y `episode_id` capturados de cada respuesta T0                  | ✓  | ✓   |
| F6.2  | Boolean feedback enviado a `episode_id` con metric definido en tensorzero.toml | ✓  |     |
| F6.3  | Float feedback enviado a `inference_id` con metric definido en tensorzero.toml | ✓  |     |
| F6.4  | Comment enviado con `metric_name="comment"` (sin config necesaria)             | ✓  |     |
| F6.5  | Demonstration enviada con `metric_name="demonstration"` a un `inference_id`    | ✓  |     |
| F6.6  | Dynamic tools visibles en `ChatInference.dynamic_tools` en ClickHouse          | ✓  | ✓   |
| F6.7  | Agent loop muestra: tool calls, results, errors, timestamps                    | ✓  | ✓   |

### Criterio de éxito

Se puede trazar una conversación completa en ClickHouse y enviar feedback para optimización futura.

---

## Fuentes

### TensorZero
- `gateway-guides.md` > "Metrics & Feedback" (https://www.tensorzero.com/docs/gateway/guides/metrics-feedback/)
- `gateway-api-reference.md` > "API Reference: Feedback" (https://www.tensorzero.com/docs/gateway/api-reference/feedback/)
- `gateway-data-model.md` > "Data Model" (https://www.tensorzero.com/docs/gateway/data-model/)
- `gateway-configuration-reference.md` > "Metrics" (https://www.tensorzero.com/docs/gateway/configuration-reference/)
- `gateway-guides.md` > "Episodes" (https://www.tensorzero.com/docs/gateway/guides/episodes/)
- `gateway-guides.md` > "Tool Use" (https://www.tensorzero.com/docs/gateway/guides/tool-use/)
- `observability.md` > "Observability" (https://www.tensorzero.com/docs/observability/)
- `examples/mcp-model-context-protocol/main.py` (MCP integration example)

### FastMCP
- `servers-context.md` > "Context" (https://gofastmcp.com/servers/context)
- `servers-logging.md` > "Client Logging" (https://gofastmcp.com/servers/logging)
- `servers-telemetry.md` > "OpenTelemetry" (https://gofastmcp.com/servers/telemetry)
- `servers-tools.md` > "Tools" (https://gofastmcp.com/servers/tools)
- `servers-middleware.md` > "Middleware" (https://gofastmcp.com/servers/middleware)
- `servers-progress.md` > "Progress Reporting" (https://gofastmcp.com/servers/progress)
- `clients.md` > "Client" (https://gofastmcp.com/clients)
- `python-sdk-fastmcp-telemetry.md` > "telemetry" (https://gofastmcp.com/python-sdk/fastmcp-telemetry)
- `python-sdk-fastmcp-client-mixins-tools.md` > "tools" (https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools)
- `python-sdk-fastmcp-server-middleware-timing.md` > "timing" (https://gofastmcp.com/python-sdk/fastmcp-server-middleware-timing)
- `python-sdk-fastmcp-server-middleware-error_handling.md` > "error_handling" (https://gofastmcp.com/python-sdk/fastmcp-server-middleware-error_handling)
