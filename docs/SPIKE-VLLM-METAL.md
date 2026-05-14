# Spike: vLLM-metal vs Ollama en M2 Max

**Fecha:** 2026-05-14
**Hardware:** Apple M2 Max, 32GB unified memory, macOS 15.4
**Modelo:** Mistral Small 24B Instruct 2501 (4-bit MLX quant, 13.3GB)
**Disparador:** El documento de diseño original del PoC recomendaba Ollama y
asumía que vLLM no corre en Mac. La página oficial de vLLM lista "Apple Silicon" como
plataforma soportada y mantiene el repo
[`vllm-project/vllm-metal`](https://github.com/vllm-project/vllm-metal)
(community plugin, v0.2.0, mantenido activamente). Faltaba validar:

1. ¿vLLM-metal levanta y sirve Mistral Small 24B?
2. ¿Soporta tool calling estructurado (PydanticAI lo necesita)?
3. ¿La promesa de batched serving se cumple bajo carga concurrente?

## TL;DR

| Concurrencia | Backend | P50 | P95 | Throughput | Speedup vs Ollama |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM-metal | 4.7s | 9.0s | 0.19/s | 1.3x |
| 1 | Ollama | 6.2s | 9.5s | 0.15/s | — |
| 5 | vLLM-metal | 11.8s | 13.6s | 0.40/s | **3.4x P50** |
| 5 | Ollama | 40.5s | 50.8s | 0.12/s | — |
| 10 | vLLM-metal | 20.5s | 23.1s | 0.45/s | **4.2x P50** |
| 10 | Ollama | 86.9s | 119.0s | 0.08/s | — |

- Correctness 100% en ambos (3/3 en preguntas con número esperado, sin cruces de tenants).
- Ollama degrada 14x al ir de 1→10 usuarios concurrentes; vLLM solo 4.4x.
- vLLM mantiene throughput **creciente** con concurrencia (batched serving real).
  Ollama lo pierde.
- **Decisión:** vLLM-metal queda como backend recomendado del PoC. Ollama
  permanece soportado como fallback simple.

## Instalación de vLLM-metal

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

El instalador oficial:
- Verifica Apple Silicon.
- Instala `uv` si falta.
- Crea `~/.venv-vllm-metal`.
- Builda vLLM 0.20.2 desde fuente (5-10 min).
- Instala el wheel `vllm-metal-0.2.0`.

Quedan: `vllm`, `mlx==0.31.2`, `mlx-lm==0.31.3`, `mlx-metal==0.31.2`. Backend de
cómputo: MLX sobre Metal; PyTorch device = `mps` para piezas auxiliares.

Total en disco (con modelo descargado): ~16GB.

## Hallazgos clave

### 1. Tool calling **necesita** chat template explícito

Con `--enable-auto-tool-choice --tool-call-parser mistral` solos, el modelo
emite el "tool call" como **texto markdown en `content`** y `tool_calls` queda
vacío. Ejemplo del primer intento:

```json
"content": "Voy a consultar la base de datos...\n```\nconsultar \"SELECT COUNT(*) FROM orders;\"\n```\nTenemos un total de 150 órdenes."
```

(Además alucinó "150 órdenes" sin ejecutar nada.)

**Causa:** el `tokenizer.json` del repo `mlx-community/Mistral-Small-...` no
incluye el chat template con `[AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]` y
`[TOOL_CALLS]`. Sin esos tokens, el parser `mistral` no tiene qué parsear.

**Fix:** pasar el template oficial de vLLM:

```bash
--chat-template scripts/tool_chat_template_mistral.jinja
```

(Versión copiada de
[`vllm/examples/tool_chat_template_mistral.jinja`](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_mistral.jinja).)

### 2. NO usar el template "parallel"

vLLM trae dos variantes: `tool_chat_template_mistral.jinja` y
`tool_chat_template_mistral_parallel.jinja`. La docs recomienda la "parallel"
para llamadas en paralelo y mayor robustez.

**Pero** la parallel **inyecta un system prompt en inglés** que pre-concatena
al system prompt del usuario:

> "You are a helpful assistant that can call tools. If you call one or more
> tools, format them in a single JSON array..."

Eso conflictúa con el system prompt del agente que está **en español** y
es estricto sobre formato de respuesta. Síntomas observados:
- Modelo emite saludos hallucinados ("Victor, le voy a ayudarte...").
- Mezcla JSON array como texto con texto coloquial.
- `tool_calls` vuelve a venir vacío.

**Conclusión:** usar la versión no-parallel. Para agentes con system prompt no
trivial (más de una línea), la parallel rompe.

### 3. `temperature: 0.1` es **obligatorio** con MLX 4-bit

Con temperature default (~1.0), el modelo emite tool call estructurado
**solo ~66% del tiempo**. El otro 33% lo manda como texto en `content`.

Tested empíricamente: 3 corridas del mismo request con el chat template
correcto y temperature default:

```
run 1: tool_calls=1 (ok)
run 2: tool_calls=0 (texto)
run 3: tool_calls=1 (ok)
```

Con `temperature=0.1`: 3/3, 5/5, 10/10. Consistente.

PydanticAI no setea temperature por defecto cuando usa `OpenAIChatModel`.
Se agrega manualmente:

```python
agent = Agent(
    model=_build_model(),
    ...,
    model_settings={"temperature": 0.1},
)
```

### 4. PydanticAI: cambiar provider, no protocolo

Inicialmente el runner usaba `OllamaModel + OllamaProvider`. Esos no funcionan
contra un endpoint OpenAI-compat puro como vLLM. La solución:

```python
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

def _build_model():
    if LLM_PROVIDER == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="EMPTY")
        return OpenAIChatModel(model_name=OLLAMA_MODEL, provider=provider)

    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    return OllamaModel(model_name=OLLAMA_MODEL, provider=OllamaProvider(base_url=OLLAMA_BASE_URL))
```

`api_key="EMPTY"` porque vLLM no exige auth pero `OpenAIProvider` rechaza
api_key vacío. Se reutilizan las env vars `OLLAMA_BASE_URL` y `OLLAMA_MODEL`
para no romper scripts existentes (nombres quedan algo abusivos pero es un
PoC).

### 5. Warning benigno del tokenizer

vLLM loguea repetidamente:

```
[mistral_tool_parser.py:135] Non-Mistral tokenizer detected when using a Mistral model
```

El MLX repo trae `tokenizer.json` (HF format) en vez del tokenizer
"tekken" oficial de Mistral. El parser funciona igual cuando el chat template
emite los tokens correctos. Ignorable para este PoC.

## Memoria y boot

- vLLM detecta Metal: **34.4GB total / 18.1GB available** (después del
  reserve del sistema).
- Wired limit configurado a **25GB** automáticamente.
- Primer boot con descarga: 5-10 min (red).
- Boots subsiguientes (modelo cacheado): ~1 min.
- Cold first request: 4-10s (caches de prefill).
- Steady state: 1.8-4.5s por request simple.

## Cómo correr el bench A/B

```bash
# Terminal 1: vLLM (mantener abierto)
bash scripts/start_vllm.sh

# Terminal 2: FastAPI con vLLM
LLM_PROVIDER=openai \
OLLAMA_BASE_URL=http://localhost:8001/v1 \
OLLAMA_MODEL=mlx-community/Mistral-Small-24B-Instruct-2501-4bit \
uv run uvicorn app.server.main:app --port 8000

# Terminal 3: bench
uv run python scripts/bench_concurrency.py --sweep --reps 1
```

Para comparar con Ollama: parar vLLM (libera 25GB wired), cambiar el `.env` a
la sección Ollama, reiniciar FastAPI, rerun bench.

## Cosas que **no** se probaron

- Mistral Small 3.1 (más nuevo, tokenizer tekken nativo, podría no necesitar
  template custom).
- Otros modelos del catálogo vllm-metal (Qwen2.5, Gemma 3, Llama 3.1).
- Concurrencia > 10.
- `--tool-call-parser hermes` o `pythonic` como alternativa.
- vLLM con `--tokenizer_mode mistral` (requiere `mistral-common` package).
- Latencia con `--max-num-batched-tokens` tuneado (default 2048).

## Conclusión para producción

Si el escenario incluye **más de un usuario simultáneo por nodo**, vLLM-metal
es claramente superior. Para single-tenant single-user, Ollama es más simple
de operar (no requiere chat template manual, instala desde Homebrew, no
necesita HF model id).

Lo que cambia respecto al diseño original:

| Antes | Ahora |
|---|---|
| "Ollama por simplicidad, degrada bajo carga" | "vLLM-metal preferido, Ollama como fallback dev" |
| No tool calling estructurado con MLX | Tool calling estructurado validado con chat template + `temperature=0.1` |
| Concurrencia >5 inviable | Concurrencia 10 con P50 ~20s aceptable |
