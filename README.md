# mt-agent-lab

PoC de agente IA multi-tenant 100% local. Objetivo: validar **eficiencia** y
**separación de información** entre tenants antes de portar el patrón a un
proyecto de producción.

## Qué probamos

1. **Aislamiento estricto** — los 5 vectores de fuga del paper SMTA
   (arXiv:2601.06627): KV cache, system prompt, tools, memory, RAG.
2. **Eficiencia bajo carga concurrente** — latencia P50/P95 con 1, 5, 10
   usuarios simultáneos. Validar empíricamente cuándo Ollama degrada.

## Stack

- Python 3.12 + FastAPI + uvicorn
- PydanticAI (agent loop + tools type-safe)
- LLM local: **vLLM-metal** + `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
  (recomendado, 4x más rápido bajo concurrencia — ver
  [`docs/SPIKE-VLLM-METAL.md`](docs/SPIKE-VLLM-METAL.md)).
  Alternativa: Ollama + `mistral-small:24b` (más simple, degrada >conc=1).
- DuckDB (queries sobre parquet)
- Postgres 16 + pgvector (audit + conversations)
- Polars + Faker (seed/multiplicador del dataset)

## Dataset

Base real: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(100K órdenes, 9 tablas). Lo traducimos a español y lo multiplicamos
sintéticamente a 3 tenants:

| Tenant | Órdenes | Período | Notas |
|---|---|---|---|
| `acme` | ~10M | 2018-2025 | Tienda grande, categorías amplias |
| `beta` | ~1M | 2020-2025 | Mediana, foco electrónicos |
| `gamma` | ~50K | últimos 6 meses | Recién onboarded, prueba empty-state |

## Bring-up

```bash
# 1) Dependencias del sistema
brew install uv

# 2) Backend LLM — elegí UNO:

#    Opción A (recomendado): vLLM-metal
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
bash scripts/start_vllm.sh    # corre en :8001, deja la terminal abierta

#    Opción B (más simple, single-user): Ollama
brew install ollama
ollama pull mistral-small:24b
# editar .env: descomentar el bloque "Ollama" y comentar el "vLLM"

# 3) Postgres
docker compose up -d

# 4) Python venv + deps
uv sync

# 5) Server (otra terminal)
uv run uvicorn app.server.main:app --port 8000

# 6) Smoke test
curl http://localhost:8000/health
```

## Estructura

```
app/server/        # FastAPI entrypoint
internal/
  tenants/         # store + middleware (resuelve slug del path)
  agent/           # PydanticAI agent, tools bound al ctx
  store/           # duckdb queries + postgres audit
  audit/           # recorder
scripts/           # seed Olist, init.sql
tests/             # isolation, concurrency, golden set
data/parquet/      # generado por seed, NO versionado
```

## Patrón crítico

**Ninguna tool acepta `tenant_slug` como parámetro.** El slug se cierra en
`ctx.deps` desde el middleware. El modelo no puede pasarlo aunque lo intente.

```python
@agent.tool
async def consultar(ctx: RunContext[AgentDeps], sql: str) -> list[dict]:
    # ctx.deps.tenant_slug viene del request, no del modelo
    path = f"{DATA_ROOT}/{ctx.deps.tenant_slug}/*.parquet"
    ...
```

## Pruebas de aislamiento

`tests/isolation_test.py` corre los 4 ataques del paper SMTA:
- Direct extraction
- Indirect extraction
- Prompt injection
- Session memory leak

Para cada uno: la respuesta NO contiene datos del otro tenant Y el
`agent_audit` registra el intento con `flagged_cross_tenant=TRUE`.

## Documentos

- [`docs/SPIKE-VLLM-METAL.md`](docs/SPIKE-VLLM-METAL.md) — Comparativa vLLM-metal vs Ollama, configuración de tool calling con MLX y bench A/B en M2 Max.

## Estado

PoC en construcción.
