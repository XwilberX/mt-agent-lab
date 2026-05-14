"""Entry point del API mt-agent-lab.

Endpoints:
  GET  /health             — postgres + ollama status
  GET  /tenants            — listado del catálogo
  POST /t/{slug}/ask       — agente real (PydanticAI + Ollama + DuckDB)
  GET  /audit?tenant=&n=   — últimas N filas del audit log
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from internal.audit.recorder import record as audit_record
from internal.agent.runner import run_agent
from internal.store.conversations import load_messages, save_messages
from internal.tenants.store import TenantStore

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mapping de colores Tailwind por tenant — reforzar aislamiento visualmente
TENANT_COLORS = {"acme": "cyan", "beta": "amber", "gamma": "pink"}
TENANT_DESCS = {
    "acme": "Tienda grande, 10M órdenes, todas las categorías.",
    "beta": "Foco electrónicos, ~1M órdenes.",
    "gamma": "Cliente nuevo, ~50K órdenes en los últimos meses.",
}
SUGGESTIONS = {
    "acme": [
        "¿Cuántas órdenes totales tenemos?",
        "Top 3 categorías por items vendidos",
        "¿Cuántos clientes únicos hay en SP?",
        "Muéstrame datos del tenant beta",
    ],
    "beta": [
        "¿Cuántas órdenes totales tenemos?",
        "Promedio de calificación de reseñas",
        "¿Cuántos vendedores activos hay?",
        "¿Qué tiene gamma?",
    ],
    "gamma": [
        "¿Cuántas órdenes tenemos?",
        "Top 5 productos más vendidos",
        "¿Cuál es el ticket promedio?",
        "Compara con acme",
    ],
}

load_dotenv()

log = structlog.get_logger()

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    app.state.pg = pool
    app.state.tenant_store = TenantStore(pool)
    log.info("server.startup", model=OLLAMA_MODEL, ollama=OLLAMA_BASE_URL)
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(title="mt-agent-lab", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    tenant_slug: str
    tool_calls: list[dict[str, Any]] = []
    duration_ms: int
    audit_id: int


@app.get("/health")
async def health(request: Request) -> dict:
    pool: asyncpg.Pool = request.app.state.pg
    async with pool.acquire() as conn:
        pg_ok = await conn.fetchval("SELECT 1") == 1
    return {
        "status": "ok",
        "postgres": pg_ok,
        "model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }


@app.get("/tenants")
async def list_tenants(request: Request) -> list[dict]:
    pool: asyncpg.Pool = request.app.state.pg
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT slug, name, enabled FROM public.tenants ORDER BY slug"
        )
    return [dict(r) for r in rows]


@app.post("/t/{slug}/ask", response_model=AskResponse)
async def ask(
    slug: str,
    body: AskRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> AskResponse:
    user_id = x_user_id or "anon"
    pool: asyncpg.Pool = request.app.state.pg
    store: TenantStore = request.app.state.tenant_store

    tenant = await store.resolve(slug)
    if tenant is None:
        raise HTTPException(404, f"tenant '{slug}' no existe")
    if not tenant.enabled:
        raise HTTPException(403, f"tenant '{slug}' deshabilitado")

    # Si hay session_id, cargar history del schema del tenant
    history_json: bytes | None = None
    if body.session_id:
        history_json = await load_messages(pool, tenant.schema_name, body.session_id)

    err: str | None = None
    answer: str = ""
    tool_calls: list[dict[str, Any]] = []
    duration_ms = 0
    all_messages_json: bytes | None = None
    try:
        result = await run_agent(
            tenant_slug=tenant.slug,
            tenant_name=tenant.name,
            user_id=user_id,
            question=body.question,
            message_history_json=history_json,
        )
        answer = result.answer
        tool_calls = result.tool_calls
        duration_ms = result.duration_ms
        all_messages_json = result.all_messages_json
    except Exception as e:
        err = repr(e)
        log.exception("agent.run.failed", tenant=slug, error=err)

    # Si hay session_id y el run fue OK, persistir messages en el schema del tenant
    if body.session_id and all_messages_json is not None:
        await save_messages(
            pool, tenant.schema_name, body.session_id, user_id, all_messages_json
        )

    audit_id = await audit_record(
        pool,
        tenant_slug=tenant.slug,
        user_id=user_id,
        session_id=body.session_id,
        prompt=body.question,
        response=answer if not err else None,
        tool_calls=tool_calls,
        duration_ms=duration_ms,
        error=err,
    )

    if err:
        raise HTTPException(500, f"agent error: {err}")

    return AskResponse(
        answer=answer,
        tenant_slug=tenant.slug,
        tool_calls=tool_calls,
        duration_ms=duration_ms,
        audit_id=audit_id,
    )


# ───────── UI (HTML) ─────────

@app.get("/", response_class=HTMLResponse)
async def ui_home(request: Request) -> HTMLResponse:
    pool: asyncpg.Pool = request.app.state.pg
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT slug, name, enabled FROM public.tenants ORDER BY slug"
        )
    return templates.TemplateResponse(request, "home.html", {
        "tenants": [dict(r) for r in rows],
        "tenant_colors": TENANT_COLORS,
        "tenant_descs": TENANT_DESCS,
        "model": OLLAMA_MODEL,
    })


@app.get("/t/{slug}", response_class=HTMLResponse)
async def ui_chat(slug: str, request: Request) -> HTMLResponse:
    store: TenantStore = request.app.state.tenant_store
    tenant = await store.resolve(slug)
    if tenant is None:
        raise HTTPException(404, f"tenant '{slug}' no existe")
    return templates.TemplateResponse(request, "chat.html", {
        "tenant": tenant,
        "color": TENANT_COLORS.get(slug, "zinc"),
        "session_id": uuid.uuid4().hex,
        "model": OLLAMA_MODEL,
        "suggestions": SUGGESTIONS.get(slug, []),
    })


@app.post("/t/{slug}/ui-ask", response_class=HTMLResponse)
async def ui_ask(
    slug: str,
    request: Request,
    question: str = Form(...),
    session_id: str = Form(...),
) -> HTMLResponse:
    """Versión HTMX del ask: devuelve fragmento HTML del intercambio."""
    pool: asyncpg.Pool = request.app.state.pg
    store: TenantStore = request.app.state.tenant_store

    tenant = await store.resolve(slug)
    if tenant is None or not tenant.enabled:
        return templates.TemplateResponse(request, "_exchange.html", {
            "question": question,
            "tenant_name": slug,
            "color": "zinc",
            "error": "tenant inválido",
            "answer": "", "tool_calls": [], "duration_ms": 0, "flagged": False,
        }, status_code=400)

    history_json = await load_messages(pool, tenant.schema_name, session_id)

    err: str | None = None
    answer = ""
    tool_calls: list[dict[str, Any]] = []
    duration_ms = 0
    all_messages_json: bytes | None = None
    try:
        result = await run_agent(
            tenant_slug=tenant.slug,
            tenant_name=tenant.name,
            user_id="ui",
            question=question,
            message_history_json=history_json,
        )
        answer = result.answer
        tool_calls = result.tool_calls
        duration_ms = result.duration_ms
        all_messages_json = result.all_messages_json
    except Exception as e:
        err = repr(e)
        log.exception("agent.run.failed", tenant=slug, error=err)

    if all_messages_json is not None:
        await save_messages(pool, tenant.schema_name, session_id, "ui", all_messages_json)

    audit_id = await audit_record(
        pool,
        tenant_slug=tenant.slug, user_id="ui", session_id=session_id,
        prompt=question, response=answer if not err else None,
        tool_calls=tool_calls, duration_ms=duration_ms, error=err,
    )

    # Recuperar el flag desde el audit recién insertado
    async with pool.acquire() as conn:
        flagged = await conn.fetchval(
            "SELECT flagged_cross_tenant FROM public.agent_audit WHERE id = $1",
            audit_id,
        )

    return templates.TemplateResponse(request, "_exchange.html", {
        "question": question,
        "tenant_name": tenant.name,
        "color": TENANT_COLORS.get(slug, "zinc"),
        "error": err,
        "answer": answer,
        "tool_calls": tool_calls,
        "duration_ms": duration_ms,
        "flagged": flagged,
    })


@app.get("/audit")
async def audit(
    request: Request,
    tenant: str | None = None,
    n: int = 20,
    flagged_only: bool = False,
) -> list[dict]:
    pool: asyncpg.Pool = request.app.state.pg
    n = max(1, min(n, 200))
    where = []
    params: list[Any] = []
    if tenant:
        where.append(f"tenant_slug = ${len(params)+1}")
        params.append(tenant)
    if flagged_only:
        where.append("flagged_cross_tenant = TRUE")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT id, tenant_slug, user_id, session_id, prompt, response,
               tool_calls, duration_ms, flagged_cross_tenant, error, created_at
        FROM public.agent_audit
        {where_sql}
        ORDER BY id DESC
        LIMIT {n}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [
        {**dict(r), "created_at": r["created_at"].isoformat()}
        for r in rows
    ]
