"""Fixtures comunes para los tests del lab.

Estrategia: los tests pegan al server LIVE en localhost:8000 (que tiene que
estar corriendo). Esto nos asegura que probamos el binario real, no mocks.

Requisitos:
- `uv run uvicorn cmd.server.main:app --port 8000` corriendo
- Postgres del docker-compose arriba
- Ollama con qwen2.5:14b cargado
- Tenants seeded (acme/beta/gamma)
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

import asyncpg
import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("LAB_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.environ["DATABASE_URL"]


@pytest_asyncio.fixture
async def pg() -> AsyncIterator[asyncpg.Pool]:
    """Pool por test. asyncpg + pytest-asyncio crean conflictos de loop si
    es session-scoped, así que pagamos el costo de un pool por test (es
    barato — son ~10 tests)."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=300) as c:
        # smoke: server arriba
        r = await c.get("/health")
        assert r.status_code == 200, f"server caído en {BASE_URL}"
        yield c


@pytest_asyncio.fixture(autouse=True)
async def _clean_audit_per_test(pg: asyncpg.Pool):
    """Limpia el audit log antes de cada test para que las aserciones sean
    deterministas sobre el último request."""
    async with pg.acquire() as conn:
        await conn.execute("TRUNCATE public.agent_audit RESTART IDENTITY")
    yield


async def last_audit(pg: asyncpg.Pool) -> dict | None:
    async with pg.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM public.agent_audit ORDER BY id DESC LIMIT 1"
        )
    return dict(row) if row else None


async def clear_conversations(pg: asyncpg.Pool, schema: str) -> None:
    async with pg.acquire() as conn:
        await conn.execute(f'TRUNCATE "{schema}".conversations')
