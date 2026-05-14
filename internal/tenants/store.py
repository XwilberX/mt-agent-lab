"""Resolución de tenant: slug del path → struct Tenant.

Lee public.tenants en Postgres. Cachea en memoria con TTL para evitar hit a la
DB en cada request.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import asyncpg


@dataclass(frozen=True)
class Tenant:
    slug: str
    name: str
    schema_name: str
    enabled: bool


class TenantStore:
    """Cache de tenants con TTL corto. Thread-safe via asyncio.Lock."""

    def __init__(self, pool: asyncpg.Pool, ttl_seconds: int = 60) -> None:
        self._pool = pool
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, Tenant | None]] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, slug: str) -> Tenant | None:
        now = time.monotonic()
        cached = self._cache.get(slug)
        if cached and (now - cached[0]) < self._ttl:
            return cached[1]

        async with self._lock:
            # double-check tras lock
            cached = self._cache.get(slug)
            if cached and (now - cached[0]) < self._ttl:
                return cached[1]

            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT slug, name, schema_name, enabled "
                    "FROM public.tenants WHERE slug = $1",
                    slug,
                )
            t = (
                Tenant(slug=row["slug"], name=row["name"],
                       schema_name=row["schema_name"], enabled=row["enabled"])
                if row else None
            )
            self._cache[slug] = (time.monotonic(), t)
            return t
