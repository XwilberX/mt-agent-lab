#!/usr/bin/env python3
"""Registra tenants en Postgres + crea schemas y tabla de conversations.

Idempotente: corre cuantas veces quieras.

Uso:
    uv run python scripts/seed_tenants.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


TENANTS = [
    ("acme", "Tienda Acme"),
    ("beta", "Beta Electronics"),
    ("gamma", "Gamma Nueva"),
]


def schema_for(slug: str) -> str:
    """Postgres no permite guion en schema name."""
    return "tenant_" + slug.replace("-", "_")


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL no está en el entorno (revisá .env)", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(dsn)
    try:
        for slug, name in TENANTS:
            schema = schema_for(slug)

            # 1) insertar en catálogo (idempotente)
            await conn.execute(
                """
                INSERT INTO public.tenants (slug, name, schema_name, enabled)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
                """,
                slug, name, schema,
            )

            # 2) crear schema tenant_<slug>
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

            # 3) tabla conversations en el schema del tenant
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{schema}".conversations (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            print(f"  ✓ {slug:8} → schema {schema!r}, tabla conversations lista")

        # listar lo que quedó
        rows = await conn.fetch(
            "SELECT slug, name, schema_name, enabled FROM public.tenants ORDER BY slug"
        )
        print("\nCatálogo final:")
        for r in rows:
            mark = "✓" if r["enabled"] else "✗"
            print(f"  {mark} {r['slug']:8} {r['name']:25} schema={r['schema_name']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
