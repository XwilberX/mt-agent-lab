"""Persistencia de message_history por sesión, scoped al schema del tenant.

Aislamiento por construcción: cada tenant tiene su propio schema
(tenant_<slug>) y la tabla `conversations` vive ahí. No hay forma de leer
conversaciones de otro tenant aunque la query SQL esté mal armada — el
schema name está cerrado desde el resolve del tenant.
"""
from __future__ import annotations

import re

import asyncpg


_SCHEMA_RE = re.compile(r"^tenant_[a-z0-9_]+$")


def _validate_schema(schema: str) -> str:
    """Defensa en profundidad: schema name solo puede ser tenant_<slug>."""
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"schema name inválido: {schema!r}")
    return schema


async def load_messages(
    pool: asyncpg.Pool, schema: str, session_id: str
) -> bytes | None:
    """Devuelve el JSON crudo de messages o None si no existe la sesión.

    Devolvemos bytes para que pydantic-ai lo parsee con su propio
    TypeAdapter — preserva el tipado interno de PydanticAI sin que tengamos
    que rearmar los objects.
    """
    schema = _validate_schema(schema)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f'SELECT messages FROM "{schema}".conversations WHERE session_id = $1',
            session_id,
        )
    return row["messages"].encode() if row else None


async def save_messages(
    pool: asyncpg.Pool,
    schema: str,
    session_id: str,
    user_id: str,
    messages_json: bytes,
) -> None:
    """Upsert de toda la conversación. `messages_json` viene de
    `result.all_messages_json()` de PydanticAI."""
    schema = _validate_schema(schema)
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO "{schema}".conversations
                (session_id, user_id, messages, updated_at)
            VALUES ($1, $2, $3::jsonb, NOW())
            ON CONFLICT (session_id) DO UPDATE
              SET messages   = EXCLUDED.messages,
                  updated_at = NOW()
            """,
            session_id, user_id, messages_json.decode(),
        )
