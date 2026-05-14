"""Audit log: cada request al agente queda registrado en public.agent_audit.

Detección de intento cross-tenant: si el prompt menciona el slug o nombre
de OTRO tenant, se marca flagged_cross_tenant=TRUE para que aparezca en
revisión.
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg


# Catálogo de slugs y nombres conocidos. Si aparecen en el prompt de un
# tenant que NO es el actual, levantamos la bandera.
KNOWN_TENANT_TERMS: dict[str, tuple[str, ...]] = {
    "acme":  ("acme", "tienda acme"),
    "beta":  ("beta", "beta electronics", "betaelectronics"),
    "gamma": ("gamma", "gamma nueva"),
}


def detect_cross_tenant_attempt(prompt: str, current_slug: str) -> bool:
    """Heurística simple: ¿el prompt menciona OTRO tenant?

    No es perfecta (un prompt legítimo podría decir "no soy acme"), pero
    para el demo cumple el rol de alarma para revisión manual.
    """
    p = prompt.lower()
    for slug, terms in KNOWN_TENANT_TERMS.items():
        if slug == current_slug:
            continue
        for term in terms:
            if term in p:
                return True
    return False


async def record(
    pool: asyncpg.Pool,
    *,
    tenant_slug: str,
    user_id: str,
    session_id: str | None,
    prompt: str,
    response: str | None,
    tool_calls: list[dict[str, Any]],
    duration_ms: int | None,
    error: str | None = None,
) -> int:
    """Inserta un row en public.agent_audit. Devuelve el id generado."""
    flagged = detect_cross_tenant_attempt(prompt, tenant_slug)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO public.agent_audit
                (tenant_slug, user_id, session_id, prompt, response,
                 tool_calls, duration_ms, flagged_cross_tenant, error)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
            RETURNING id
            """,
            tenant_slug, user_id, session_id, prompt, response,
            json.dumps(tool_calls, default=str),
            duration_ms, flagged, error,
        )
    return row["id"]
