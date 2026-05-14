"""Tools del agente. Las 3 son bound al ctx del request.

PATRÓN CRÍTICO:
- Ninguna tool acepta `tenant_slug` como parámetro.
- El slug se toma de `ctx.deps.tenant_slug` (closure desde el request).
- El modelo NUNCA puede inyectar otro slug aunque lo intente.
"""
from __future__ import annotations

import re
from typing import Any

import duckdb
from pydantic_ai import RunContext

from internal.agent.deps import AgentDeps

# Tablas a las que el agente puede acceder. Si una pregunta del usuario menciona
# una "tabla" fuera de esta lista, la tool falla.
TABLAS_PERMITIDAS: tuple[str, ...] = (
    "ordenes", "items", "productos", "clientes",
    "vendedores", "pagos", "resenas", "categorias",
)

# Tokens prohibidos en SQL: cualquier cosa que pueda escapar del scope
_SQL_FORBIDDEN = (
    "read_parquet", "read_csv", "read_json", "attach", "copy",
    "pragma", "load ", "install ",
    "create ", "drop ", "delete ", "update ", "insert ", "alter ",
    "truncate", "merge ", "vacuum",
    # Escape de path: si el modelo intenta leer otro tenant
    "/parquet/", "data/parquet", "..",
)

# Cache de conexiones DuckDB por tenant (read-only, vistas fijas).
# Una vez creada, sirve a varios requests del mismo tenant.
_duckdb_conns: dict[str, duckdb.DuckDBPyConnection] = {}


def _get_duckdb(deps: AgentDeps) -> duckdb.DuckDBPyConnection:
    """Conn DuckDB en memoria con vistas pre-creadas hacia los parquets del
    tenant. El modelo solo ve nombres `ordenes`, `items`, etc. — nunca paths."""
    slug = deps.tenant_slug
    if slug in _duckdb_conns:
        return _duckdb_conns[slug]

    con = duckdb.connect(":memory:")
    base = deps.parquet_dir
    if not base.exists():
        raise RuntimeError(f"data dir no existe: {base}")
    for tabla in TABLAS_PERMITIDAS:
        path = base / f"{tabla}.parquet"
        if not path.exists():
            # algunas tablas pueden no estar para todos los tenants — skip
            continue
        # CREATE VIEW con path quoted; el path es de confianza (no input del usuario)
        con.execute(f"CREATE VIEW {tabla} AS SELECT * FROM read_parquet('{path}')")
    _duckdb_conns[slug] = con
    return con


def _validate_sql(sql: str) -> str | None:
    """Devuelve None si OK, o un string con motivo de rechazo."""
    s = sql.strip().lower()
    if not s:
        return "SQL vacío"
    if not s.startswith(("select", "with")):
        return "Solo SELECT (o WITH ... SELECT) está permitido"
    for tok in _SQL_FORBIDDEN:
        if tok in s:
            return f"Token prohibido en SQL: '{tok.strip()}'"
    # statements múltiples
    # un ';' al final es OK; en el medio no.
    body = s.rstrip(";").strip()
    if ";" in body:
        return "Solo se permite un SELECT por consulta"
    return None


# ───────── Tools registrables al Agent ─────────

async def listar_tablas(ctx: RunContext[AgentDeps]) -> list[str]:
    """Devuelve las tablas disponibles para este tenant."""
    con = _get_duckdb(ctx.deps)
    rows = con.execute(
        "SELECT view_name FROM duckdb_views() WHERE schema_name = 'main' ORDER BY view_name"
    ).fetchall()
    return [r[0] for r in rows]


async def describir_tabla(ctx: RunContext[AgentDeps], tabla: str) -> dict[str, Any]:
    """Devuelve columnas con su tipo + 3 filas de muestra de una tabla.

    Args:
        tabla: nombre de tabla (debe estar en la whitelist).
    """
    if tabla not in TABLAS_PERMITIDAS:
        return {"error": f"Tabla '{tabla}' no permitida. "
                          f"Tablas válidas: {list(TABLAS_PERMITIDAS)}"}
    con = _get_duckdb(ctx.deps)
    try:
        schema = con.execute(f"DESCRIBE {tabla}").fetchall()
        muestra = con.execute(f"SELECT * FROM {tabla} LIMIT 3").fetchall()
        cols = [c[0] for c in con.description]
        return {
            "columnas": [{"nombre": r[0], "tipo": str(r[1])} for r in schema],
            "muestra": [dict(zip(cols, row)) for row in muestra],
            "filas_total": con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0],
        }
    except Exception as e:
        return {"error": str(e)}


async def consultar(ctx: RunContext[AgentDeps], sql: str) -> dict[str, Any]:
    """Ejecuta una consulta SELECT contra las tablas del tenant.

    El resultado se trunca a 100 filas. Solo SELECT está permitido.
    No se pueden usar funciones que lean archivos arbitrarios.

    Args:
        sql: consulta SQL (debe empezar con SELECT o WITH).
    """
    reason = _validate_sql(sql)
    if reason:
        return {"error": f"SQL rechazado: {reason}"}

    con = _get_duckdb(ctx.deps)
    try:
        rows = con.execute(sql).fetchmany(100)
        cols = [c[0] for c in con.description] if con.description else []
        truncated = len(rows) == 100
        # cast valores no-json-serializables (datetime, Decimal)
        clean_rows = [
            {c: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
             for c, v in zip(cols, row)}
            for row in rows
        ]
        return {
            "columnas": cols,
            "filas": clean_rows,
            "n_filas": len(clean_rows),
            "truncado_a_100": truncated,
        }
    except Exception as e:
        return {"error": str(e)}
