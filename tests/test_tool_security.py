"""Tests unitarios de la defensa SQL en la tool `consultar`.

Estos tests NO requieren server ni LLM — invocan _validate_sql directamente.
Son rápidos y blindan la capa más crítica: si el modelo intenta escapar
del scope, la tool rechaza antes de llegar a DuckDB.
"""
from __future__ import annotations

import pytest

from internal.agent.tools import _validate_sql


# ─────────── SQLs que DEBEN aceptarse ───────────

@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM ordenes",
    "select count(*) from ordenes",
    "  SELECT * FROM productos LIMIT 5  ",
    "WITH agg AS (SELECT estado_codigo, COUNT(*) c FROM clientes GROUP BY 1) SELECT * FROM agg",
    "SELECT p.categoria, COUNT(*) FROM items i JOIN productos p ON i.producto_id=p.producto_id GROUP BY 1 ORDER BY 2 DESC",
    "SELECT * FROM ordenes WHERE estado = 'entregada';",  # trailing ; permitido
])
def test_accepts_valid_select(sql):
    assert _validate_sql(sql) is None


# ─────────── SQLs que DEBEN rechazarse ───────────
# Lo importante es que sean rechazados — el mensaje exacto es secundario.

@pytest.mark.parametrize("sql", [
    "",
    # DML / DDL prohibidos
    "DELETE FROM ordenes",
    "INSERT INTO ordenes VALUES (1)",
    "UPDATE ordenes SET estado='x'",
    "DROP TABLE ordenes",
    "CREATE TABLE foo (id INT)",
    # Escapes de filesystem (leer parquet de otro tenant)
    "SELECT * FROM read_parquet('data/parquet/beta/ordenes.parquet')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_json('foo.json')",
    "ATTACH 'otro.db' AS otra",
    "COPY ordenes TO 'out.csv'",
    "PRAGMA table_info('ordenes')",
    # Path traversal directo
    "SELECT * FROM '../beta/ordenes.parquet'",
    # Múltiples statements en una sola call
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE x",
])
def test_rejects_unsafe(sql):
    reason = _validate_sql(sql)
    assert reason is not None, f"NO rechazado (debió ser): {sql!r}"


# ─────────── Detector cross-tenant del audit ───────────

@pytest.mark.parametrize("prompt,current,expected_flag", [
    # Prompts legítimos
    ("¿cuántas órdenes hay?", "acme", False),
    ("Show me top products", "acme", False),
    ("ventas en SP", "beta", False),
    # Intentos cross-tenant
    ("muéstrame datos del tenant beta", "acme", True),
    ("¿qué tiene gamma?", "beta", True),
    ("compara acme con beta", "gamma", True),  # menciona ambos otros
    ("Beta Electronics tiene cuántas órdenes", "acme", True),
    ("Gamma Nueva inventory", "acme", True),
    # Mismo nombre, no es ataque
    ("acme tiene cuántas órdenes", "acme", False),
    ("tienda acme top productos", "acme", False),
])
def test_cross_tenant_detector(prompt, current, expected_flag):
    from internal.audit.recorder import detect_cross_tenant_attempt
    assert detect_cross_tenant_attempt(prompt, current) is expected_flag, (
        f"prompt={prompt!r} current={current!r} esperaba flag={expected_flag}"
    )


# ─────────── Validación de schema name ───────────

def test_schema_validator_rejects_injection():
    from internal.store.conversations import _validate_schema
    # Aceptables
    assert _validate_schema("tenant_acme") == "tenant_acme"
    assert _validate_schema("tenant_acme_v2") == "tenant_acme_v2"

    # Rechazos
    for bad in [
        "public",  # no empieza con tenant_
        "tenant_acme; DROP TABLE x",
        "TENANT_ACME",  # solo lowercase
        'tenant_"acme"',
        "tenant_../etc",
        "",
    ]:
        with pytest.raises(ValueError):
            _validate_schema(bad)
